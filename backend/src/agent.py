# ======================================================
#  AI SALES DEVELOPMENT REP (SDR)
#  Auto-Lead Capture Agent
#  Features: FAQ Retrieval, Lead Qualification, JSON Database
# ======================================================

import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal, Optional, List
from dataclasses import dataclass, asdict

print("\n" + "-" * 50)
print("AI SDR AGENT INITIALIZING")
print("agent.py loaded successfully")
print("-" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 1. KNOWLEDGE BASE
# ======================================================

FAQ_FILE = "stored_faq.json"
LEADS_FILE = "lead_db.json"

DEFAULT_FAQ = [
    {
        "question": "What do you offer?",
        "answer": "Murf AI is an advanced AI platform that allows you to create natural-sounding voiceovers, build voice agents, and integrate speech technology into your applications with ease."
    },
    {
        "question": "How much do your courses cost?",
        "answer": "Murf AI offers a free tier for experimenting with voice agents and voiceovers, with paid plans available for higher usage, premium voices, and advanced features."
    },
    {
        "question": "Do you offer free content?",
        "answer": "Yes, we provide some free tutorials and learning materials."
    },
    {
        "question": "Do you provide consulting?",
        "answer": "Yes, consulting services are available depending on the project scope."
    }
]

def load_knowledge_base():
    """Generates FAQ file if missing, then loads it."""
    try:
        path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_FAQ, f, indent=4)
        with open(path, "r", encoding='utf-8') as f:
            return json.dumps(json.load(f))
    except Exception as e:
        print(f"Error loading FAQ: {e}")
        return ""

STORE_FAQ_TEXT = load_knowledge_base()

# ======================================================
# 2. LEAD DATA STRUCTURE
# ======================================================

@dataclass
class LeadProfile:
    name: str | None = None
    company: str | None = None
    email: str | None = None
    role: str | None = None
    use_case: str | None = None
    team_size: str | None = None
    timeline: str | None = None

    def is_qualified(self):
        return all([self.name, self.email, self.use_case])

@dataclass
class Userdata:
    lead_profile: LeadProfile

# ======================================================
# 3. SDR TOOLS
# ======================================================

@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's name")] = None,
    company: Annotated[Optional[str], Field(description="Customer's company name")] = None,
    email: Annotated[Optional[str], Field(description="Customer's email address")] = None,
    role: Annotated[Optional[str], Field(description="Customer's job title")] = None,
    use_case: Annotated[Optional[str], Field(description="Customer's use case or requirement")] = None,
    team_size: Annotated[Optional[str], Field(description="Team size")] = None,
    timeline: Annotated[Optional[str], Field(description="When they plan to start")] = None,
) -> str:
    """Captures lead details provided by the user during conversation."""
    profile = ctx.userdata.lead_profile

    if name: profile.name = name
    if company: profile.company = company
    if email: profile.email = email
    if role: profile.role = role
    if use_case: profile.use_case = use_case
    if team_size: profile.team_size = team_size
    if timeline: profile.timeline = timeline

    print(f"Updating lead profile: {profile}")
    return "Lead profile updated."

@function_tool
async def submit_lead_and_end(
    ctx: RunContext[Userdata],
) -> str:
    """Saves the lead to the database and ends the session."""
    profile = ctx.userdata.lead_profile

    db_path = os.path.join(os.path.dirname(__file__), LEADS_FILE)

    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()

    existing_data = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                existing_data = json.load(f)
        except:
            pass

    existing_data.append(entry)

    with open(db_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Lead saved to {LEADS_FILE}")
    return (
        f"Lead saved. Summary: "
        f"Name: {profile.name}, Use Case: {profile.use_case}. "
        f"We will contact you at {profile.email}. Goodbye!"
    )

# ======================================================
# 4. AGENT DEFINITION
# ======================================================

class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
            You are an AI Sales Development Representative (SDR).

            KNOWLEDGE BASE:
            {STORE_FAQ_TEXT}

            GOALS:
            - Answer questions using the FAQ.
            - Gradually collect the user's details:
                * Name
                * Company
                * Email
                * Role
                * Use Case
                * Team size
                * Timeline

            BEHAVIOR:
            - Be conversational, natural, and helpful.
            - When the user provides information, immediately call update_lead_profile.
            - At the end of the conversation, call submit_lead_and_end.

            RESTRICTIONS:
            - If unsure about something, say: "I can confirm this by email."
            """,
            tools=[update_lead_profile, submit_lead_and_end],
        )

# ======================================================
# ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "-" * 25)
    print("Starting SDR Session")

    userdata = Userdata(lead_profile=LeadProfile())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie",
            style="Promo",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        )
    )
