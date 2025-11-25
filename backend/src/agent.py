import logging
import json
import os
import asyncio
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

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
# CONTENT SETUP
# ======================================================

CONTENT_FILE = "solar_system.json"

DEFAULT_CONTENT = [
    {
        "id": "sun",
        "title": "The Sun",
        "summary": "The Sun is a massive ball of hot plasma at the center of our solar system. It provides light and heat necessary for life on Earth.",
        "sample_question": "Why is the Sun considered the primary source of energy for the solar system?"
    },
    {
        "id": "planets",
        "title": "Planets",
        "summary": "Planets are large celestial bodies orbiting the Sun. They are classified into terrestrial planets and gas giants based on their composition.",
        "sample_question": "What are the main differences between terrestrial planets and gas giants?"
    },
    {
        "id": "moon",
        "title": "Earth's Moon",
        "summary": "The Moon is Earth's only natural satellite. It affects ocean tides and is the only celestial body visited by humans.",
        "sample_question": "What causes the different phases of the Moon?"
    },
    {
        "id": "asteroids",
        "title": "Asteroids",
        "summary": "Asteroids are small rocky bodies that orbit the Sun, mostly found in the asteroid belt between Mars and Jupiter.",
        "sample_question": "Where is the asteroid belt located and what is found there?"
    }
]

def load_content():
    """Load or create solar system content."""
    try:
        path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)

        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONTENT, f, indent=4)

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return []

COURSE_CONTENT = load_content()

# ======================================================
# STATE
# ======================================================

@dataclass
class TutorState:
    current_topic_id: str | None = None
    current_topic_data: dict | None = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"

    def set_topic(self, topic_id: str):
        topic = next((t for t in COURSE_CONTENT if t["id"] == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            return True
        return False

@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None

# ======================================================
# TOOLS
# ======================================================

@function_tool
async def select_topic(
    ctx: RunContext[Userdata],
    topic_id: Annotated[str, Field(description="ID of the topic")]
):
    state = ctx.userdata.tutor_state
    ok = state.set_topic(topic_id.lower())

    if ok:
        return f"Topic set to {state.current_topic_data['title']}. Ask the user whether to Learn, Quiz, or Teach Back."
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic not found. Available topics: {available}"

@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata],
    mode: Annotated[str, Field(description="Mode: learn, quiz, teach_back")]
):
    state = ctx.userdata.tutor_state
    state.mode = mode.lower()
    session = ctx.userdata.agent_session

    if session:
        if state.mode == "learn":
            session.tts.update_options(voice="en-US-matthew", style="Promo")
            instruction = f"Explain: {state.current_topic_data['summary']}"

        elif state.mode == "quiz":
            session.tts.update_options(voice="en-US-alicia", style="Conversational")
            instruction = f"Ask: {state.current_topic_data['sample_question']}"

        elif state.mode == "teach_back":
            session.tts.update_options(voice="en-US-ken", style="Promo")
            instruction = "Ask the user to explain the topic back."

        else:
            return "Invalid mode."
    else:
        instruction = "Voice session not found."

    return f"Switched to {state.mode} mode. {instruction}"

@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="User's explanation")]
):
    return "Analyze the explanation, give a score out of 10, and correct mistakes."

# ======================================================
# AGENT
# ======================================================

class TutorAgent(Agent):
    def __init__(self):
        topics = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])

        super().__init__(
            instructions=f"""
            You are a Astrology Tutor.

            Topics: {topics}

            Modes:
            - Learn: Explain concepts.
            - Quiz: Ask questions.
            - Teach Back: User explains; you evaluate.
            """,
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )

# ======================================================
# ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    userdata = Userdata(tutor_state=TutorState())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Promo", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    userdata.agent_session = session

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

