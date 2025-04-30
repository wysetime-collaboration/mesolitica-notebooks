#ref: https://openai.github.io/openai-agents-python/voice/quickstart/

import asyncio
import random

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from agents import (
    Agent,
    function_tool,
    set_tracing_disabled,
)
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    SingleAgentWorkflowCallbacks,
    VoicePipeline,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
import os

from util import AudioPlayer, record_audio

"""
This is a simple example that uses a recorded audio buffer. Run it via:
`python -m examples.voice.static.main`

1. You can record an audio clip in the terminal.
2. The pipeline automatically transcribes the audio.
3. The agent workflow is a simple one that starts at the Assistant agent.
4. The output of the agent is streamed to the audio player.

Try examples like:
- Tell me a joke (will respond with a joke)
- What's the weather in Tokyo? (will call the `get_weather` tool and then speak)
- Hola, como estas? (will handoff to the spanish agent)
"""


load_dotenv()

@function_tool
def get_english_name() -> str:
    """Get the english name of the user given the language."""
    return "Leonard"

@function_tool
def get_chinese_name() -> str:
    """Get the chinese name of the user."""
    return "羅淨智"

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


malay_agent = Agent(
    name="Malay",
    handoff_description="A Bahasa Malaysia speaking agent working in a bank as customer service agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Bahasa Malaysia.",
    ),
    model="gpt-4o-mini",
)

chinese_agent = Agent(
    name="Chinese",
    handoff_description="A Chinese speaking agent working in a bank as customer service agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Chinese.",
    ),
    model="gpt-4o-mini",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in English by default. If the user speaks in Malay, handoff to the Malay agent. If the user speaks in Chinese, handoff to the Chinese agent.",
    ),
    model="gpt-4o-mini",
    handoffs=[malay_agent, chinese_agent],
    tools=[get_weather, get_english_name, get_chinese_name],
)


class WorkflowCallbacks(SingleAgentWorkflowCallbacks):
    def on_run(self, workflow: SingleAgentVoiceWorkflow, transcription: str) -> None:
        print(f"[debug] on_run called with transcription: {transcription}")


async def main():
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent, callbacks=WorkflowCallbacks()),
    )

    audio_input = AudioInput(buffer=record_audio())

    result = await pipeline.run(audio_input)

    with AudioPlayer() as player:
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.add_audio(event.data)
                print("Received audio")
            elif event.type == "voice_stream_event_lifecycle":
                print(f"Received lifecycle event: {event.event}")

        # Add 1 second of silence to the end of the stream to avoid cutting off the last audio.
        player.add_audio(np.zeros(24000 * 1, dtype=np.int16))

if __name__ == "__main__":
    asyncio.run(main())