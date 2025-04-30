"""
Voice-to-Voice Chat with Gemini 2.0 Flash exp

This script creates a real-time voice conversation with Gemini AI. It captures your voice 
from the microphone, sends it to Gemini, and plays Gemini's voice responses through speakers.

Dependencies:
- google-genai
- pyaudio

Setup:
1. Install dependencies: pip install google-genai pyaudio
2. For Google AI Studio; Set GOOGLE_API_KEY environment variable with your API key and set use_vertexai to False in line 50.
   If you are using VertexAI check provide PROJECT_ID in line 51 and set use_vertexai to True in line 50.

Usage:
1. Run the script: python google_live_multimodel.py
2. Start speaking into your microphone
3. Listen to Gemini's responses
4. Press Ctrl+C to exit

Note: Headphones are recommended to prevent audio feedback

Ref: https://ai.google.dev/gemini-api/docs/live
"""

import asyncio
import os
import sys
import traceback
import pyaudio
from google import genai
from google.genai.types import LiveConnectConfig, HttpOptions, Modality, SpeechConfig
import os
from dotenv import load_dotenv

load_dotenv()

# check if  Python >= 3.11
if sys.version_info < (3, 11, 0):
    print("Error: This script requires Python 3.11 or newer.")
    print("Python 3.11 introduced asyncio.TaskGroup, which this script uses")
    print("for concurrent task management with proper error handling.")
    print("Please upgrade your Python installation.")
    sys.exit(1)

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000     # Microphone input rate
RECEIVE_SAMPLE_RATE = 24000  # Gemini output rate
CHUNK_SIZE = 1024

# Choose if you want to use VertexAI or Gemini Developer API
use_vertexai = False  # Set to True for Vertex AI, False for Gemini Developer API (Google AI Studio API_KEY)
PROJECT_ID = 'set-me-up'  # set this value with proper Project ID if you plan to use Vertex AI

# Configure API client and model based on selection
if use_vertexai:
    # Vertex AI configuration
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location='us-central1',
        http_options=HttpOptions(api_version="v1beta1")
    )
    MODEL = "gemini-2.0-flash-live-001"  # Just the model name for Vertex AI
    CONFIG = LiveConnectConfig(response_modalities=[Modality.AUDIO])
else:
    # Gemini Developer API configuration
    # Make sure you have 'GOOGLE_API_KEY' variable set with API KEY or pass the api_key='...' in genai.Client()

    client = genai.Client(
        http_options={"api_version": "v1alpha"}
    )
    MODEL = "models/gemini-2.0-flash-live-001"
    CONFIG = LiveConnectConfig(response_modalities=[Modality.AUDIO], system_instruction="Anda seorang agen bank yang berbahasa Malaysia. Berikan jawaban dalam bahasa Malaysia.")
    


pya = pyaudio.PyAudio()


class AudioLoop:
    """Manages bidirectional audio streaming with Gemini."""
    
    def __init__(self):
        self.audio_in_queue = None  # Audio from Gemini to speakers
        self.out_queue = None       # Audio from microphone to Gemini
        self.session = None         # Gemini API session
        self.audio_stream = None    # Microphone stream
    
    async def listen_audio(self):
        """Captures audio from microphone and queues it for sending."""
        # Get default microphone
        mic_info = pya.get_default_input_device_info()
        
        # Open microphone stream
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT, 
            channels=CHANNELS, 
            rate=SEND_SAMPLE_RATE,
            input=True, 
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        # Handle buffer overflow silently
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        
        # Continuously read audio chunks from microphone
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
    
    async def receive_audio(self):
        """Receives audio responses from Gemini."""
        while True:
            # Get next response from Gemini
            turn = self.session.receive()
            
            async for response in turn:
                # Handle audio data
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                
                # Handle text (if model includes it)
                if text := response.text:
                    print("Gemini:", text, end="")
            
            print()  # New line after Gemini's turn completes
    
    async def play_audio(self):
        """Plays audio responses through speakers."""
        # Open audio output stream
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT, 
            channels=CHANNELS, 
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        # Play each audio chunk as it arrives
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)
    
    async def send_realtime(self):
        """Sends microphone audio to Gemini API."""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)
    
    async def run(self):
        """Coordinates all audio streaming tasks."""
        try:
            # Connect to Gemini API
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)  # Limit buffer size
                
                print("Voice chat started. Speak into your microphone. Press Ctrl+C to quit.")
                print("Note: Using headphones is recommended to prevent feedback.")
                
                # Start all tasks
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                
                # Run until interrupted
                await asyncio.Future() 
                
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            print("Voice chat session ended.")


if __name__ == "__main__":
    try:
        main = AudioLoop()
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nChat terminated by user.")
    finally:
        pya.terminate()
        print("Audio resources released.")