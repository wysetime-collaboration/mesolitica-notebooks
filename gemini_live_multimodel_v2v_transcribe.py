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
import google.generativeai as generative
import io
import wave
import asyncio
import os
import sys
import traceback
import pyaudio
from google import genai
from google.genai.types import LiveConnectConfig, HttpOptions, Modality, SpeechConfig, AudioTranscriptionConfig
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

TRANSCRIPTION_MODEL = "gemini-2.0-flash"

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
    CONFIG = LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    system_instruction="Anda seorang agen yang berkerja di Bank HSBC yang berbahasa Malaysia. Berikan jawaban dalam bahasa Malaysia yang formal dan dengan menggunakan bahasa yang sopan dan mesra.",
    output_audio_transcription=AudioTranscriptionConfig(),
    )
    


pya = pyaudio.PyAudio()


class AudioLoop:
    """Manages bidirectional audio streaming with Gemini."""
    
    def __init__(self):
        self.audio_in_queue = None  # Audio from Gemini to speakers
        self.out_queue = None       # Audio from microphone to Gemini
        self.session = None         # Gemini API session
        self.audio_stream = None    # Microphone stream
        self.accumulated_user_audio = b'' # To store user's voice for transcription
        self.user_transcription_task = None # To hold the async transcription task
    
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
        print("--- Microphone listening ---")
        while True:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                # --- ADDED: Accumulate audio ---
                self.accumulated_user_audio += data
                # --- END ADDED ---

                # Queue data for sending to Gemini Live API (no change here)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except Exception as e:
                print(f"\nError in listen_audio: {e}")
                # Decide how to handle errors, maybe break or continue
                break # Exit loop on error for now
    
    # async def receive_audio(self):
    #     """Receives audio responses AND TRANSCRIPTIONS from Gemini, assembling fragments."""
    #     print("--- Listening for Gemini responses and transcriptions ---")

    #     # --- Reset transcription accumulators before processing each new turn stream ---
    #     # These will now build up the text from fragments during the turn processing.
    #     current_user_text = ""
    #     current_gemini_text = ""
    #     gemini_started_speaking_on_line = False # Flag to help manage console line breaks

    #     while True:
    #         # Reset accumulators at the start of EACH new turn stream from receive()
    #         current_user_text = ""
    #         current_gemini_text = ""
    #         gemini_started_speaking_on_line = False

    #         try:
    #             turn = self.session.receive() # Get the async iterator for the turn

    #             async for response in turn:
    #                 # --- 1. Process Audio Data ---
    #                 if data := response.data:
    #                     self.audio_in_queue.put_nowait(data)

    #                 # --- 2. Extract Transcription Fragments (if available) ---
    #                 user_text_fragment = None
    #                 gemini_text_fragment = None

    #                 if hasattr(response, 'server_content') and response.server_content:
    #                     server_content = response.server_content
    #                     if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
    #                         # *** This is likely JUST the fragment ***
    #                         user_text_fragment = server_content.input_transcription.text
    #                     if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
    #                         # *** This is likely JUST the fragment ***
    #                         gemini_text_fragment = server_content.output_transcription.text

    #                 # --- 3. Accumulate and Print User Transcription ---
    #                 if user_text_fragment: # Check if fragment is not None/empty string
    #                     # *** ACCUMULATE the fragment ***
    #                     current_user_text += user_text_fragment
    #                     print(f"\rUser: {current_user_text}        ", end="", flush=True)
    #                     gemini_started_speaking_on_line = False # Reset flag

    #                 # --- 4. Accumulate and Print Gemini Transcription ---
    #                 if gemini_text_fragment: # Check if fragment is not None/empty string
    #                     # If this is the first Gemini fragment after user text, move to a new line
    #                     if not gemini_started_speaking_on_line:
    #                         # Avoid extra newline if user didn't say anything this turn
    #                         if current_user_text:
    #                             print() # Move cursor off the user's line
    #                         gemini_started_speaking_on_line = True # Mark that Gemini is now 'owning' the line

    #                     # *** ACCUMULATE the fragment ***
    #                     current_gemini_text += gemini_text_fragment
    #                     print(f"\rGemini (Transcription): {current_gemini_text}        ", end="", flush=True)

    #                 # --- 5. Handle Final Text (response.text) ---
    #                 # This might be the final *word* or just confirmation text, print it distinctly
    #                 if final_text := response.text:
    #                     # Ensure we are on a new line before printing this block text
    #                     if gemini_started_speaking_on_line:
    #                          # Clear the accumulated transcription line first
    #                          print(f"\r{' ' * 80}\r", end="")
    #                     elif current_user_text: # Or if user was last speaking
    #                          print() # Move off the user line

    #                     print(f"Gemini (Final Text): {final_text}", flush=True) # Prints with newline
    #                     # Resetting gemini state here might be premature if more fragments follow
    #                     # Let the reset at the start of the while loop handle full clearing.
    #                     gemini_started_speaking_on_line = False # Reset flag

    #             # --- Optional: Action after processing all responses in the 'turn' stream ---
    #             # Add a final newline if the last thing printed was using end=""
    #             if current_user_text and not gemini_started_speaking_on_line:
    #                  print(flush=True)
    #             elif current_gemini_text:
    #                  print(flush=True)
    #             print("--- Turn Processed ---")


    #         except StopAsyncIteration:
    #             print("\n--- StopAsyncIteration (End of Turn Stream) ---")
    #             # The loop will continue and reset accumulators at the top
    #             continue
    #         except Exception as e:
    #             print(f"\nError in receive_audio: {e}")
    #             traceback.print_exc()
    #             break # Exit the loop on other errors

    #     print("--- Exited receive_audio loop ---")
    
    async def receive_audio(self):
        """Receives audio and transcription responses from Gemini."""
        print("--- Listening for Gemini responses ---")
        # Keep track of the latest full transcription text received for each speaker within the current exchange
        current_gemini_text = ""
        gemini_started_speaking_on_line = False
        # Flag to ensure transcription is only triggered once per user utterance
        transcription_triggered_for_turn = False

        while True:
            try:
                turn = self.session.receive()

                async for response in turn:
                    # --- 1. Process Audio Data & Trigger Transcription---
                    if data := response.data:
                        # --- ADDED: Trigger user transcription ---
                        # Trigger ONLY if Gemini sends audio AND we haven't already triggered for this user utterance
                        if self.accumulated_user_audio and not transcription_triggered_for_turn:
                            print("\n--- Triggering User Transcription ---")
                            audio_to_transcribe = self.accumulated_user_audio
                            self.accumulated_user_audio = b'' # Clear accumulator
                            transcription_triggered_for_turn = True # Mark as triggered

                            # Cancel previous task if still running (optional, depends on desired behavior)
                            # if self.user_transcription_task and not self.user_transcription_task.done():
                            #    self.user_transcription_task.cancel()

                            # Launch transcription in background
                            self.user_transcription_task = asyncio.create_task(
                                self.transcribe_user_audio(audio_to_transcribe)
                            )
                        # --- END ADDED ---

                        # Put Gemini audio into the queue to be played (no change here)
                        self.audio_in_queue.put_nowait(data)

                    # --- 2. Extract Gemini Transcription Fragments (if available) ---
                    gemini_text_fragment = None
                    if hasattr(response, 'server_content') and response.server_content:
                        server_content = response.server_content
                        if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                            gemini_text_fragment = server_content.output_transcription.text

                    # --- 3. Accumulate and Print Gemini Transcription ---
                    if gemini_text_fragment:
                        if not gemini_started_speaking_on_line:
                            # (Optional: Add newline only if user text was previously printed)
                            # We aren't printing user transcription here directly anymore
                            print() # Start Gemini transcription on a new line
                            gemini_started_speaking_on_line = True

                        current_gemini_text += gemini_text_fragment
                        print(f"\rGemini (Live Transcription): {current_gemini_text}        ", end="", flush=True)

                    # --- 4. Handle Final Text (response.text) ---
                    if final_text := response.text:
                        if gemini_started_speaking_on_line:
                             print(f"\r{' ' * 80}\r", end="") # Clear transcription line
                        print(f"Gemini (Final Text): {final_text}", flush=True)
                        gemini_started_speaking_on_line = False
                        current_gemini_text = "" # Reset gemini transcription state


                # --- After processing all responses in the 'turn' stream ---
                # Reset flags/state for the next turn
                if current_gemini_text: # If last thing was transcription
                     print(flush=True) # Ensure newline
                print("--- Gemini Turn Processed ---")
                transcription_triggered_for_turn = False # Ready to transcribe next user input
                gemini_started_speaking_on_line = False
                current_gemini_text = ""


            except StopAsyncIteration:
                print("\n--- StopAsyncIteration (End of Turn Stream) ---")
                transcription_triggered_for_turn = False # Ready to transcribe next user input
                gemini_started_speaking_on_line = False
                current_gemini_text = ""
                continue
            except Exception as e:
                print(f"\nError in receive_audio: {e}")
                traceback.print_exc()
                break

        print("--- Exited receive_audio loop ---")
        
    # Add this new method inside the AudioLoop class
    async def transcribe_user_audio(self, pcm_audio_data):
        """Transcribes the accumulated user audio using a separate Gemini call."""
        print(f"--- Transcribing {len(pcm_audio_data)} bytes of user audio ---")
        if not pcm_audio_data:
            print("--- No user audio data to transcribe ---")
            return

        try:
            # 1. Prepare WAV data in memory from PCM
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)                     # From your config (e.g., 1)
                wf.setsampwidth(pya.get_sample_size(FORMAT)) # From your config (e.g., 2 for paInt16)
                wf.setframerate(SEND_SAMPLE_RATE)             # Mic rate (e.g., 16000)
                wf.writeframes(pcm_audio_data)
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.getvalue()

            # Optional: Save WAV for debugging
            # with open("user_audio.wav", "wb") as f_debug:
            #    f_debug.write(wav_bytes)
            #    print("--- Saved user_audio.wav for debugging ---")


            # 2. Create audio blob for the API
            # Reference: https://ai.google.dev/api/rest/v1beta/Content#blob
            audio_blob = {"mime_type": "audio/wav", "data": wav_bytes}

            # 3. Get the transcription model
            # Consider creating the model instance once in __init__ if performance is critical
            model = generative.GenerativeModel(TRANSCRIPTION_MODEL)

            # 4. Prepare prompt
            # Using a prompt similar to the example helps guide the model
            prompt = """Transcribe the following audio recording.
Only output the transcribed text. If the audio is silent or unintelligible, output "<inaudible>"."""

            # 5. Call generate_content asynchronously
            # Make sure your GOOGLE_API_KEY has access to the TRANSCRIPTION_MODEL
            response = await model.generate_content_async([prompt, audio_blob])

            # 6. Print the result
            if response.text:
                 # Print on a new line, clearly marked
                print(f"\nUser (Transcribed): {response.text.strip()}", flush=True)
            else:
                 print("\nUser (Transcription): <No text returned>", flush=True)


        except Exception as e:
            print(f"\n--- Error during user audio transcription: {e} ---")
            traceback.print_exc() # Print detailed error

    
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