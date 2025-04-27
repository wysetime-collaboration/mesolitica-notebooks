from dotenv import load_dotenv
import os
import websockets
import json
import asyncio
import base64


load_dotenv()


# Load the API key from the .env file
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("elevenlabs_api_key")

voice_id = 'UcqZLa941Kkt8ZhEEybf' # malay voice (Afifah)

# For use cases where latency is important, we recommend using the 'eleven_flash_v2_5' model.
model_id = 'eleven_flash_v2_5'

uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"

async def text_to_speech_ws_streaming(voice_id, model_id):
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "use_speaker_boost": False},
            "generation_config": {
                "chunk_length_schedule": [120, 160, 250, 290]
            },
            "xi_api_key": ELEVENLABS_API_KEY,
        }))
        text = "Penduduk di Serening bertambah ramai sekarang, semenjak projeck tanam kelapa sawit berhasil, bak kata pepatah, ada gula ada semut."
        await websocket.send(json.dumps({"text": text}))
        # Send empty string to indicate the end of the text sequence which will close the WebSocket connection
        await websocket.send(json.dumps({"text": ""}))
        listen_task = asyncio.create_task(write_to_local(listen(websocket)))
        await listen_task
        
async def write_to_local(audio_stream):
    """Write the audio encoded in base64 string to a local mp3 file."""
    with open(f'./output/elevenlabs_tts_test_malay.mp3', "wb") as f:
        async for chunk in audio_stream:
            if chunk:
                f.write(chunk)
async def listen(websocket):
    """Listen to the websocket for audio data and stream it."""
    while True:
        try:
            message = await websocket.recv()
            data = json.loads(message)
            if data.get("audio"):
                yield base64.b64decode(data["audio"])
            elif data.get('isFinal'):
                break
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
            break
        
asyncio.run(text_to_speech_ws_streaming(voice_id, model_id))