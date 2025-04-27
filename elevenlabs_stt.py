from elevenlabs import ElevenLabs
from dotenv import load_dotenv
import os
load_dotenv()

ELEVENLABS_API_KEY = os.getenv("elevenlabs_api_key")

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)        
response = client.speech_to_text.convert(
	model_id="scribe_v1",
	file=open("test.mp3", "rb"),
	language_code="ms"
 )

print(response.text)
