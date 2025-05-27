import asyncio
import aiohttp
from base_transcriber import BaseTranscriber, TranscriberConfig

WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"

class WhisperTranscriberConfig(TranscriberConfig):
    def __init__(self, api_key: str, sampling_rate: int = 16000, audio_encoding: str = "linear16", endpointing_config=None):
        super().__init__(sampling_rate, audio_encoding, endpointing_config)
        self.api_key = api_key

class WhisperTranscriber(BaseTranscriber):
    def __init__(self, transcriber_config: WhisperTranscriberConfig):
        super().__init__(transcriber_config)
        self.api_key = transcriber_config.api_key
        if not self.api_key:
            raise ValueError("Please set OPENAI_API_KEY for Whisper")
        self._ended = False
        self.is_ready = False
        self.audio_buffer = bytearray()
        self.buffer_duration = 0.0  # In seconds
        self.time_silent = 0.0

    async def start(self):
        self.is_running = True
        self.is_ready = True
        print("WhisperTranscriber started")

    async def process(self, audio_chunk: bytes):
        if not self.is_running:
            return None
        # Accumulate audio chunks
        byte_rate = self.get_byte_rate()
        chunk_duration = len(audio_chunk) / byte_rate
        self.audio_buffer.extend(audio_chunk)
        self.buffer_duration += chunk_duration

        # Check for endpointing
        if self.should_endpoint():
            transcription = await self.transcribe_buffer()
            if transcription:
                self.audio_buffer.clear()
                self.buffer_duration = 0.0
                self.time_silent = 0.0
                return {
                    "message": transcription,
                    "is_final": True,
                    "is_interrupt": False
                }
        else:
            self.time_silent += chunk_duration
        return None

    async def stop(self):
        self.is_running = False
        print("WhisperTranscriber stopped")

    async def terminate(self):
        self._ended = True
        await super().terminate()

    def get_byte_rate(self):
        return self.config.sampling_rate * (2 if self.config.audio_encoding == "linear16" else 1)

    def should_endpoint(self):
        endpointing_config = self.config.endpointing_config
        if not endpointing_config:
            return self.buffer_duration >= 5.0  # Default to 5 seconds
        return self.time_silent >= endpointing_config.min_silence_duration

    async def transcribe_buffer(self) -> str:
        if not self.audio_buffer:
            return ""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "multipart/form-data"
        }
        form_data = aiohttp.FormData()
        form_data.add_field('file', bytes(self.audio_buffer), filename='audio.wav', content_type='audio/wav')
        form_data.add_field('model', 'whisper-1')
        form_data.add_field('response_format', 'json')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(WHISPER_API_URL, headers={"Authorization": f"Bearer {self.api_key}"}, data=form_data) as response:
                if response.status != 200:
                    error = await response.text()
                    print(f"Whisper API error: {response.status} - {error}")
                    return ""
                result = await response.json()
                return result.get("text", "")