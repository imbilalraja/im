import asyncio
import hashlib
from typing import Optional

import aiohttp
from base_synthesizer import BaseSynthesizer, SynthesisResult

LEMONFOX_BASE_URL = "https://api.lemonfox.ai/tts"
STREAMED_CHUNK_SIZE = 16000 * 2 // 4  # 1/8 of a second of 16kHz audio with 16-bit samples

class LemonFoxSynthesizerConfig:
    def __init__(self, api_key: str, voice_id: str = "default", sampling_rate: int = 16000, audio_encoding: str = "linear16"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.sampling_rate = sampling_rate
        self.audio_encoding = audio_encoding

class LemonFoxSynthesizer(BaseSynthesizer[LemonFoxSynthesizerConfig]):
    def __init__(self, synthesizer_config: LemonFoxSynthesizerConfig):
        super().__init__(synthesizer_config)
        assert synthesizer_config.api_key is not None, "API key must be set"
        self.api_key = synthesizer_config.api_key
        self.voice_id = synthesizer_config.voice_id
        self.output_format = self._determine_output_format()

    def _determine_output_format(self) -> str:
        if self.synthesizer_config.audio_encoding == "linear16":
            return "pcm"  # Adjust based on LemonFox API documentation
        elif self.synthesizer_config.audio_encoding == "mulaw":
            return "ulaw"
        raise ValueError(f"Unsupported audio encoding: {self.synthesizer_config.audio_encoding}")

    async def create_speech_uncached(self, message: str, chunk_size: int) -> SynthesisResult:
        self.total_chars += len(message) if hasattr(self, 'total_chars') else 0
        url = f"{LEMONFOX_BASE_URL}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {
            "text": message,
            "format": self.output_format,
            "sampling_rate": self.synthesizer_config.sampling_rate,
            "voice_id": self.voice_id
        }

        chunk_queue = asyncio.Queue()
        asyncio.create_task(self.get_chunks(url, headers, body, chunk_size, chunk_queue))

        return SynthesisResult(
            self.chunk_result_generator_from_queue(chunk_queue),
            lambda seconds: self.get_message_cutoff_from_voice_speed(message, seconds, 150)
        )

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: LemonFoxSynthesizerConfig) -> str:
        hashed_api_key = hashlib.sha256(synthesizer_config.api_key.encode("utf-8")).hexdigest()
        return ":".join([
            "lemonfox",
            hashed_api_key,
            str(synthesizer_config.voice_id),
            synthesizer_config.audio_encoding,
        ])

    async def get_chunks(self, url: str, headers: dict, body: dict, chunk_size: int, chunk_queue: asyncio.Queue[Optional[bytes]]):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"LemonFox API error: {response.status} - {error}")
                    async for chunk in response.content.iter_chunked(chunk_size):
                        chunk_queue.put_nowait(chunk[0])  # Use first part of chunk
        except asyncio.CancelledError:
            pass
        finally:
            chunk_queue.put_nowait(None)  # Sentinel value