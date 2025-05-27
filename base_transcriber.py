import asyncio
from abc import ABC, abstractmethod

class TranscriberConfig:
    def __init__(self, sampling_rate: int = 16000, audio_encoding: str = "linear16", endpointing_config=None):
        self.sampling_rate = sampling_rate
        self.audio_encoding = audio_encoding
        self.endpointing_config = endpointing_config

class Transcription:
    def __init__(self, message: str, is_final: bool, is_interrupt: bool):
        self.message = message
        self.is_final = is_final
        self.is_interrupt = is_interrupt

class BaseTranscriber(ABC):
    streaming_conversation: 'StreamingConversation'

    def __init__(self, config: TranscriberConfig):
        self.config = config
        self.streaming_conversation = None
        self.is_muted = False

    def mute(self):
        self.is_muted = True

    def unmute(self):
        self.is_muted = False

    async def start(self):
        pass

    @abstractmethod
    async def process(self, audio_chunk: bytes) -> dict:
        pass

    async def stop(self):
        pass

    async def terminate(self):
        pass