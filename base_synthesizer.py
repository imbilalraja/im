import asyncio
from typing import AsyncGenerator, Callable, Optional

class SynthesisResult:
    class ChunkResult:
        def __init__(self, chunk: bytes, is_last_chunk: bool):
            self.chunk = chunk
            self.is_last_chunk = is_last_chunk

    def __init__(
        self,
        chunk_generator: AsyncGenerator[ChunkResult, None],
        get_message_up_to: Callable[[Optional[float]], str],
    ):
        self.chunk_generator = chunk_generator
        self.get_message_up_to = get_message_up_to

class SynthesizerConfig:
    def __init__(self, sampling_rate: int = 16000, audio_encoding: str = "linear16"):
        self.sampling_rate = sampling_rate
        self.audio_encoding = audio_encoding

class BaseSynthesizer:
    streaming_conversation: 'StreamingConversation'

    def __init__(self, synthesizer_config: SynthesizerConfig):
        self.synthesizer_config = synthesizer_config
        self.streaming_conversation = None

    async def create_speech(self, message: str, chunk_size: int) -> SynthesisResult:
        raise NotImplementedError

    async def stop(self):
        pass

    async def tear_down(self):
        pass

    @staticmethod
    def get_message_cutoff_from_total_response_length(
        synthesizer_config: SynthesizerConfig,
        message: str,
        seconds: Optional[float],
        size_of_output: int,
    ) -> str:
        if not message or seconds is None:
            return message
        estimated_output_seconds = size_of_output / synthesizer_config.sampling_rate
        if estimated_output_seconds <= 0:
            return message
        estimated_chars_per_second = len(message) / estimated_output_seconds
        return message[:int(seconds * estimated_chars_per_second)]

    async def empty_generator(self):
        yield SynthesisResult.ChunkResult(b"", True)