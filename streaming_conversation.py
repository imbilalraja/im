import asyncio
import queue
import threading
from typing import Optional

from base_transcriber import BaseTranscriber
from base_synthesizer import BaseSynthesizer, SynthesisResult
from chat_gpt_agent import ChatGPTAgent
from audio_pipeline import AudioPipeline, OutputDeviceType

class StreamingConversation(AudioPipeline[OutputDeviceType]):
    def __init__(
        self,
        output_device,
        transcriber: BaseTranscriber,
        agent: ChatGPTAgent,
        synthesizer: BaseSynthesizer,
    ):
        super().__init__(output_device)
        self.id = f"conv_{hash(threading.current_thread().ident)}"  # Simple ID
        self.transcriber = transcriber
        self.agent = agent
        self.synthesizer = synthesizer
        self.synthesis_enabled = True
        self.transcript = ""  # Initialize transcript
        self.interruptible_events = queue.Queue()
        self.is_human_speaking = False
        self.is_terminated = asyncio.Event()
        self.interrupt_lock = asyncio.Lock()
        self.current_transcription_is_interrupt = False

    async def start(self):
        self.transcriber.streaming_conversation = self
        await self.transcriber.start()
        self.start()  # Start the audio pipeline
        self.synthesizer.streaming_conversation = self
        await self.synthesizer.start()  # Assuming start method exists
        await self.agent.start()
        self.is_terminated.clear()

    async def broadcast_interrupt(self):
        """Stops all inflight events and cancels workers sending output."""
        async with self.interrupt_lock:
            num_interrupts = 0
            while True:
                try:
                    interruptible_event = self.interruptible_events.get_nowait()
                    if not interruptible_event.is_interrupted():
                        if interruptible_event.interrupt():
                            num_interrupts += 1
                except queue.Empty:
                    break
            self.output_device.interrupt()
            return num_interrupts > 0

    async def send_speech_to_output(
        self,
        message: str,
        synthesis_result: SynthesisResult,
        stop_event: threading.Event,
        seconds_per_chunk: float = 0.5,
    ):
        """Sends speech chunk by chunk to the output device, stopping if interrupted."""
        audio_chunks = []
        interrupted_before_all_chunks_sent = False

        async for chunk_idx, chunk_result in self._enumerate_async(synthesis_result.chunk_generator):
            if stop_event.is_set():
                interrupted_before_all_chunks_sent = True
                break
            audio_chunk = AudioChunk(data=chunk_result.chunk)
            setattr(audio_chunk, "on_interrupt", lambda: stop_event.set())
            async with self.interrupt_lock:
                self.output_device.consume_nonblocking(audio_chunk)
            audio_chunks.append(audio_chunk)

        return not interrupted_before_all_chunks_sent, interrupted_before_all_chunks_sent

    def mark_terminated(self):
        self.is_terminated.set()

    async def terminate(self):
        self.mark_terminated()
        await self.broadcast_interrupt()
        await self.synthesizer.tear_down()
        await self.agent.terminate()
        await self.output_device.terminate()
        await self.transcriber.stop()

    def is_active(self):
        return not self.is_terminated.is_set()

    async def wait_for_termination(self):
        await self.is_terminated.wait()

    # Helper method to enumerate async iterator
    async def _enumerate_async(self, iterable):
        idx = 0
        async for item in iterable:
            yield idx, item
            idx += 1

# Simple AudioChunk class
class AudioChunk:
    def __init__(self, data):
        self.data = data
        self.state = None

    def __setattr__(self, name, value):
        if name in ["on_play", "on_interrupt"]:
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)

    def is_interrupted(self):
        return getattr(self, "state", None) == "interrupted"

    def interrupt(self):
        if hasattr(self, "on_interrupt"):
            self.on_interrupt()
            self.state = "interrupted"
            return True
        return False