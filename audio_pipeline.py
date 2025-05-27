import asyncio

class OutputDeviceType:
    def __init__(self):
        self.is_active = True

    def start(self):
        self.is_active = True

    def interrupt(self):
        pass

    def consume_nonblocking(self, item):
        # Simulate playing audio (replace with actual speaker output if needed)
        print("Playing audio chunk:", item.data if hasattr(item, 'data') else item)

    async def terminate(self):
        self.is_active = False

class MicrophoneInput:
    async def read(self):
        # Simulate reading audio chunks (replace with actual microphone input)
        await asyncio.sleep(0.1)
        return b"audio_data"

class SpeakerOutput(OutputDeviceType):
    pass

class AudioPipeline:
    def __init__(self, output_device: OutputDeviceType):
        self.output_device = output_device
        self.is_running = False

    def receive_audio(self, chunk: bytes):
        if self.is_running:
            self.output_device.consume_nonblocking(chunk)

    def is_active(self):
        return self.is_running and self.output_device.is_active

    def start(self):
        self.is_running = True
        self.output_device.start()

    async def terminate(self):
        self.is_running = False
        await self.output_device.terminate()

def create_microphone_input_and_speaker_output(use_default_devices=True):
    return MicrophoneInput(), SpeakerOutput()