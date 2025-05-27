import asyncio
import requests
import threading
from streaming_conversation import StreamingConversation
from base_transcriber import BaseTranscriber
from base_synthesizer import BaseSynthesizer, SynthesisResult
from chat_gpt_agent import ChatGPTAgent, ChatGPTAgentConfig
from default_factory import DefaultAgentFactory
from audio_pipeline import create_microphone_input_and_speaker_output

# Custom Grok Transcriber
class GrokTranscriber(BaseTranscriber):
    def __init__(self, config, api_key):
        super().__init__(config)
        self.api_key = api_key
        self.endpoint = "https://api.x.ai/stt"  # Replace with actual Grok STT endpoint
        self.is_speech = False
        self.is_running = False

    async def start(self):
        self.is_running = True
        print("GrokTranscriber started")

    async def process(self, audio_chunk):
        if not self.is_running:
            return None
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=audio_chunk
        )
        result = response.json()
        transcription = result.get("transcription", "")
        self.is_speech = result.get("is_speech", len(transcription) > 0)
        if self.is_speech and transcription:
            return {"message": transcription, "is_final": True, "is_interrupt": self.is_speech}
        return None

    async def stop(self):
        self.is_running = False
        print("GrokTranscriber stopped")

# Custom LemonFox Synthesizer
class LemonFoxSynthesizer(BaseSynthesizer):
    def __init__(self, config, api_key):
        super().__init__(config)
        self.api_key = api_key
        self.endpoint = "https://api.lemonfox.ai/tts"  # Replace with actual LemonFox TTS endpoint
        self.is_synthesizing = False

    async def create_speech(self, text, chunk_size):
        if not text:
            return SynthesisResult(None, chunk_size)
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"text": text, "format": self.config.audio_encoding, "sampling_rate": self.config.sampling_rate}
        )
        audio_data = response.content
        self.is_synthesizing = True
        return SynthesisResult(audio_data, chunk_size)

    async def stop(self):
        self.is_synthesizing = False
        print("LemonFoxSynthesizer stopped")

# Simplified Configurations
class EndpointingConfig:
    def __init__(self, min_speech_duration=0.3, min_silence_duration=0.5, sensitivity=0.8):
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.sensitivity = sensitivity

class TranscriberConfig:
    def __init__(self, sampling_rate=16000, audio_encoding="linear16", endpointing_config=None):
        self.sampling_rate = sampling_rate
        self.audio_encoding = audio_encoding
        self.endpointing_config = endpointing_config

class SynthesizerConfig:
    def __init__(self, sampling_rate=16000, audio_encoding="mp3"):
        self.sampling_rate = sampling_rate
        self.audio_encoding = audio_encoding

async def main():
    # Audio input/output
    microphone_input, speaker_output = create_microphone_input_and_speaker_output(use_default_devices=True)

    # Configure endpointing for interruption detection
    endpointing_config = EndpointingConfig()

    # Initialize components
    transcriber = GrokTranscriber(
        config=TranscriberConfig(endpointing_config=endpointing_config),
        api_key="your_grok_api_key"
    )
    agent_config = ChatGPTAgentConfig(model_name="gpt-4", max_tokens=500, temperature=0.7)
    agent = DefaultAgentFactory().create_agent(agent_config)
    agent.agent_responses_consumer = None  # To be set later
    agent.openai_api_key = "your_openai_api_key"  # Assuming chat_gpt_agent.py accepts this
    synthesizer = LemonFoxSynthesizer(
        config=SynthesizerConfig(),
        api_key="your_lemonfox_api_key"
    )

    # Set up conversation with interruption handling
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=transcriber,
        agent=agent,
        synthesizer=synthesizer
    )
    agent.agent_responses_consumer = conversation  # Link agent to conversation for response handling

    # Start the conversation
    await conversation.start()
    print("Conversation started. Speak to interact, and the assistant will pause on interruptions.")

    # Simple loop to handle interruptions
    stop_event = threading.Event()
    while conversation.is_active():
        audio_chunk = await microphone_input.read()
        transcription_result = await transcriber.process(audio_chunk)
        if transcription_result:
            transcription = type('Transcription', (), transcription_result)  # Dynamic class for compatibility
            if not conversation.is_human_speaking:
                stop_event.clear()
                conversation.current_transcription_is_interrupt = await conversation.broadcast_interrupt()
                if conversation.current_transcription_is_interrupt:
                    stop_event.set()
                conversation.is_human_speaking = True
            elif transcription_result.get("is_final", False) and conversation.is_human_speaking:
                conversation.is_human_speaking = False
                # Simulate agent input for compatibility
                class AgentInput:
                    def __init__(self, transcription, conversation_id):
                        self.transcription = transcription
                        self.conversation_id = conversation_id
                        self.is_interrupt = transcription_result.get("is_interrupt", False)
                agent_input = AgentInput(transcription, conversation.id)
                await agent.process(InterruptibleEvent(payload=agent_input))
        await asyncio.sleep(0.1)

    await conversation.terminate()

if __name__ == "__main__":
    asyncio.run(main())