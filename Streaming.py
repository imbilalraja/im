import asyncio
import threading
import aiohttp
from streaming_conversation import StreamingConversation
from base_transcriber import BaseTranscriber, TranscriberConfig
from lemonfox_synthesizer import LemonFoxSynthesizer, LemonFoxSynthesizerConfig
from chat_gpt_agent import ChatGPTAgent, ChatGPTAgentConfig
from default_factory import DefaultAgentFactory
from audio_pipeline import create_microphone_input_and_speaker_output
from whisper_transcriber import WhisperTranscriber, WhisperTranscriberConfig
from conversation_state_manager import ConversationStateManager

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
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=audio_chunk
            ) as response:
                result = await response.json()
                transcription = result.get("transcription", "")
                self.is_speech = result.get("is_speech", len(transcription) > 0)
                if self.is_speech and transcription:
                    return {"message": transcription, "is_final": True, "is_interrupt": self.is_speech}
                return None

    async def stop(self):
        self.is_running = False
        print("GrokTranscriber stopped")

    async def terminate(self):
        await self.stop()  # Align with BaseTranscriber interface

# Simplified Configurations
class EndpointingConfig:
    def __init__(self, min_speech_duration=0.3, min_silence_duration=0.5, sensitivity=0.8):
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.sensitivity = sensitivity

async def main():
    # Audio input/output
    microphone_input, speaker_output = create_microphone_input_and_speaker_output(use_default_devices=True)

    # Configure endpointing for interruption detection
    endpointing_config = EndpointingConfig()

    # Initialize transcriber (toggle between Grok and Whisper)
    use_whisper = True  # Set to False to use Grok STT
    if use_whisper:
        transcriber = WhisperTranscriber(
            transcriber_config=WhisperTranscriberConfig(
                api_key="your_openai_api_key",
                endpointing_config=endpointing_config
            )
        )
    else:
        transcriber = GrokTranscriber(
            config=TranscriberConfig(endpointing_config=endpointing_config),
            api_key="your_grok_api_key"
        )

    agent_config = ChatGPTAgentConfig(model_name="gpt-4", max_tokens=500, temperature=0.7)
    factory = DefaultAgentFactory()
    agent = factory.create_agent(agent_config)
    agent.openai_api_key = "your_openai_api_key"

    synthesizer = LemonFoxSynthesizer(
        synthesizer_config=LemonFoxSynthesizerConfig(
            api_key="your_lemonfox_api_key",
            voice_id="default"
        )
    )

    # Set up conversation with state manager
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=transcriber,
        agent=agent,
        synthesizer=synthesizer
    )
    state_manager = ConversationStateManager(conversation)
    agent.agent_responses_consumer = conversation  # Link agent to conversation for response handling

    # Start the conversation
    await conversation.start()
    print(f"Conversation started. ID: {conversation.id}. Initial transcript: {state_manager.transcript}")

    # Simple loop to handle interruptions
    stop_event = threading.Event()
    while conversation.is_active():
        audio_chunk = await microphone_input.read()
        transcription_result = await transcriber.process(audio_chunk)
        if transcription_result:
            transcription = type('Transcription', (), transcription_result)  # Dynamic class for compatibility
            conversation.transcript += f" {transcription_result['message']}"  # Update transcript
            if not conversation.is_human_speaking:
                stop_event.clear()
                conversation.current_transcription_is_interrupt = await conversation.broadcast_interrupt()
                if conversation.current_transcription_is_interrupt:
                    stop_event.set()
                conversation.is_human_speaking = True
            elif transcription_result.get("is_final", False) and conversation.is_human_speaking:
                conversation.is_human_speaking = False
                class AgentInput:
                    def __init__(self, transcription, conversation_id):
                        self.transcription = transcription
                        self.conversation_id = conversation_id
                        self.is_interrupt = transcription_result.get("is_interrupt", False)
                agent_input = AgentInput(transcription, conversation.id)
                synthesis_result = await agent.process(InterruptibleEvent(payload=agent_input))
                if synthesis_result and conversation.synthesis_enabled:
                    success, _ = await conversation.send_speech_to_output(
                        synthesis_result.message,
                        synthesis_result,
                        stop_event
                    )
                    if not success:
                        print("Speech interrupted")

        await asyncio.sleep(0.1)

    await conversation.terminate()

if __name__ == "__main__":
    asyncio.run(main())