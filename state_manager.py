from typing import Optional

from streaming_conversation import StreamingConversation

class ConversationStateManager:
    def __init__(self, conversation: StreamingConversation):
        self._conversation = conversation

    @property
    def transcript(self):
        return self._conversation.transcript

    def get_transcriber_endpointing_config(self) -> Optional[object]:
        if hasattr(self._conversation.transcriber, 'get_transcriber_config'):
            return getattr(self._conversation.transcriber.get_transcriber_config(), 'endpointing_config', None)
        return None

    def set_transcriber_endpointing_config(self, endpointing_config: object):
        if hasattr(self._conversation.transcriber, 'get_transcriber_config'):
            config = self._conversation.transcriber.get_transcriber_config()
            if hasattr(config, 'endpointing_config'):
                config.endpointing_config = endpointing_config

    def disable_synthesis(self):
        if hasattr(self._conversation, 'synthesis_enabled'):
            self._conversation.synthesis_enabled = False

    def enable_synthesis(self):
        if hasattr(self._conversation, 'synthesis_enabled'):
            self._conversation.synthesis_enabled = True

    def mute_agent(self):
        if hasattr(self._conversation, 'agent') and hasattr(self._conversation.agent, 'is_muted'):
            self._conversation.agent.is_muted = True

    def unmute_agent(self):
        if hasattr(self._conversation, 'agent') and hasattr(self._conversation.agent, 'is_muted'):
            self._conversation.agent.is_muted = False

    def using_input_streaming_synthesizer(self):
        return False  # Simplified, as InputStreamingSynthesizer isn't in your setup

    async def terminate_conversation(self):
        if hasattr(self._conversation, 'mark_terminated'):
            self._conversation.mark_terminated()

    def set_call_check_for_idle_paused(self, value: bool):
        pass  # Not implemented in your setup

    def get_conversation_id(self):
        return getattr(self._conversation, 'id', None)