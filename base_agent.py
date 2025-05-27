from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Optional, Tuple

from chat_gpt_agent import ChatGPTAgentConfig

class AgentResponseType:
    MESSAGE = "agent_response_message"
    STOP = "agent_response_stop"

class AgentResponse:
    pass

class AgentResponseMessage(AgentResponse):
    def __init__(self, message: str, is_interruptible: bool = True, is_first: bool = False):
        self.message = message
        self.is_interruptible = is_interruptible
        self.is_first = is_first

class AgentResponseStop(AgentResponse):
    pass

class GeneratedResponse:
    def __init__(self, message: str, is_interruptible: bool):
        self.message = message
        self.is_interruptible = is_interruptible

class AgentConfig:
    def __init__(self, initial_message: Optional[str] = None, allow_agent_to_be_cut_off: bool = True):
        self.initial_message = initial_message
        self.allow_agent_to_be_cut_off = allow_agent_to_be_cut_off

class AbstractAgent:
    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config

    def get_agent_config(self) -> AgentConfig:
        return self.agent_config

class BaseAgent(AbstractAgent):
    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)
        self.agent_responses_consumer = None
        self.is_muted = False

    async def respond(self, human_input: str, conversation_id: str, is_interrupt: bool = False) -> Tuple[Optional[str], bool]:
        raise NotImplementedError

    async def generate_response(self, human_input: str, conversation_id: str, is_interrupt: bool = False) -> AsyncGenerator[GeneratedResponse, None]:
        response = await self.respond(human_input, conversation_id, is_interrupt)
        if response[0]:
            yield GeneratedResponse(message=response[0], is_interruptible=self.agent_config.allow_agent_to_be_cut_off)
        yield GeneratedResponse(message="", is_interruptible=True)  # End of turn

    async def process(self, item):
        if self.is_muted:
            return
        if hasattr(item.payload, 'transcription'):
            transcription = item.payload.transcription.message
            conversation_id = item.payload.conversation_id
            should_stop = await self.handle_response(transcription, conversation_id, item.payload.is_interrupt)
            if should_stop:
                if self.agent_responses_consumer:
                    self.agent_responses_consumer.consume_nonblocking(AgentResponseStop())

    async def handle_response(self, transcription: str, conversation_id: str, is_interrupt: bool) -> bool:
        async for response in self.generate_response(transcription, conversation_id, is_interrupt):
            if self.agent_responses_consumer and response.message:
                self.agent_responses_consumer.consume_nonblocking(
                    AgentResponseMessage(
                        message=response.message,
                        is_interruptible=response.is_interruptible,
                        is_first=True
                    )
                )
        return False