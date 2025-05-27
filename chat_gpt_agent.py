import os
import asyncio
from typing import AsyncGenerator

from openai import AsyncOpenAI
from base_agent import BaseAgent, GeneratedResponse, AgentConfig

class ChatGPTAgentConfig:
    def __init__(self, model_name: str = "gpt-4", max_tokens: int = 500, temperature: float = 0.7):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

class ChatGPTAgent(BaseAgent):
    def __init__(self, agent_config: ChatGPTAgentConfig, openai_api_key: str):
        super().__init__(agent_config)
        self.openai_client = AsyncOpenAI(
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )
        if not self.openai_client.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
        self.messages = [{"role": "system", "content": "You are a helpful voice assistant."}]

    async def respond(self, human_input: str, conversation_id: str, is_interrupt: bool = False) -> tuple[Optional[str], bool]:
        self.messages.append({"role": "user", "content": human_input})
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.agent_config.model_name,
                messages=self.messages,
                max_tokens=self.agent_config.max_tokens,
                temperature=self.agent_config.temperature
            )
            message = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": message})
            return message, False
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error.", False

    async def generate_response(self, human_input: str, conversation_id: str, is_interrupt: bool = False) -> AsyncGenerator[GeneratedResponse, None]:
        response, should_stop = await self.respond(human_input, conversation_id, is_interrupt)
        if response:
            yield GeneratedResponse(message=response, is_interruptible=self.agent_config.allow_agent_to_be_cut_off)
        yield GeneratedResponse(message="", is_interruptible=True)  # End of turn

    async def terminate(self):
        self.openai_client.close()
        await super().terminate()