from anthropic import Anthropic, AsyncAnthropic, AsyncStream, Stream
from anthropic.types import Message as AnthropicMessage

from .base import BaseAsyncProvider, BaseProvider, get_messages


class AnthropicProvider(BaseProvider):

    def __init__(self, url: str, api_key: str, model: str, **anthropic_kwargs):
        super().__init__()
        self.url = url
        self.api_key = api_key
        self.model = model
        self.client = Anthropic(api_key=api_key, base_url=url, **anthropic_kwargs)

    def text(self, message: str):
        completion = self.client.messages.create(model=self.model, max_tokens=1024, messages=get_messages(message))
        return completion.content[0].text

    def text_stream(self, message: str):
        completion: Stream[AnthropicMessage] = self.client.messages.create(
            model=self.model, max_tokens=1024, messages=get_messages(message), stream=True
        )
        for chunk in completion:
            if chunk.type == "content_block_delta":
                content = chunk.delta.text
                if content:
                    yield content
            elif chunk.type == "message_delta":
                pass


class AsyncAnthropicProvider(BaseAsyncProvider):

    def __init__(self, url: str, api_key: str, model: str, **anthropic_kwargs):
        super().__init__()
        self.url = url
        self.api_key = api_key
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key, base_url=url, **anthropic_kwargs)

    async def text(self, message: str):
        completion = await self.client.messages.create(
            model=self.model, max_tokens=1024, messages=get_messages(message)
        )
        return completion.content[0].text

    async def text_stream(self, message: str):
        completion: AsyncStream[AnthropicMessage] = await self.client.messages.create(
            model=self.model, max_tokens=1024, messages=get_messages(message), stream=True
        )
        async for chunk in completion:
            if chunk.type == "content_block_delta":
                content = chunk.delta.text
                if content:
                    yield content
            elif chunk.type == "message_delta":
                pass
