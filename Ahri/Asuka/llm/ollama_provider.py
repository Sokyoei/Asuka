from typing import Iterator

from ollama import AsyncClient, ChatResponse, Client

from .base import BaseAsyncProvider, BaseProvider, get_messages


class OllamaProvider(BaseProvider):

    def __init__(self, url: str, model: str, **ollama_kwargs):
        super().__init__()
        self.url = url
        self.model = model
        self.client = Client(host=url, **ollama_kwargs)

    def text(self, message: str):
        response = self.client.chat(model=self.model, messages=get_messages(message))
        return response.message.content

    def text_stream(self, message: str):
        response: Iterator[ChatResponse] = self.client.chat(
            model=self.model, messages=get_messages(message), stream=True
        )
        for chunk in response:
            if chunk.message and chunk.message.content:
                content = chunk.message.content
                yield content


class AsyncOllamaProvider(BaseAsyncProvider):

    def __init__(self, url: str, model: str, **ollama_kwargs):
        super().__init__()
        self.url = url
        self.model = model
        self.client = AsyncClient(host=url, **ollama_kwargs)

    async def text(self, message: str):
        response = await self.client.chat(model=self.model, messages=get_messages(message))
        return response.message.content

    async def text_stream(self, message: str):
        response = await self.client.chat(model=self.model, messages=get_messages(message), stream=True)
        async for chunk in response:
            if chunk.message and chunk.message.content:
                content = chunk.message.content
                yield content
