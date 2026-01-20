from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

from .base import BaseAsyncProvider, BaseProvider, get_messages


class OpenAIProvider(BaseProvider):

    def __init__(self, url: str, api_key: str, model: str, **openai_kwargs):
        super().__init__()
        self.url = url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=url, **openai_kwargs)

    def text(self, message: str):
        completion = self.client.chat.completions.create(model=self.model, messages=get_messages(message))
        return completion.choices[0].message.content

    def text_stream(self, message: str):
        completion: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
            model=self.model, messages=get_messages(message), stream=True
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content


class AsyncOpenAIProvider(BaseAsyncProvider):

    def __init__(self, url: str, api_key: str, model: str, **openai_kwargs):
        super().__init__()
        self.url = url
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=url, **openai_kwargs)

    async def text(self, message: str):
        completion = await self.client.chat.completions.create(model=self.model, messages=get_messages(message))
        return completion.choices[0].message.content

    async def text_stream(self, message: str):
        completion: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(
            model=self.model, messages=get_messages(message), stream=True
        )
        async for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content
