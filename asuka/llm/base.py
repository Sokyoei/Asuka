from abc import ABC, abstractmethod


def get_messages(message: str):
    return [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': message}]


class BaseProvider(ABC):

    @abstractmethod
    def text(self, message: str):
        raise NotImplementedError

    @abstractmethod
    def text_stream(self, message: str):
        raise NotImplementedError


class BaseAsyncProvider(ABC):

    @abstractmethod
    async def text(self, message: str):
        raise NotImplementedError

    @abstractmethod
    async def text_stream(self, message: str):
        raise NotImplementedError
