from abc import ABC


def get_messages(message: str):
    return [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': message}]


class BaseProvider(ABC):

    def __init__(self):
        pass

    def text(self, message: str):
        raise NotImplementedError

    def text_stream(self, message: str):
        raise NotImplementedError


class BaseAsyncProvider(ABC):

    def __init__(self):
        pass

    async def text(self, message: str):
        raise NotImplementedError

    async def text_stream(self, message: str):
        raise NotImplementedError
