import pytest

from Ahri.Asuka.llm.openai_provider import AsyncOpenAIProvider, OpenAIProvider


def test_openai_provider():
    client = OpenAIProvider(url="https://api.openai.com/v1", api_key="your_openai_api_key", model="gpt-5")
    result = client.text("你好，请简单介绍一下自己")
    assert result is not None, "返回的结果不能为空"
    print(result)


@pytest.mark.asyncio
async def test_async_openai_provider():
    client = AsyncOpenAIProvider(url="https://api.openai.com/v1", api_key="your_openai_api_key", model="gpt-5")
    result = await client.text("你好，请简单介绍一下自己")
    assert result is not None, "返回的结果不能为空"
    print(result)


if __name__ == '__main__':
    pytest.main([__file__])
