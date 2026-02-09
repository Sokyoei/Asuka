import pytest

from Ahri.Asuka.llm.ollama_provider import AsyncOllamaProvider, OllamaProvider


def test_ollama_provider():
    client = OllamaProvider(url='http://localhost:11434', model='qwen3:8b')
    result = client.text('你好')
    assert result is not None, "返回的结果不能为空"
    assert isinstance(result, str) and len(result.strip()) > 0, "返回的结果不能为空字符串"
    print(result)


@pytest.mark.asyncio
async def test_async_ollama_provider():
    client = AsyncOllamaProvider(url='http://localhost:11434', model='qwen3:8b')
    result = await client.text('你好')
    assert result is not None, "返回的结果不能为空"
    assert isinstance(result, str) and len(result.strip()) > 0, "返回的结果不能为空字符串"
    print(result)


if __name__ == '__main__':
    pytest.main(["-s", __file__])
