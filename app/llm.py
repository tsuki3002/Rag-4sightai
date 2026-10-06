from functools import lru_cache

from openai import OpenAI

from .config import get_settings


@lru_cache
def client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key or "missing")


@lru_cache
def vision_client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.vision_base_url or s.llm_base_url,
                  api_key=s.vision_api_key or s.llm_api_key or "missing")


def chat(messages: list[dict], model: str | None = None, **kwargs) -> str:
    completion = client().chat.completions.create(
        model=model or get_settings().llm_model, messages=messages, **kwargs
    )
    return completion.choices[0].message.content or ""


def stream_chat(messages: list[dict], model: str | None = None, **kwargs):
    """Yield the answer text in pieces as it is generated (reasoning tokens, if any, are skipped)."""
    stream = client().chat.completions.create(
        model=model or get_settings().llm_model, messages=messages, stream=True, **kwargs
    )
    for chunk in stream:
        if chunk.choices and (piece := chunk.choices[0].delta.content):
            yield piece
