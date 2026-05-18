"""
src/llm_call.py — Provider-agnostic LLM helper.

Exposes a single shape for all LLM calls in the repo, hiding the Anthropic/OpenAI
SDK differences behind two functions:

    provider, client = resolve_llm_key()
    client_tuple = build_client(provider, api_key)
    text = chat(client_tuple, system=..., user=..., max_tokens=...)

Callers always receive plain text — no provider-specific response shapes leak.
No try/except (fail-loudly per CLAUDE.md).
"""
from __future__ import annotations


DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4.1-nano",
}


def build_client(provider: str, api_key: str) -> tuple[str, object]:
    """Construct the SDK client for *provider* and return (provider, client).

    Lazy-imports the SDK so installing both is not required.
    """
    assert provider in DEFAULT_MODELS, (
        f"Unsupported provider {provider!r}. Supported: {sorted(DEFAULT_MODELS)}"
    )
    if provider == "anthropic":
        import anthropic
        return (provider, anthropic.Anthropic(api_key=api_key))
    from openai import OpenAI
    return (provider, OpenAI(api_key=api_key))


def chat(
    client_tuple: tuple[str, object],
    *,
    system: str | None,
    user: str,
    max_tokens: int,
    model: str | None = None,
) -> str:
    """Send a single user message (optionally with a system prompt) and return text.

    `model` defaults to DEFAULT_MODELS[provider] when omitted.
    """
    provider, client = client_tuple
    chosen_model = model or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        kwargs: dict = {
            "model": chosen_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system is not None:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    # openai
    msgs: list[dict] = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    response = client.chat.completions.create(
        model=chosen_model,
        max_tokens=max_tokens,
        messages=msgs,
    )
    return response.choices[0].message.content
