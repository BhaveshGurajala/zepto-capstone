"""Optional real-LLM client (only called when MOCK_LLM=0).

Uses Groq's free tier through its OpenAI-compatible chat endpoint. The API key
comes from the GROQ_API_KEY environment variable - it is never hard-coded.
"""
import json
import re

import requests
from pydantic import ValidationError

from app.config import GROQ_API_KEY, GROQ_MODEL, GROQ_URL, MAX_EXTRA_RETRIES
from app.prompts import CORRECTION_PROMPT
from app.schemas import AskResponse


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("MOCK_LLM=0 but GROQ_API_KEY is not set")
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": temperature},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LLM API returned HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]


def _parse(raw: str) -> AskResponse:
    # tolerate a stray ```json fence around the object
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    return AskResponse.model_validate(json.loads(cleaned))


def generate_validated(prompt: str) -> AskResponse:
    """Ask the LLM for JSON; retry up to MAX_EXTRA_RETRIES times with a corrective message.

    Returns a clearly marked error response if every attempt fails validation.
    """
    messages = [{"role": "user", "content": prompt}]
    last_error = ""
    for _attempt in range(1 + MAX_EXTRA_RETRIES):
        raw = chat(messages)
        try:
            return _parse(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)[:500]
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": CORRECTION_PROMPT.format(error=last_error)},
            ]
    return AskResponse(
        answer=f"[ERROR] The model did not return valid JSON after {1 + MAX_EXTRA_RETRIES} attempts. "
               f"Last validation error: {last_error}",
        sources=[],
        confidence=0.0,
    )
