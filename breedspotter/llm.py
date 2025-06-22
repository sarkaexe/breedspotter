"""Light wrapper around OpenAI chat completions. Fallbacks gracefully if no key."""
from __future__ import annotations

import os
from typing import Optional

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

_SYSTEM_PROMPT = (
    "Jesteś entuzjastycznym ekspertem kynologicznym, odpowiadaj po polsku. "
    "Gdy użytkownik przesyła zdjęcie psa i podajesz rasę, "
    "odpowiedz dwoma–trzema zdaniami po polsku, zwięźle opisując charakterystyczne cechy rasy." )


def describe_breed(breed: str, profile: str) -> str:
    """Generate (or retrieve) a short Polish description for *breed*."""
    # 1️⃣ If LLM unavailable, just return profile
    if openai is None or not os.getenv("API_KEY"):
        return profile

    client = openai.OpenAI()
    prompt = f"Rasa: {breed}\nInformacje referencyjne: {profile}\nOpis po polsku(2–3 zdania):"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # tiny, cheap, works fine
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return profile  # fallback silently