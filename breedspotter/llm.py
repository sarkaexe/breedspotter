# breedspotter/llm.pyAdd commentMore actions

import streamlit as st
import openai

_SYSTEM_PROMPT = (
    "Jesteś entuzjastycznym ekspertem kynologicznym, odpowiadaj po polsku. "
    "Gdy użytkownik przesyła zdjęcie psa i podajesz rasę, "
    "odpowiedz dwoma–trzema zdaniami po polsku, zwięźle opisując cechy rasy."
)


def describe_breed(breed: str, profile: str) -> str:
    """
    Generate (or retrieve) a short Polish description for *breed*,
    zawsze odczytując api_key i base_url z st.secrets["openai"].
    """
    # 1️⃣ Odczyt z .streamlit/secrets.toml
    openai_secrets = st.secrets["openai"]
    api_key  = openai_secrets["api_key"]
    base_url = openai_secrets["base_url"]

    # 2️⃣ Jeśli nie ma klucza – fallback na lokalny profile
    if not api_key:
        return profile

    # 3️⃣ Konfiguracja klienta
    openai.api_key = api_key
    client = openai.OpenAI(base_url=base_url)

    # 4️⃣ Przygotowanie wiadomości
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Rasa: {breed}\n"
                f"Informacje referencyjne: {profile}\n"
                "Opis po polsku (2–3 zdania):"
            ),
        },
    ]

    # 5️⃣ Wywołanie API z obsługą błędów
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return profile
