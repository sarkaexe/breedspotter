import os
import random

# Wyłącz warningi Transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from PIL import Image
import pandas as pd
import torch
import clip
import jsonschema

# OpenRouter klient (nieoficjalny)
from openrouter_client import OpenRouterClient

# 1) Inicjalizacja OpenRouterClient
router = OpenRouterClient(api_key=st.secrets["openrouter_api_key"])

# 2) Konfiguracja Streamlit
st.set_page_config(page_title="🐶 BreedSpotter", layout="centered")

# 3) Wczytywanie metadanych
@st.cache_data
def load_metadata():
    df = pd.read_csv("stanford_dogs_metadata.csv")
    prof = pd.read_csv("breeds_profiles.csv")  # kolumny: breed,text,source
    profile_map = prof.groupby("breed")["text"].apply(list).to_dict()
    return df, profile_map

df, profile_map = load_metadata()
BREEDS = sorted(df.breed.unique())

# 4) Ładowanie CLIP
@st.cache_resource
def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = load_clip(device)

# 5) Precompute embeddingów
@st.cache_resource
def embed_texts(texts):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)

breed_embeddings = embed_texts(BREEDS)
GENERAL_LABELS = ["dog", "cat", "car", "flower", "chair", "person", "bird"]
general_embeddings = embed_texts([f"a photo of a {lbl}" for lbl in GENERAL_LABELS])

# 6) JSON-schema dla walidacji outputu
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "Rasa": {"type": "string"},
        "Opis": {"type": "string"},
    },
    "required": ["Rasa", "Opis"]
}

# 7) Funkcja detekcji: czy to pies?
def detect_dog(img: Image.Image) -> bool:
    x = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ general_embeddings.T).squeeze(0)
    return GENERAL_LABELS[sims.argmax().item()] == "dog"

# 8) Funkcja klasyfikująca rasę
def classify_breed(img: Image.Image) -> str:
    x = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ breed_embeddings.T).squeeze(0)
    return BREEDS[sims.argmax().item()]

# 9) RAG + OpenRouter: generujemy 4 zdania
def retrieve_and_generate(breed: str):
    # weź do 3 snippetów
    docs = [d for d in profile_map.get(breed, []) if isinstance(d, str) and d.strip()][:3]
    while len(docs) < 3:
        docs.append("No further detail available.")

    snippets = "\n".join(f"- {d}" for d in docs)

    # przygotuj wiadomości dla OpenRouterClient
    system_msg = {
        "role": "system",
        "content": (
            "You are a crisp assistant. Given three facts about a dog breed, "
            "write a 4-sentence paragraph describing only its TEMPERAMENT. "
            "Each sentence must be unique, not echo the facts verbatim, "
            "and flow as a cohesive paragraph."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"Breed: {breed}\nFacts:\n{snippets}"
    }

    # wywołanie OpenRouter
    resp = router.chat.create(
        model="gpt-3.5-turbo",
        messages=[system_msg, user_msg],
        temperature=0.7,
        max_tokens=200,
    )

    text = resp.choices[0].message.content

    # podziel na zdania & usuń powtórki
    sents = [s.strip() for s in text.split('.') if s.strip()]
    unique = []
    prev = None
    for s in sents:
        if s != prev:
            unique.append(s)
        prev = s
        if len(unique) >= 4:
            break

    paragraph = ". ".join(unique)
    if paragraph and not paragraph.endswith('.'):
        paragraph += '.'

    result = {"Rasa": breed, "Opis": paragraph}
    jsonschema.validate(instance=result, schema=RESPONSE_SCHEMA)
    return result

# 10) UI Streamlit
st.title("🐶 BreedSpotter — Dog breed recognition")

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your photo", use_container_width=True)

    with st.spinner("Checking content..."):
        if not detect_dog(img):
            st.error("This image does not appear to contain a dog. 🐶")
        else:
            with st.spinner("Recognizing breed..."):
                breed = classify_breed(img)
            st.write(f"**Breed:** {breed}")

            with st.spinner("Generating temperament description..."):
                result = retrieve_and_generate(breed)
            st.markdown("### Description")
            st.write(result["Opis"])
