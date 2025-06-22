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
from transformers import pipeline
from transformers.pipelines import TextGenerationPipeline

# 1) Konfiguracja strony
st.set_page_config(page_title="🐶 BreedSpotter", layout="centered")

# 2) Wczytywanie metadanych
@st.cache_data
def load_metadata():
    df = pd.read_csv("stanford_dogs_metadata.csv")
    prof = pd.read_csv("breeds_profiles.csv")  # kolumny: breed,text,source
    profile_map = prof.groupby("breed")["text"].apply(list).to_dict()
    return df, profile_map

df, profile_map = load_metadata()
BREEDS = sorted(df.breed.unique())

# 3) Ładowanie CLIP
@st.cache_resource
def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = load_clip(device)

# 4) Precompute text embeddings dla ras i klas ogólnych
@st.cache_resource
def embed_texts(texts):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)

breed_embeddings = embed_texts(BREEDS)
GENERAL_LABELS = ["dog", "cat", "car", "flower", "chair", "person", "bird"]
general_embeddings = embed_texts([f"a photo of a {lbl}" for lbl in GENERAL_LABELS])

# 5) Schemat JSON dla walidacji
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "Rasa": {"type": "string"},
        "Opis": {"type": "string"},
    },
    "required": ["Rasa", "Opis"]
}

# 6) Inicjalizacja generatora HF z GPT-Neo 125M (sampling enabled)
@st.cache_resource
def get_generator() -> TextGenerationPipeline:
    return pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-125M",
        do_sample=True,
        temperature=0.6,
        top_k=30,
        top_p=0.85,
        truncation=True
    )

generator = get_generator()

# 7) Detekcja, czy obraz przedstawia psa
def detect_dog(img: Image.Image) -> bool:
    x = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ general_embeddings.T).squeeze(0)
    best = sims.argmax().item()
    return GENERAL_LABELS[best] == "dog"

# 8) Klasyfikacja rasy
def classify_breed(img: Image.Image) -> str:
    x = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ breed_embeddings.T).squeeze(0)
    idx = sims.argmax().item()
    return BREEDS[idx]

# 9) RAG-based generation: 3 snippet’y → 4 unikalne zdania o temperamencie
def retrieve_and_generate(breed: str):
    # Pobierz dokładnie 3 snippet’y
    docs = [d for d in profile_map.get(breed, []) if isinstance(d, str) and d.strip()][:3]
    while len(docs) < 3:
        docs.append("No further detail available.")

    # Zbuduj prompt
    snippets_block = "\n".join(f"- {d}" for d in docs)
    prompt = (
        f"Breed: {breed}\n"
        "Below are three key facts about this dog breed:\n"
        f"{snippets_block}\n\n"
        "Using these facts, write a coherent 4-sentence paragraph describing the breed’s TEMPERAMENT only. "
        "Each sentence must be unique (no repeated phrases), must not echo the facts verbatim, "
        "and should flow naturally as a single paragraph.\n"
    )

    # Generuj
    out = generator(prompt, max_new_tokens=120)
    text = out[0].get("generated_text") or out[0].get("text", "")

    # Usuń echo prompta, jeśli wystąpiło
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()

    # Podziel na zdania i oczyść
    sents = [s.strip() for s in text.split('.') if s.strip()]

    # Usuń powtórzenia kolejnych zdań
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

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your photo", use_container_width=True)

    with st.spinner("Checking content..."):
        if not detect_dog(img):
            st.error("This image does not appear to contain a dog 🐶. Please upload a photo of a dog.")
        else:
            with st.spinner("Recognizing breed..."):
                breed = classify_breed(img)
            st.write(f"**Breed:** {breed}")

            with st.spinner("Generating temperament description..."):
                result = retrieve_and_generate(breed)

            st.markdown("### Description")
            st.write(result["Opis"])
