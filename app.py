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
    prof = pd.read_csv("breeds_profiles.csv")
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

# 4) Precompute text embeddings
@st.cache_resource
def embed_breeds(breeds):
    with torch.no_grad():
        tokens = clip.tokenize(breeds).to(device)
        emb = clip_model.encode_text(tokens)
        return emb / emb.norm(dim=-1, keepdim=True)

breed_embeddings = embed_breeds(BREEDS)

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

# 7) Funkcja klasyfikująca rasę
def classify_image(img: Image.Image):
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ breed_embeddings.T).squeeze(0)
    idx = sims.argmax().item()
    return BREEDS[idx]

# 8) Retrieval + generowanie opisu w formie 5 przymiotników
def retrieve_and_generate(breed: str):
    raw_docs = profile_map.get(breed, [])
    # wybierz wszystkie niepuste snippet’y spoza placeholderów
    valid = [
        d.strip() for d in raw_docs
        if isinstance(d, str) and d.strip() and "combination of the two" not in d.lower()
    ]

    # jeśli brak wartościowych snippetów, przejdź do prompta bez fragmentu
    if valid:
        snippet = random.choice(valid)
        prompt = (
            f"Breed: {breed}\n"
            "Here is one key fact about this breed:\n"
            f"- {snippet}\n\n"
            "Provide exactly 5 adjectives (in English) that best describe this breed’s temperament, "
            "separated by commas. Do not echo the fact, do not repeat adjectives, and do not write anything else.\n"
        )
    else:
        # fallback: samą rasę
        prompt = (
            f"Breed: {breed}\n"
            "Provide exactly 5 adjectives (in English) that best describe this breed’s temperament, "
            "separated by commas. Do not write anything else.\n"
        )

    # Generuj
    out = generator(prompt, max_new_tokens=50)
    text = out[0].get("generated_text") or out[0].get("text", "")

    # Usuń echo prompta
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()

    # Weź pierwszą linię i rozdziel przecinkami
    line = text.splitlines()[0]
    parts = [a.strip().rstrip('.') for a in line.split(',') if a.strip()]
    top5 = parts[:5]

    result = {"Rasa": breed, "Opis": ", ".join(top5)}
    jsonschema.validate(instance=result, schema=RESPONSE_SCHEMA)
    return result

# 9) UI Streamlit
st.title("🐶 BreedSpotter — Dog breed recognition")

uploaded = st.file_uploader("Upload dog's photo", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your photo", use_container_width=True)

    with st.spinner("Recognizing breed..."):
        breed = classify_image(img)
    st.write(f"**Breed:** {breed}")

    with st.spinner("Generating temperament adjectives..."):
        result = retrieve_and_generate(breed)

    st.markdown("### Description")
    st.write(result["Opis"])

