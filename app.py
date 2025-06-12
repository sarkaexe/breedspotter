import os
# Wyłącz warningi Transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from PIL import Image
import pandas as pd
import torch
import clip
import json
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

# 3) Ładowanie CLIP
@st.cache_resource
def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = load_clip(device)
BREEDS = sorted(df.breed.unique())

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

# 6) Inicjalizacja generatora HF z GPT-Neo 125M
@st.cache_resource
def get_generator() -> TextGenerationPipeline:
    return pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-125M",
        do_sample=False,
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

# 8) Retrieval + generowanie opisu
def retrieve_and_generate(breed: str):
    # Pobierz top-3 snippetów
    raw_docs = prof := profile_map.get(breed, [])
    docs = [d for d in prof if isinstance(d, str) and d.strip()][:3]
    if not docs:
        # fallback: czysta generacja
        prompt = (
            f"Breed: {breed}\n"
            "Provide a detailed, 3-sentence description of the temperament "
            "and needs of this dog breed based on your general canine knowledge.\n"
        )
    else:
        snippets = "\n".join(f"- {d}" for d in docs)
        prompt = (
            f"Breed: {breed}\n"
            "Provide a detailed, 3-sentence description of the temperament "
            "and needs of this dog breed based on the following snippets:\n"
            f"{snippets}\n"
        )

    # Generuj
    out = generator(prompt, max_new_tokens=80)
    text = out[0].get("generated_text") or out[0].get("text", "")

    # Usuń echo prompta, jeśli wystąpiło
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()

    # Weź pierwsze 3 zdania
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    first_three = '. '.join(sentences[:3])
    if first_three and not first_three.endswith('.'):
        first_three += '.'

    result = {"Rasa": breed, "Opis": first_three}
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

    with st.spinner("Generating description..."):
        result = retrieve_and_generate(breed)

    st.markdown("### Description")
    st.write(result["Opis"])
