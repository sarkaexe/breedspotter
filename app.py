import os
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
        temperature=0.7,
        top_k=50,
        top_p=0.9,
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

# 8) Retrieval + generowanie opisu (3 zdania) z 1 snippetem
def retrieve_and_generate(breed: str):
    # Pobierz top-1 snippet
    raw_docs = profile_map.get(breed, [])
    snippet = ""
    for d in raw_docs:
        if isinstance(d, str) and d.strip():
            snippet = d
            break

    # Buduj prompt na podstawie jednego snippet’a
    prompt = (
        f"Breed: {breed}\n"
        "Based on the following fact, write a clear 3-sentence paragraph "
        "describing the temperament and needs of this dog breed:\n"
        f"- {snippet}\n"
    )

    # Generuj tekst
    out = generator(prompt, max_new_tokens=100)
    text = out[0].get("generated_text") or out[0].get("text", "")

    # Usuń echo prompta, jeśli wystąpiło
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()

    # Podziel na zdania i oczyść
    raw_sents = [s.strip() for s in text.split('.') if s.strip()]

    # Usuń konsekutywne duplikaty
    sentences, prev = [], None
    for s in raw_sents:
        if s != prev:
            sentences.append(s)
        prev = s

    # Weź pierwsze 3 unikalne zdania
    paragraph = ". ".join(sentences[:3])
    if paragraph and not paragraph.endswith('.'):
        paragraph += '.'

    result = {"Rasa": breed, "Opis": paragraph}
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
