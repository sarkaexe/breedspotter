import os
import random
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

# 4) Precompute text embeddings dla ras + ogólnych klas
@st.cache_resource
def embed_texts(texts):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)

# embeddings ras
breed_embeddings = embed_texts(BREEDS)
# embeddings klas ogólnych
GENERAL_LABELS = ["dog","cat","car","flower","chair","person","bird"]
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

# 6) Inicjalizacja generatora HF z GPT-Neo 125M
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

# 7) Funkcja sprawdzająca czy to pies
def detect_dog(img: Image.Image, threshold: float = 0.3) -> bool:
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ general_embeddings.T).squeeze(0)
    # highest similarity label
    best_idx = sims.argmax().item()
    best_label = GENERAL_LABELS[best_idx]
    return best_label == "dog"

# 8) Funkcja klasyfikująca rasę (zakładamy, że to pies)
def classify_breed(img: Image.Image) -> str:
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ breed_embeddings.T).squeeze(0)
    idx = sims.argmax().item()
    return BREEDS[idx]

# 9) Retrieval + generowanie 5 unikalnych przymiotników
def retrieve_and_generate(breed: str):
    raw_docs = profile_map.get(breed, [])
    valid = [
        d.strip() for d in raw_docs
        if isinstance(d, str) and d.strip()
    ]
    snippet = random.choice(valid) if valid else ""
    prompt = (
        f"Breed: {breed}\n"
        "Here is one key fact about this breed:\n"
        f"- {snippet}\n\n"
        "Provide exactly 5 unique adjectives (in English) that best describe this breed’s temperament, "
        "separated by commas. Each adjective must be different; do not echo the fact or repeat adjectives.\n"
    )
    out = generator(prompt, max_new_tokens=50)
    text = out[0].get("generated_text") or out[0].get("text", "")
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()
    line = text.splitlines()[0]
    parts = [a.strip().rstrip('.') for a in line.split(',') if a.strip()]
    unique = []
    for a in parts:
        if a.lower() not in [u.lower() for u in unique]:
            unique.append(a)
        if len(unique) == 5:
            break
    result = {"Rasa": breed, "Opis": ", ".join(unique)}
    jsonschema.validate(instance=result, schema=RESPONSE_SCHEMA)
    return result

# 10) UI Streamlit
st.title("🐶 BreedSpotter — Dog breed recognition")

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your photo", use_container_width=True)

    # najpierw detekcja: pies czy nie?
    with st.spinner("Checking if this is a dog..."):
        is_dog = detect_dog(img)
    if not is_dog:
        st.error("This image does not appear to contain a dog 🐶. Please upload a photo of a dog.")
    else:
        # klasyfikacja rasy
        with st.spinner("Recognizing breed..."):
            breed = classify_breed(img)
        st.write(f"**Breed:** {breed}")

        # generowanie przymiotników
        with st.spinner("Generating temperament adjectives..."):
            result = retrieve_and_generate(breed)
        st.markdown("### Description")
        st.write(result["Opis"])
