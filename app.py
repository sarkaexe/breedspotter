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
import openrouter
from transformers.pipelines import TextGenerationPipeline

# --- 1) OpenRouter API client ---
router = openrouter.OpenRouter(api_key=st.secrets["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"])

# --- 2) Streamlit page config ---
st.set_page_config(page_title="🐶 BreedSpotter", layout="centered")

# --- 3) Load metadata ---
@st.cache_data
def load_metadata():
    df = pd.read_csv("stanford_dogs_metadata.csv")
    prof = pd.read_csv("breeds_profiles.csv")  # columns: breed,text,source
    profile_map = prof.groupby("breed")["text"].apply(list).to_dict()
    return df, profile_map

df, profile_map = load_metadata()
BREEDS = sorted(df.breed.unique())

# --- 4) Load CLIP ---
@st.cache_resource
def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = load_clip(device)

# --- 5) Precompute embeddings ---
@st.cache_resource
def embed_texts(texts):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)

breed_embeddings = embed_texts(BREEDS)
GENERAL_LABELS = ["dog", "cat", "car", "flower", "chair", "person", "bird"]
general_embeddings = embed_texts([f"a photo of a {lbl}" for lbl in GENERAL_LABELS])

# --- 6) JSON schema for validation ---
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "Rasa": {"type": "string"},
        "Opis": {"type": "string"},
    },
    "required": ["Rasa", "Opis"]
}

# --- 7) Detect if image is a dog ---
def detect_dog(img: Image.Image) -> bool:
    x = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ general_embeddings.T).squeeze(0)
    return GENERAL_LABELS[sims.argmax().item()] == "dog"

# --- 8) Classify breed ---
def classify_breed(img: Image.Image) -> str:
    x = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ breed_embeddings.T).squeeze(0)
    return BREEDS[sims.argmax().item()]

# --- 9) Retrieve & generate via OpenRouter RAG ---
def retrieve_and_generate(breed: str):
    # fetch up to 3 snippets
    docs = [d for d in profile_map.get(breed, []) if isinstance(d, str) and d.strip()][:3]
    while len(docs) < 3:
        docs.append("No further detail available.")

    snippets = "\n".join(f"- {d}" for d in docs)
    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant that writes concise, non-repetitive "
            "4-sentence paragraphs about dog temperament based on provided facts."
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Breed: {breed}\n"
            "Here are three facts about this breed:\n"
            f"{snippets}\n\n"
            "Write a 4-sentence paragraph describing this breed’s TEMPERAMENT only. "
            "Each sentence must be unique, not echo the facts verbatim, and flow naturally."
        )
    }

    resp = router.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_msg, user_msg],
        temperature=0.7,
        max_tokens=200,
    )
    text = resp.choices[0].message.content

    # split into sentences & dedupe
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

# --- 10) Streamlit UI ---
st.title("🐶 BreedSpotter — Dog breed recognition")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your photo", use_container_width=True)

    with st.spinner("Checking content..."):
        if not detect_dog(img):
            st.error("This image does not appear to contain a dog. Please upload a photo of a dog.")
        else:
            with st.spinner("Recognizing breed..."):
                breed = classify_breed(img)
            st.write(f"**Breed:** {breed}")

            with st.spinner("Generating temperament description..."):
                result = retrieve_and_generate(breed)
            st.markdown("### Description")
            st.write(result["Opis"])
