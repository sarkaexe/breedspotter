import streamlit as st
from PIL import Image
import pandas as pd
import torch
import clip  # openai/CLIP
from transformers import pipeline
import jsonschema
import json

# Page config must be first Streamlit command
st.set_page_config(page_title="🐶 BreedSpotter", layout="centered")

# --- 1. Load metadata ---
@st.cache_data
def load_metadata():
    df = pd.read_csv("stanford_dogs_metadata.csv")  # filepath, breed
    prof = pd.read_csv("breeds_profiles.csv")       # breed, text, source
    profile_map = prof.groupby("breed")["text"].apply(list).to_dict()
    source_map = prof.groupby("breed")["source"].apply(list).to_dict()
    return df, profile_map, source_map

df, profile_map, source_map = load_metadata()

# --- 2. Load CLIP model ---
@st.cache_resource
def load_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = load_clip_model(device)

# Prepare embeddings for dog detection and breeds
BREEDS = sorted(df.breed.unique())
@st.cache_resource
def embed_texts(texts):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)

breed_embeddings = embed_texts(BREEDS)
detector_texts = ["a photo of a dog", "not a dog"]
detector_embeddings = embed_texts(detector_texts)

# --- 3. Initialize Hugging Face Llama-2 pipeline ---
@st.cache_resource
def get_generator():
    return pipeline(
        "text-generation",
        model="meta-llama/Llama-2-7b-chat-hf",
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False,
        temperature=0.7,
        max_new_tokens=100
    )
generator = get_generator()

# JSON schema for output validation
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "Rasa": {"type": "string"},
        "Opis": {"type": "string"},
        "Źródła": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["Rasa", "Opis", "Źródła"]
}

# --- 4. Detection function ---
def is_dog(img: Image.Image) -> bool:
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ detector_embeddings.T).squeeze(0)
    return sims[0] > sims[1]

# --- 5. Classification function ---
def classify_image(img: Image.Image) -> str:
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ breed_embeddings.T).squeeze(0)
    idx = sims.argmax().item()
    return BREEDS[idx]

# --- 6. Generation using Llama-2 pipeline ---
@st.cache_data(show_spinner=False)
def retrieve_and_generate(breed: str):
    docs = profile_map.get(breed, [])[:2]
    sources = source_map.get(breed, [])[:2]
    sources = [str(s) for s in sources if isinstance(s, str) and s.strip()]
    snippets = "\n".join(f"- {d}" for d in docs)
    prompt = f"""Breed: {breed}.
Describe the temperament and needs of this breed based on the following snippets:
{snippets}"""
    out = generator(prompt)
    text = out[0]["generated_text"]
    result = {"Rasa": breed, "Opis": text.strip(), "Źródła": sources}
    jsonschema.validate(instance=result, schema=RESPONSE_SCHEMA)
    return result, sources

# --- 7. Streamlit UI ---
st.title("🐶 BreedSpotter — Dog breed recognition")

uploaded = st.file_uploader("Upload a photo", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your photo", use_container_width=True)
    if not is_dog(img):
        st.error("This does not look like a dog. Please upload a dog photo.")
    else:
        with st.spinner("Classifying breed..."):
            breed = classify_image(img)
        st.write(f"**Breed:** {breed}")
        with st.spinner("Generating description..."):
            result, srcs = retrieve_and_generate(breed)
        st.markdown("### Description")
        st.write(result["Opis"])
        st.markdown("#### Sources")
        for s in srcs:
            st.write(f"- {s}")

