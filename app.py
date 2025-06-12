import streamlit as st
from PIL import Image
import pandas as pd
import torch
import clip  # openai/CLIP
import json
import jsonschema
from transformers import pipeline, Pipeline
from transformers.pipelines import TextGenerationPipeline

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
BREEDS = sorted(df.breed.unique())

# Precompute text embeddings
@st.cache_resource
def embed_breeds(breeds):
    with torch.no_grad():
        text_tokens = clip.tokenize(breeds).to(device)
        text_emb = clip_model.encode_text(text_tokens)
        return text_emb / text_emb.norm(dim=-1, keepdim=True)

breed_embeddings = embed_breeds(BREEDS)

# --- 3. JSON schema for response validation ---
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "Rasa":    {"type": "string"},
        "Opis":    {"type": "string"},
        "Źródła":  {"type": "array", "items": {"type": "string"}}
    },
    "required": ["Rasa", "Opis", "Źródła"]
}

# --- 4. Generator factory with fallback ---
@st.cache_resource
def get_generator() -> TextGenerationPipeline:
    try:
        # Llama-2 chat model
        return pipeline(
            "text-generation",
            model="meta-llama/Llama-2-7b-chat-hf",
            max_new_tokens=100,
            do_sample=False
        )
    except Exception:
        # Fallback to distilgpt2
        return pipeline("text-generation", model="distilgpt2", max_new_tokens=100, do_sample=False)

generator: Pipeline = get_generator()

# --- 5. Classification function ---
def classify_image(img: Image.Image):
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ breed_embeddings.T).squeeze(0)
    idx = sims.argmax().item()
    return BREEDS[idx]

# --- 6. Retrieval + generation with HF model ---
def retrieve_and_generate(breed: str):
    docs = profile_map.get(breed, [])[:2]
    sources = source_map.get(breed, [])[:2]
    # Filter out non-string or nan sources
    sources = [s for s in sources if isinstance(s, str) and s.strip()]
    snippets = "\n".join(f"- {d}" for d in docs)
    prompt = f"""Breed: {breed}
Describe the temperament and needs of this breed based on the following snippets:
{snippets}
Provide the answer as JSON with fields Rasa, Opis, Źródła."""
    out = generator(prompt)
    # For HF pipelines, generated_text may be key "generated_text" or "text"
    text = out[0].get("generated_text") or out[0].get("text", "")
    result = {"Rasa": breed, "Opis": text.strip(), "Źródła": sources}
    jsonschema.validate(instance=result, schema=RESPONSE_SCHEMA)
    return result, sources

# --- 7. Streamlit UI ---
st.title("🐶 BreedSpotter — Dog breed recognition")

uploaded = st.file_uploader("Upload dog's photo", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your photo", use_container_width=True)
    with st.spinner("Recognizing breed..."):
        breed = classify_image(img)
    st.write(f"**Breed:** {breed}")
    with st.spinner("Generating description..."):
        result, srcs = retrieve_and_generate(breed)
    st.markdown("### Description")
    st.write(result["Opis"])
    st.markdown("#### Sources")
    for s in srcs:
        st.write(f"- {s}")
