import streamlit as st
from PIL import Image
import pandas as pd
import torch
import clip  # openai/CLIP
import openai
import json
import jsonschema

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

# Set OpenAI API key
openai.api_key = st.secrets.get("openai_api_key")

# --- 4. Detection function ---
def is_dog(img: Image.Image) -> bool:
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ detector_embeddings.T).squeeze(0)
    return sims[0] > sims[1]

# --- 5. Classification function ---
def classify_image(img: Image.Image):
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ breed_embeddings.T).squeeze(0)
    score, idx = sims.max().item(), sims.argmax().item()
    return BREEDS[idx], score * 100

# --- 6. Retrieval + generation using caching and cheaper model ---
@st.cache_data(show_spinner=False)
def retrieve_and_generate(breed: str):
    docs = profile_map.get(breed, [])[:2]  # use top-2 fragments
    sources = source_map.get(breed, [])[:2]
    prompt = (
        f"Identify breed: {breed}.\n"
        "Describe its temperament and needs based on the following snippets (top 2).\n"
        + "\n".join(docs)
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
        jsonschema.validate(instance=data, schema=RESPONSE_SCHEMA)
        return data, sources
    except Exception:
        return {"Rasa": breed, "Opis": text, "Źródła": sources}, sources

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
            breed, conf = classify_image(img)
        st.write(f"**Breed:** {breed} ({conf:.1f}% confidence)")
        if conf < 50:
            st.warning("Low confidence. Try another photo.")
        else:
            with st.spinner("Generating description..."):
                result, srcs = retrieve_and_generate(breed)
            st.markdown("### Description")
            st.write(result.get("Opis"))
            st.markdown("#### Sources")
            for s in srcs:
                st.write(f"- {s}")
