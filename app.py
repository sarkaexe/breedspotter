import streamlit as st
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
import clip  # openai/CLIP
import chromadb
from chromadb.config import Settings
import openai
import json
import jsonschema

# --- 1. Load metadata ---
@st.cache_resource
def load_metadata():
    df = pd.read_csv("stanford_dogs_metadata.csv")  # filepath, breed
    prof = pd.read_csv("breeds_profiles.csv")       # breed, text, source
    return df, prof

df, prof = load_metadata()

# --- 2. Initialize ChromaDB ---
@st.cache_resource
def init_chroma(prof_df):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_data"
    ))
    col = client.get_or_create_collection(name="breed_profiles")
    if col.count() == 0:
        for i, row in prof_df.iterrows():
            col.add(
                ids=[f"profile_{i}"],
                documents=[row.text],
                metadatas=[{"breed": row.breed, "source": row.source}]
            )
    return col

chroma = init_chroma(prof)

# --- 3. Load CLIP (OpenAI) ---
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

# --- 4. JSON schema for response validation ---
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "Rasa":    {"type": "string"},
        "Pewność": {"type": "string", "pattern": "^\\d{1,3}%$"},
        "Opis":    {"type": "string"},
        "Źródła":  {"type": "array", "items": {"type": "string"}}
    },
    "required": ["Rasa", "Pewność", "Opis", "Źródła"]
}

# Set OpenAI API key
openai.api_key = st.secrets.get("openai_api_key")

# --- 5. Classification function ---
def classify_image(img: Image.Image):
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ breed_embeddings.T).squeeze(0)
    score, idx = sims.max().item(), sims.argmax().item()
    return BREEDS[idx], score * 100

# --- 6. Retrieval + generation using OpenAI---
def retrieve_and_generate(breed, conf):
    if conf < 50:
        return None, False, []
    res = chroma.query(
        query_texts=[breed], n_results=3, where={"breed": breed}
    )
    docs = res["documents"][0]
    sources = [md["source"] for md in res["metadatas"][0]]
    prompt = (
        f"Zidentyfikowano rasę: {breed} ({conf:.1f}%).\n"
        "Na podstawie poniższych fragmentów opisz temperament i potrzeby tej rasy "
        "w formie JSON z polami Rasa, Pewność, Opis, Źródła:\n" +
        "\n".join(docs)
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
        jsonschema.validate(instance=data, schema=RESPONSE_SCHEMA)
        return data, True, sources
    except Exception:
        return text, False, sources

# --- 7. Streamlit UI ---
st.set_page_config(page_title="🐶 BreedSpotter", layout="centered")
st.title("🐶 BreedSpotter — Rozpoznawanie ras psów")

uploaded = st.file_uploader("Wgraj zdjęcie psa", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Twoje zdjęcie", use_column_width=True)
    with st.spinner("Rozpoznawanie rasy..."):
        breed, conf = classify_image(img)
    st.write(f"**Rasa:** {breed}")
    st.write(f"**Pewność:** {conf:.1f}%")
    if conf < 50:
        st.warning("Nie jestem pewien – podaj lepsze zdjęcie.")
    else:
        with st.spinner("Generowanie opisu..."):
            result, valid, srcs = retrieve_and_generate(breed, conf)
        if not valid:
            st.error("Nie udało się zwalidować odpowiedzi.")
        else:
            st.markdown("### Opis temperamentu i potrzeb")
            st.write(result["Opis"] if isinstance(result, dict) else result)
            st.markdown("#### Źródła")
            for s in srcs:
                st.write(f"- {s}")

