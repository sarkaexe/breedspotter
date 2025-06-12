import streamlit as st
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
import chromadb
from chromadb.config import Settings
from guardrails import Guard

# --- 1. Dane i bazy ---
@st.cache_resource
def load_metadata():
    # CSV wygenerowane wcześniej lokalnie z ImageNetDogs + profile ras:
    df = pd.read_csv("stanford_dogs_metadata.csv")         # filepath, breed
    prof = pd.read_csv("breeds_profiles.csv")              # breed, text, source
    return df, prof

df, prof = load_metadata()

@st.cache_resource
def init_chroma(prof_df):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_data"
    ))
    col = client.get_or_create_collection(name="breed_profiles")
    # jeśli baza jest pusta, wypełnij:
    if col.count() == 0:
        for i, row in prof_df.iterrows():
            col.add(
                ids=[f"profile_{i}"],
                documents=[row["text"]],
                metadatas=[{"breed": row["breed"], "source": row["source"]}]
            )
    return col

chroma = init_chroma(prof)

# --- 2. CLIP setup ---
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc

clip_model, clip_processor = load_clip()

# --- 3. Guardrails schema ---
guard_schema = {
    "schema_version": "1.0",
    "templates": [],
    "rails": [
        {
            "name": "breed_response",
            "type": "json",
            "schema": {
                "type": "object",
                "properties": {
                    "Rasa":      {"type": "string"},
                    "Pewność":   {"type": "string", "pattern": "^\\d{1,3}%$"},
                    "Opis":      {"type": "string"},
                    "Źródła":    {"type": "array","items":{"type":"string"}}
                },
                "required": ["Rasa","Pewność","Opis","Źródła"]
            }
        }
    ]
}
guard = Guard.from_rail_string(guard_schema)

# --- 4. Klassifier & RAG functions ---
def classify_image(img: Image.Image):
    img = img.resize((224,224))
    inp = clip_processor(images=img, return_tensors="pt")
    emb_i = clip_model.get_image_features(**inp)
    emb_i = emb_i / emb_i.norm(p=2, dim=-1, keepdim=True)
    # embed all breed names once
    global breed_embeddings, BREEDS
    sims = (emb_i @ breed_embeddings.T).squeeze(0)
    idx, score = sims.argmax().item(), sims.max().item()
    return BREEDS[idx], score * 100

@st.cache_resource
def embed_breeds(breeds):
    inp = clip_processor(text=breeds, return_tensors="pt", padding=True)
    emb = clip_model.get_text_features(**inp)
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

BREEDS = sorted(df["breed"].unique())
breed_embeddings = embed_breeds(BREEDS)

def retrieve_and_generate(breed, conf):
    if conf < 50:
        return None, False, []
    res = chroma.query(
        query_texts=[breed],
        n_results=3,
        where={"breed": breed}
    )
    docs   = res["documents"][0]
    srcs   = [md["source"] for md in res["metadatas"][0]]
    prompt = (
        f"Pewność rozpoznania: {conf:.1f}%.\n"
        f"Proszę o krótki opis temperamentu i potrzeb rasy {breed}.\n"
        f"Użyj poniższych fragmentów i podaj cytaty:\n" +
        "\n".join(docs)
    )
    # generatywny model
    tok   = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    genm  = AutoModelForSeq2SeqLM.from_pretrained("tiiuae/falcon-7b-instruct", device_map="auto")
    inp   = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(genm.device)
    out   = genm.generate(**inp, max_new_tokens=200)
    text  = tok.decode(out[0], skip_special_tokens=True)
    valid, _ = guard.run(text, rail_name="breed_response")
    return text, valid, srcs

# --- 5. Streamlit UI ---
st.set_page_config(page_title="🐶 BreedSpotter", layout="centered")
st.title("🐶 BreedSpotter — rozpoznawanie ras psów")

uploaded = st.file_uploader("Wgraj zdjęcie psa", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Twoje zdjęcie", use_column_width=True)
    with st.spinner("Rozpoznaję rasę..."):
        breed, conf = classify_image(img)
    st.write(f"**Rasa:** {breed}")
    st.write(f"**Pewność:** {conf:.1f}%")
    if conf < 50:
        st.warning("Nie jestem pewien – podaj lepsze zdjęcie.")
    else:
        with st.spinner("Generuję opis..."):
            desc, ok, srcs = retrieve_and_generate(breed, conf)
        if not ok:
            st.error("Nie udało się wygenerować odpowiedzi, spróbuj ponownie.")
        else:
            st.markdown("### Opis temperamentu i potrzeb")
            st.write(desc)
            st.markdown("#### Źródła")
            for s in srcs:
                st.write(f"- {s}")

