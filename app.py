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
    df = pd.read_csv("stanford_dogs_metadata.csv")  # filepath, breed
    prof = pd.read_csv("breeds_profiles.csv")       # breed, text, source
    profile_map = prof.groupby("breed")["text"].apply(list).to_dict()
    source_map  = prof.groupby("breed")["source"].apply(list).to_dict()
    return df, profile_map, source_map

df, profile_map, source_map = load_metadata()

# 3) Ładowanie CLIP
@st.cache_resource
def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = load_clip(device)
BREEDS = sorted(df.breed.unique())

# 4) Precompute embeddings tekstowe ras
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
        "Rasa":   {"type": "string"},
        "Opis":   {"type": "string"},
        "Źródła": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["Rasa", "Opis", "Źródła"]
}

# 6) Inicjalizacja generatora HF (distilgpt2)
@st.cache_resource
def get_generator() -> TextGenerationPipeline:
    return pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=100,
        do_sample=False
    )

generator: TextGenerationPipeline = get_generator()

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
    # Pobierz i przefiltruj snippet’y
    raw_docs = profile_map.get(breed, [])
    docs = [d for d in raw_docs if isinstance(d, str) and d.strip()]
    if not docs:
        docs = [f"No profile snippets available for {breed}."]
    docs = docs[:3]

    # Pobierz źródła
    raw_srcs = source_map.get(breed, [])
    sources = [s for s in raw_srcs if isinstance(s, str) and s.strip()][:3]

    # Buduj prompt
    snippets = "\n".join(f"- {d}" for d in docs)
    prompt = (
        f"Breed: {breed}\n"
        "Provide a detailed, 3-sentence description of the temperament and needs of this breed based on the following snippets:\n"
        f"{snippets}\n"
    )

    out = generator(prompt)
    text = out[0].get("generated_text") or out[0].get("text", "")

    # Weź pierwsze 3 zdania
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    first_three = '. '.join(sentences[:3])
    if first_three and not first_three.endswith('.'):
        first_three += '.'

    result = {"Rasa": breed, "Opis": first_three, "Źródła": sources}
    jsonschema.validate(instance=result, schema=RESPONSE_SCHEMA)
    return result, sources

# 9) UI Streamlit
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
