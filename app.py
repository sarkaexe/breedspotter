"""Streamlit chatbot front-end for BreedSpotter."""
from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from breedspotter.classifier import load_classifier
from breedspotter.data import load_metadata, load_breed_profiles
from breedspotter.llm import describe_breed

st.set_page_config(page_title="BreedSpotter 🐶", page_icon="🐶")

st.title("🐶 BreedSpotter")
st.markdown(
    "Upload a photo of a dog, and I’ll tell you what breed it is, and share a few sentences about it."
)

# Lazy init
if "_init" not in st.session_state:
    st.session_state._init = True
    _df, st.session_state.breeds = load_metadata()
    st.session_state.profiles = load_breed_profiles()
    st.session_state.clf = load_classifier(st.session_state.breeds)

uploaded = st.file_uploader("Wybierz zdjęcie", type=["jpg", "jpeg", "png"])
if uploaded:
    with st.spinner("Analyzing image…"):
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        breed, prob, ranked = st.session_state.clf.predict(img)

    # Wyświetl obraz z opisem; zamiast use_container_width używamy width
    st.image(
        img,
        caption=f"Most likely breed: **{breed}** ({prob*100:.1f}%)",
        width=700,
    )

    # Generate / fetch description
    profile = st.session_state.profiles.get(breed, "Brak opisu w bazie.")
    description = describe_breed(breed, profile)

    st.markdown(f"### Opis rasy: {breed}")
    st.write(description)

    # Zamiast st.toggle używamy checkbox
    show_top5 = st.checkbox("Show the top 5 predictions", key="show_top5")
    if show_top5:
        for b, p in sorted(ranked, key=lambda t: t[1], reverse=True)[:5]:
            st.write(f"• {b}: {p*100:.1f}%")
