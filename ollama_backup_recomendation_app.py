import streamlit as st
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
from pathlib import Path

# =========================
# OLLAMA CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

# =========================
# SESSION STATE INIT
# =========================
if "data" not in st.session_state:
    st.session_state.data = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "similar_cars" not in st.session_state:
    st.session_state.similar_cars = None

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("cars_new_final.csv").dropna()

    if "MSRP" in df.columns:
        df["MSRP"] = df["MSRP"].astype(int)

    text_columns = ["product_name", "category", "gender", "gear", "description"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["gender"] = df["gender"].str.capitalize()
    return df.reset_index(drop=True)

# =========================
# OLLAMA EMBEDDINGS
# =========================
@st.cache_data
def get_embedding(text):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": text.replace("\n", " ")
    }

    r = requests.post(OLLAMA_URL, json=payload)
    return r.json()["embedding"]

# =========================
# LAZY EMBEDDINGS
# =========================
def ensure_embeddings(df):
    if "embedding" not in df.columns:
        with st.spinner("Preparing embeddings with Ollama (first time only)..."):
            combined_text = df["gender"] + " driver. " + df["description"]
            df["embedding"] = combined_text.apply(get_embedding)
    return df

# =========================
# RECOMMENDER FUNCTIONS
# =========================
def recommend_products(user_input, data):

    user_text = user_input.lower()
    user_embedding = get_embedding(user_input)

    data_filtered = data.copy()

    similarities = cosine_similarity(
        [user_embedding],
        data_filtered["embedding"].tolist()
    )[0]

    data_filtered["similarity"] = similarities

    # ===== GEAR BONUS =====
    if "automat" in user_text or "automatic" in user_text:
        data_filtered["similarity"] += (data_filtered["gear"].str.lower().str.contains("auto")).astype(int) * 0.20
        data_filtered["similarity"] -= (data_filtered["gear"].str.lower().str.contains("manual")).astype(int) * 0.25

    if "manual" in user_text:
        data_filtered["similarity"] += (data_filtered["gear"].str.lower().str.contains("manual")).astype(int) * 0.20

    # ===== GENDER BONUS =====
    if "męski" in user_text or "men" in user_text:
        data_filtered["similarity"] += (data_filtered["gender"].str.lower() == "men").astype(int) * 0.15

    if "damski" in user_text or "women" in user_text:
        data_filtered["similarity"] += (data_filtered["gender"].str.lower() == "women").astype(int) * 0.15

    # ===== SIZE BONUS =====
    if "duży" in user_text or "large" in user_text or "big" in user_text:
        data_filtered["similarity"] += data_filtered["category"].str.lower().str.contains("suv|truck|van|wagon").astype(int) * 0.15

    if "mały" in user_text or "small" in user_text or "compact" in user_text:
        data_filtered["similarity"] += data_filtered["category"].str.lower().str.contains("compact|city|mini").astype(int) * 0.15

    return data_filtered.sort_values("similarity", ascending=False).head(5)


def recommend_similar_cars(data, car_row, top_k=3):

    data_filtered = data[data["product_name"] != car_row["product_name"]].copy()

    similarities = cosine_similarity(
        [car_row["embedding"]],
        data_filtered["embedding"].tolist()
    )[0]

    gender_bonus = (data_filtered["gender"] == car_row["gender"]).astype(int) * 0.05
    category_bonus = (data_filtered["category"] == car_row["category"]).astype(int) * 0.05

    data_filtered["similarity"] = similarities + gender_bonus + category_bonus
    return data_filtered.sort_values("similarity", ascending=False).head(top_k)

# =========================
# IMAGE HANDLING
# =========================
def get_random_images(folder_path, n=3):
    if not folder_path:
        return []

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []

    images = [
        folder / f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if not images:
        return []

    return random.sample(images, min(n, len(images)))

# =========================
# STREAMLIT UI
# =========================

if st.session_state.data is None:
    st.session_state.data = load_and_clean_data()

st.title("Car Recommendation System (Ollama Edition)")

user_input = st.text_input("Describe what kind of car you're looking for")

if st.button("Recommend"):
    data = ensure_embeddings(st.session_state.data)
    st.session_state.recommendations = recommend_products(user_input, data)

# =========================
# DISPLAY RECOMMENDATIONS
# =========================
if st.session_state.recommendations is not None:
    st.header("Recommended Cars")

    for idx, row in st.session_state.recommendations.iterrows():
        st.subheader(row["product_name"])
        st.write(row["description"])
        st.caption(f"Gear: {row['gear']} | Price: {row['price']}")

        images = get_random_images(row.get("file_path"), n=3)

        if images:
            cols = st.columns(3)
            for i, img_path in enumerate(images):
                with cols[i]:
                    st.image(str(img_path), use_container_width=True)


        if st.button("View details", key=f"details_{idx}"):

            st.session_state.similar_cars = recommend_similar_cars(st.session_state.data, row)

            st.header("Similar Cars")
            cols = st.columns(3)

            for j, (_, alt) in enumerate(st.session_state.similar_cars.iterrows()):
                with cols[j % 3]:
                    st.subheader(alt["product_name"])
                    st.write(alt["details"])
