import streamlit as st
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
import hashlib
from pathlib import Path
import numpy as np
import re

# =========================
# KONFIGURACJA
# =========================
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_LLM = "llama3"

DATA_FILE = r"C:\\Users\\sfran\\Desktop\\Pondel\\CARequester\\Car_Recomendation_System\\cars_new_final.csv"
EMBEDDINGS_FILE = "cars_with_embeddings.parquet"
EMBEDDING_CACHE_FILE = "embedding_cache.parquet"

# =========================
# SESSION STATE INIT
# =========================
if "data" not in st.session_state:
    st.session_state.data = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# =========================
# HELPERS
# =========================
def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# =========================
# INTENT EXTRACTION (BRAND + GEAR + CATEGORY)
# =========================
def extract_preferences(text):
    text = text.lower()
    prefs = {}

    # -------- gearbox --------
    if re.search(r"\bmanual\b|\bman\b|\bręczn|\bmanualna\b", text):
        prefs["gear"] = "manual"
    elif re.search(r"\bautomat\b|\bauto\b|\bautomatycz|\bautomatic\b", text):
        prefs["gear"] = "automatic"

    # -------- brands --------
    brands = [
        "bmw", "audi", "mercedes", "toyota", "honda", "ford",
        "mazda", "volkswagen", "vw", "skoda", "seat", "hyundai",
        "kia", "volvo", "lexus", "nissan", "peugeot", "renault",
        "opel", "fiat", "jeep", "tesla"
    ]
    for b in brands:
        if re.search(rf"\b{b}\b", text):
            prefs["brand"] = b
            break

    # -------- category / segment --------
    if re.search(r"\bsmall\b|\bcompact\b|\bcity\b|\bmiejski\b|\bmał", text):
        prefs["category"] = ["Hatchback", "Crossover"]
    elif re.search(r"\bsuv\b|\b4x4\b|\bcrossover\b|\bteren", text):
        prefs["category"] = ["SUV", "Crossover"]
    elif re.search(r"\bsedan\b|\blimousine\b|\blimuz", text):
        prefs["category"] = ["Sedan"]
    elif re.search(r"\bwagon\b|\bkombi\b|\bestate\b", text):
        prefs["category"] = ["Wagon"]
    elif re.search(r"\bcoupe\b|\bsport\b|\bsportowy\b", text):
        prefs["category"] = ["Coupe"]
    elif re.search(r"\bhatchback\b|\bhatch\b", text):
        prefs["category"] = ["Hatchback"]

    return prefs

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    if os.path.exists(EMBEDDINGS_FILE):
        return pd.read_parquet(EMBEDDINGS_FILE)

    df = pd.read_csv(DATA_FILE).dropna()
    text_columns = ["product_name", "category", "gear", "description"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df.reset_index(drop=True)

# =========================
# EMBEDDINGS
# =========================
def get_embedding(text: str, session: requests.Session):
    payload = {"model": OLLAMA_MODEL, "prompt": text.replace("\n", " ")}
    r = session.post(OLLAMA_EMBED_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]

def load_embedding_cache():
    if os.path.exists(EMBEDDING_CACHE_FILE):
        return pd.read_parquet(EMBEDDING_CACHE_FILE)
    return pd.DataFrame(columns=["text_hash", "embedding"])

def save_embedding_cache(df):
    df.to_parquet(EMBEDDING_CACHE_FILE, index=False)

def ensure_embeddings(df):
    if "embedding" in df.columns:
        return df

    with st.spinner("🔄 Generating embeddings (first run only)..."):
        cache = load_embedding_cache()
        cache_map = dict(zip(cache["text_hash"], cache["embedding"]))

        session = requests.Session()
        hashes = []
        embeddings = []

        progress = st.progress(0)
        total = len(df)

        for i, row in enumerate(df.itertuples(index=False), start=1):
            combined_text = (
                f"Brand and model: {row.product_name}. "
                f"Body type: {row.category}. "
                f"Gearbox type: {row.gear}. "
                f"Description: {row.description}"
            )

            text_hash = sha256_hash(combined_text)
            hashes.append(text_hash)

            if text_hash in cache_map:
                emb = cache_map[text_hash]
            else:
                emb = get_embedding(combined_text, session)
                cache_map[text_hash] = emb

            embeddings.append(np.array(emb, dtype="float32"))
            progress.progress(i / total)

        df = df.copy()
        df["text_hash"] = hashes
        df["embedding"] = embeddings

        df.to_parquet(EMBEDDINGS_FILE, index=False)
        save_embedding_cache(pd.DataFrame(cache_map.items(), columns=["text_hash", "embedding"]))
        progress.empty()

    return df

# =========================
# LLM SALES DESCRIPTION
# =========================
def generate_sales_description(row, user_query):
    prompt = f"""
You are a professional car salesman.

Customer is looking for:
"{user_query}"

Car:
{row['product_name']}

Write a persuasive, friendly sales description.
Focus on benefits and comfort.
Do not list raw attributes.
Do not mention AI.
"""

    payload = {"model": OLLAMA_LLM, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

# =========================
# RECOMMENDATION ENGINE (HYBRID + CATEGORY)
# =========================
def recommend_products(user_input, data):
    prefs = extract_preferences(user_input)
    filtered = data.copy()

    # -------- hard filters --------
    if "gear" in prefs:
        filtered = filtered[filtered["gear"].str.lower().str.contains(prefs["gear"], na=False)]

    if "brand" in prefs:
        filtered = filtered[
            filtered["product_name"].str.lower().str.contains(prefs["brand"], na=False)
        ]

    if "category" in prefs:
        filtered = filtered[filtered["category"].isin(prefs["category"])]

    # fallback jeśli wszystko wycięte
    if len(filtered) == 0:
        filtered = data.copy()

    # -------- embeddings ranking --------
    user_embedding = np.array(get_embedding(user_input, requests.Session()), dtype="float32")
    matrix = np.vstack(filtered["embedding"].values)
    similarities = cosine_similarity([user_embedding], matrix)[0]

    filtered = filtered.copy()
    filtered["similarity"] = similarities

    return filtered.sort_values("similarity", ascending=False).head(5)

# =========================
# RANDOM IMAGES
# =========================
def get_random_images(folder_path, n=3):
    if not folder_path:
        return []

    folder = Path(folder_path)
    if not folder.exists():
        return []

    images = [
        folder / f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".webp"))
    ]

    return random.sample(images, min(len(images), n))

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="AI Car Recommendation", layout="wide")

if st.session_state.data is None:
    st.session_state.data = load_data()

st.title("🚗 AI Car Recommendation System")

user_input = st.text_input("Describe what kind of car you're looking for")

if st.button("Recommend"):
    data = ensure_embeddings(st.session_state.data)
    st.session_state.recommendations = recommend_products(user_input, data)

# =========================
# DISPLAY RESULTS
# =========================
if st.session_state.recommendations is not None:
    st.header("🔥 Best Matches For You")

    for _, row in st.session_state.recommendations.iterrows():
        st.subheader(row["product_name"])

        sales_text = generate_sales_description(row, user_input)
        st.write(sales_text)

        if "price" in row:
            st.caption(f"Gear: {row.get('gear', 'N/A')} | Price: {row.get('price', 'N/A')}")
        else:
            st.caption(f"Gear: {row.get('gear', 'N/A')}")

        images = get_random_images(row.get("file_path"))
        if images:
            cols = st.columns(len(images))
            for c, img in zip(cols, images):
                c.image(str(img), use_container_width=True)
