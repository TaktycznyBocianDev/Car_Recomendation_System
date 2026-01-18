import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
from pathlib import Path
import re
import numpy as np

# =========================
# OPENAI CONFIG
# =========================
client = OpenAI(api_key="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")

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# =========================
# SESSION STATE
# =========================
if "data" not in st.session_state:
    st.session_state.data = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

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
def load_and_clean_data():
    df = pd.read_csv(r"C:\\Users\\sfran\\Desktop\\Pondel\\CARequester\\Car_Recomendation_System\\cars_new_final.csv").dropna()

    text_columns = ["product_name", "category", "gender", "gear", "description"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["gender"] = df["gender"].str.capitalize()
    return df.reset_index(drop=True)

# =========================
# EMBEDDINGS
# =========================
@st.cache_data
def get_embedding(text: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    ).data[0].embedding

def ensure_embeddings(df):
    if "embedding" not in df.columns:
        with st.spinner("Preparing recommendations (first time only)..."):
            combined_text = (
                "Brand and model: " + df["product_name"] + ". " +
                "Body type: " + df["category"] + ". " +
                "Gearbox: " + df["gear"] + ". " +
                "Description: " + df["description"]
            )
            df["embedding"] = combined_text.apply(get_embedding)
    return df

# =========================
# SALES DESCRIPTION (LLM)
# =========================
def generate_sales_description(row, user_query):
    prompt = f"""
You are a professional car salesman.

Customer is looking for:
"{user_query}"

Car:
{row['product_name']}
Category: {row['category']}
Gearbox: {row['gear']}
Price: {row.get('price', 'N/A')}
Description: {row['description']}

Write a persuasive, friendly sales description.
Focus on benefits and comfort.
Do not list raw attributes.
Do not mention AI.
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You sell cars."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

# =========================
# RECOMMENDATION ENGINE (HYBRID + CATEGORY + BRAND + GEAR)
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
    user_embedding = get_embedding(user_input)
    matrix = np.vstack(filtered["embedding"].values)
    similarities = cosine_similarity([user_embedding], matrix)[0]

    filtered = filtered.copy()
    filtered["similarity"] = similarities

    return filtered.sort_values("similarity", ascending=False).head(5)

# =========================
# IMAGES
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

    return random.sample(images, min(len(images), n))

# =========================
# STREAMLIT UI
# =========================
if st.session_state.data is None:
    st.session_state.data = load_and_clean_data()

st.title("🚗 AI Car Recommendation System")

user_input = st.text_input("Describe what kind of car you're looking for")

if st.button("Recommend"):
    data = ensure_embeddings(st.session_state.data)
    st.session_state.recommendations = recommend_products(user_input, data)

# =========================
# DISPLAY
# =========================
if st.session_state.recommendations is not None:
    st.header("🔥 Best Matches For You")

    for _, row in st.session_state.recommendations.iterrows():
        st.subheader(row["product_name"])

        sales_text = generate_sales_description(row, user_input)
        st.write(sales_text)

        images = get_random_images(row.get("file_path"))
        if images:
            cols = st.columns(len(images))
            for col, img in zip(cols, images):
                col.image(str(img), use_container_width=True)

        st.caption("Images for illustrative purposes only")
