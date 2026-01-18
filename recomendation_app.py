import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
from pathlib import Path

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
# DATA LOADING
# =========================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(r"C:\\Users\\sfran\\Desktop\\Pondel\\CARequester\\Car_Recomendation_System\\cars_new_final.csv").dropna()

    text_columns = [
        "product_name",
        "category",
        "gender",
        "details",
        "description"
    ]

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
                df["product_name"] + ". " +
                df["category"] + ". " +
                df["gender"] + " driver. " +
                df["description"]
            )
            df["embedding"] = combined_text.apply(get_embedding)
    return df

# =========================
# SALES DESCRIPTION (LLM)
# =========================
def generate_sales_description(row, user_query):
    prompt = f"""
You are an experienced car salesman.

Customer request:
"{user_query}"

Car information:
Name: {row['product_name']}
Category: {row['category']}
Target driver: {row['gender']}
Price: {row.get('price', 'N/A')}
Base description: {row['description']}
Technical details: {row['details']}

Write a persuasive, natural sales description.
Focus on benefits, comfort and matching the customer needs.
Do not list raw specs.
Do not mention AI.
Tone: confident, friendly, professional.
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
# RECOMMENDER
# =========================
def recommend_products(user_input, data):
    user_embedding = get_embedding(user_input)

    similarities = cosine_similarity(
        [user_embedding],
        data["embedding"].tolist()
    )[0]

    data = data.copy()
    data["similarity"] = similarities

    return data.sort_values("similarity", ascending=False).head(5)

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
