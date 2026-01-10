import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# OPENAI CONFIG
# =========================
APIKey = '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'
openai.api_key = APIKey
client = OpenAI(api_key=APIKey)


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
    df = pd.read_csv("cars_ready.csv").dropna()
    df["MSRP"] = df["MSRP"].astype(int)

    text_columns = ["product_name", "category", "gender", "details", "description"]
    for col in text_columns:
        df[col] = df[col].astype(str)

    df["gender"] = df["gender"].str.capitalize()
    return df.reset_index(drop=True)


@st.cache_data
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(
        input=[text],
        model=model
    ).data[0].embedding


# =========================
# LAZY EMBEDDINGS
# =========================
def ensure_embeddings(df):
    if "embedding" not in df.columns:
        with st.spinner("Preparing recommendations (first time only)..."):
            combined_text = df["gender"] + " driver. " + df["description"]
            df["embedding"] = combined_text.apply(get_embedding)
    return df


# =========================
# INITIALIZE DATA
# =========================
if st.session_state.data is None:
    st.session_state.data = load_and_clean_data()

data = st.session_state.data


# =========================
# RECOMMENDER FUNCTIONS
# =========================
def recommend_products(user_input, data, gender=None, category=None):

    if gender:
        user_input = f"{gender} driver looking for a car. " + user_input

    user_embedding = get_embedding(user_input)
    data_filtered = data.copy()

    if category:
        data_filtered = data_filtered[data_filtered["category"] == category]

    similarities = cosine_similarity(
        [user_embedding],
        data_filtered["embedding"].tolist()
    )[0]

    gender_bonus = (data_filtered["gender"] == gender).astype(int) * 0.05 if gender else 0
    data_filtered["similarity"] = similarities + gender_bonus

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
# STREAMLIT UI
# =========================
st.title("Car Recommendation System")

user_input = st.text_input("Describe what kind of car you're looking for")

if st.button("Recommend"):
    data = ensure_embeddings(data)
    st.session_state.recommendations = recommend_products(user_input, data)


# =========================
# DISPLAY RECOMMENDATIONS
# =========================
if st.session_state.recommendations is not None:
    st.header("Recommended Cars")

    for idx, row in st.session_state.recommendations.iterrows():
        st.subheader(row["product_name"])
        st.write(row["details"])

        # 🔑 UNIKALNY KLUCZ
        if st.button(
            "View details",
            key=f"details_{row['product_name']}_{idx}"
        ):
            st.session_state.similar_cars = recommend_similar_cars(data, row)

            st.header("Similar / Alternative Options")
            cols = st.columns(3)

            for j, (_, alt) in enumerate(st.session_state.similar_cars.iterrows()):
                with cols[j % 3]:
                    st.subheader(alt["product_name"])
                    st.write(alt["details"])
