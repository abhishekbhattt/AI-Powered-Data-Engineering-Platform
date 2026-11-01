
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Gemini models
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"

NEW_REVIEWS = 500
TOP_K = 5
CACHE_FILE = "review_embeddings.parquet"

# Gemini client
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing from your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# 1. Read reviews from Snowflake
def read_reviews_from_snowflake():

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    try:
        query = f"""
            SELECT REVIEW_ID, CITY, RATING, COMMENT
            FROM ZOMATO.STAGING.STG_REVIEWS
            SAMPLE ({NEW_REVIEWS} ROWS)
            WHERE COMMENT IS NOT NULL
        """

        df = conn.cursor().execute(query).fetch_pandas_all()

    finally:
        conn.close()

    df.columns = [col.lower() for col in df.columns]

    return df


# 2. Generate embeddings using Gemini
def embed(texts):

    all_embeddings = []

    for i in range(0, len(texts), 100):

        batch = texts[i:i + 100]

        print(f"Processing reviews {i + 1} to {i + len(batch)}")

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch
        )

        batch_embeddings = [
            item.values for item in response.embeddings
        ]

        all_embeddings.extend(batch_embeddings)

        # Wait 60 seconds before the next batch
        if i + 100 < len(texts):
            print("Waiting 60 seconds to respect Gemini quota...")
            time.sleep(60)

    return all_embeddings
# 3. Load reviews and cache embeddings
@st.cache_data(show_spinner="Loading reviews and generating embeddings...")
def load_reviews():

    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    df = read_reviews_from_snowflake()

    if df.empty:
        raise ValueError("No reviews found in Snowflake.")

    df["embedding"] = embed(df["comment"].tolist())

    df.to_parquet(CACHE_FILE, index=False)

    return df


st.title("Chat with your Zomato Reviews")

st.caption(
    f"Searching {NEW_REVIEWS} reviews using Gemini embeddings "
    f"and {CHAT_MODEL} for answers."
)


# 4. Calculate cosine similarity
def cosine_similarity(vec_a, vec_b):

    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

    if denominator == 0:
        return 0.0

    return np.dot(vec_a, vec_b) / denominator


# 5. Find the most relevant reviews
def find_similar_reviews(question, df):

    question_vector = embed([question])[0]

    scores = []

    for review_vector in df["embedding"]:
        scores.append(
            cosine_similarity(question_vector, review_vector)
        )

    df = df.copy()

    df["score"] = scores

    return df.nlargest(TOP_K, "score")


# 6. Generate answer using Gemini
def ask_llm(question, top_reviews):

    context = ""

    for _, row in top_reviews.iterrows():

        context += (
            f"({row['city']}, {row['rating']} stars) "
            f"{row['comment']}\n"
        )

    system_prompt = """
    Answer ONLY using the customer reviews provided.
    Be concise.
    If the reviews do not cover the question, say so.
    Do not invent information.
    """

    user_prompt = f"""
    Instructions:
    {system_prompt}

    Question:
    {question}

    Customer reviews:
    {context}
    """

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config={
            "temperature": 0.2
        }
    )

    return response.text


# 7. Streamlit application
try:

    review_df = load_reviews()

    st.success(f"Loaded {len(review_df)} customer reviews.")

    question = st.text_input(
        "Ask a question about your reviews:",
        placeholder="e.g. What are the most common complaints about delivery?"
    )

    if question:

        with st.spinner("Searching relevant reviews..."):

            top_reviews = find_similar_reviews(
                question,
                review_df
            )

            answer = ask_llm(
                question,
                top_reviews
            )

        st.markdown("### Answer")

        st.write(answer)

        with st.expander("Reviews used to build this answer"):

            st.dataframe(
                top_reviews[["city", "rating", "comment"]],
                hide_index=True
            )

except Exception as e:

    st.error(f"Something went wrong: {e}")
