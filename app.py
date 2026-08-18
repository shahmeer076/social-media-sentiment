import sqlite3
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from preprocessing import clean_text
from sentiment_analysis import analyze_sentiment


DB_PATH = Path(__file__).resolve().parent / "data" / "sentiment_analysis.db"


st.set_page_config(
    page_title="Sentiment Analysis & Topic Modeling",
    page_icon="📊",
    layout="wide"
)


def get_connection():
    return sqlite3.connect(DB_PATH)


def get_posts():
    connection = get_connection()

    query = """
        SELECT id, platform, username, text, sentiment,
               sentiment_score, topic
        FROM social_media_posts
    """

    df = pd.read_sql_query(query, connection)
    connection.close()

    return df


def decode_topic(value):
    if isinstance(value, bytes):
        return int.from_bytes(value, byteorder="little")
    return value


def get_topic_model():
    connection = get_connection()

    rows = connection.execute(
        "SELECT id, text FROM social_media_posts"
    ).fetchall()

    connection.close()

    ids = [row[0] for row in rows]
    texts = [clean_text(row[1]) for row in rows]

    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts)

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(matrix)

    return vectorizer, model, ids, labels


st.title("📊 Sentiment Analysis & Topic Modeling")
st.write("Social Media Insights Dashboard")

df = get_posts()

if not df.empty:
    df["topic"] = df["topic"].apply(decode_topic)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Posts", len(df))
    col2.metric(
        "Sentiment Analyzed",
        df["sentiment"].notna().sum()
    )
    col3.metric(
        "Topics Assigned",
        df["topic"].notna().sum()
    )

    st.divider()

    st.subheader("📈 Sentiment Distribution")

    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("📝 Analyzed Posts")

    st.dataframe(
        df[
            [
                "id",
                "platform",
                "username",
                "text",
                "sentiment",
                "sentiment_score",
                "topic"
            ]
        ],
        use_container_width=True
    )

else:
    st.warning("No posts found in the database.")


st.divider()

st.subheader("🔍 Analyze New Post")

user_text = st.text_area(
    "Enter a social media post:"
)

if st.button("Analyze Post"):

    if not user_text.strip():
        st.warning("Please enter a post.")
    else:

        sentiment, score = analyze_sentiment(
            clean_text(user_text)
        )

        vectorizer, model, ids, labels = get_topic_model()

        cleaned = clean_text(user_text)
        vector = vectorizer.transform([cleaned])
        topic = model.predict(vector)[0] + 1

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"Sentiment: {sentiment}")
            st.write(f"Sentiment Score: {score}")

        with col2:
            st.info(f"Predicted Topic: Topic {topic}")