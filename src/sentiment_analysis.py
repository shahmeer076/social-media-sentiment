import sqlite3
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from preprocessing import clean_text


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sentiment_analysis.db"

analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """Analyze sentiment and return label with compound score."""

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, compound


def analyze_database_posts():
    """Analyze all posts in the database and save sentiment results."""

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, text
        FROM social_media_posts
    """)

    posts = cursor.fetchall()

    for post_id, text in posts:

        cleaned_text = clean_text(text)

        sentiment, score = analyze_sentiment(cleaned_text)

        cursor.execute("""
            UPDATE social_media_posts
            SET sentiment = ?,
                sentiment_score = ?
            WHERE id = ?
        """, (sentiment, score, post_id))

    connection.commit()

    print(f"Sentiment analysis completed for {len(posts)} posts.")

    cursor.execute("""
        SELECT id, text, sentiment, sentiment_score
        FROM social_media_posts
    """)

    results = cursor.fetchall()

    for row in results:
        print(row)

    connection.close()


if __name__ == "__main__":
    analyze_database_posts()