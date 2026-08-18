import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from preprocessing import clean_text


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sentiment_analysis.db"


def get_posts():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, text
        FROM social_media_posts
    """)

    posts = cursor.fetchall()
    connection.close()

    return posts


def perform_topic_modeling():
    posts = get_posts()

    ids = [post[0] for post in posts]
    texts = [clean_text(post[1]) for post in posts]

    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    number_of_topics = 3

    model = KMeans(
        n_clusters=number_of_topics,
        random_state=42,
        n_init=10
    )

    topic_labels = model.fit_predict(tfidf_matrix)

    terms = vectorizer.get_feature_names_out()

    print("\nDiscovered Topics:")
    print("=" * 50)

    for topic_number in range(number_of_topics):
        center = model.cluster_centers_[topic_number]
        top_indices = center.argsort()[-5:][::-1]

        top_words = [
            terms[index]
            for index in top_indices
        ]

        print(
            f"Topic {topic_number + 1}: "
            + ", ".join(top_words)
        )

    print("\nPost Topic Assignments:")
    print("=" * 50)

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    for post_id, topic in zip(ids, topic_labels):

        print(
            f"Post {post_id} → Topic {topic + 1}"
        )

        cursor.execute("""
            UPDATE social_media_posts
            SET topic = ?
            WHERE id = ?
        """, (topic + 1, post_id))

    connection.commit()
    connection.close()


if __name__ == "__main__":
    perform_topic_modeling()