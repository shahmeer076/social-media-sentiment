import sqlite3
from pathlib import Path
import pandas as pd


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sentiment_analysis.db"
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "social_media_posts.csv"


def get_connection():
    """Create and return a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def create_table(location TEXT,
sentiment TEXT,
sentiment_score REAL):
    """Create the social_media_posts table if it does not exist."""
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS social_media_posts (
            id INTEGER PRIMARY KEY,
            platform TEXT NOT NULL,
            username TEXT,
            text TEXT NOT NULL,
            created_at TEXT,
            likes INTEGER DEFAULT 0,
            comments INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            location TEXT
        )
    """)

    connection.commit()
    connection.close()


def import_csv():
    """Import social media posts from CSV into SQLite."""
    df = pd.read_csv(CSV_PATH)

    connection = get_connection()

    df.to_sql(
        "social_media_posts",
        connection,
        if_exists="append",
        index=False
    )

    connection.close()

    print(f"{len(df)} records imported successfully.")


if __name__ == "__main__":
    create_table()
    import_csv()