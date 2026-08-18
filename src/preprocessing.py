import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    """Clean social media text for NLP analysis."""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [
        word for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(words)


if __name__ == "__main__":
    sample = "I absolutely LOVE this product!!!"
    print("Original:", sample)
    print("Cleaned:", clean_text(sample))