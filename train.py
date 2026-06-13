"""
train.py — Train the sentiment analysis model.

Uses the NLTK movie_reviews dataset (built-in, no download needed beyond NLTK data).
Trains a TF-IDF + Logistic Regression pipeline and saves it to disk.

Run:  python train.py
"""

import nltk
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from model import build_pipeline, save_model

# Download NLTK datasets on first run
nltk.download("movie_reviews", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import movie_reviews


def load_data():
    """Load NLTK movie reviews: returns (texts, labels)."""
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            words = movie_reviews.words(fileid)
            text = " ".join(words)
            documents.append((text, category))

    random.seed(42)
    random.shuffle(documents)

    texts = [doc[0] for doc in documents]
    labels = [doc[1] for doc in documents]
    return texts, labels


def main():
    print("Loading data...")
    texts, labels = load_data()
    print(f"  Total samples: {len(texts)}")
    print(f"  Classes: {set(labels)}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("\nBuilding and training pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    save_model(pipeline)
    print("\nDone! Model is ready.")


if __name__ == "__main__":
    main()
