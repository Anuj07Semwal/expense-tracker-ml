import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = "sentiment_model.pkl"


def build_pipeline():
    """Build a TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="auto"
        ))
    ])


def save_model(pipeline):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run `python train.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(text: str, pipeline) -> dict:
    """
    Predict sentiment for a single text string.
    Returns a dict with label and confidence scores.
    """
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    label = classes[proba.argmax()]
    confidence = round(float(proba.max()) * 100, 2)

    scores = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}

    return {
        "label": label,
        "confidence": confidence,
        "scores": scores
    }
