"""
app.py — Flask web server for the Sentiment Analyzer.

Routes:
  GET  /          → Serve the frontend UI
  POST /predict   → Accept JSON { "text": "..." }, return sentiment prediction
  GET  /health    → Health check endpoint
"""

from flask import Flask, request, jsonify, render_template
from model import load_model, predict

app = Flask(__name__)

# Load model once at startup
print("Loading model...")
try:
    pipeline = load_model()
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    pipeline = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_sentiment():
    if pipeline is None:
        return jsonify({
            "error": "Model not loaded. Please run `python train.py` first."
        }), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400

    result = predict(text, pipeline)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline is not None
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
