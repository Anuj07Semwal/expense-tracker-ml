# 🎯 Sentiment Analyzer

A machine learning web application that classifies text as **positive** or **negative** using a TF-IDF + Logistic Regression pipeline, deployed with a Flask REST API and an interactive frontend.

---

## 📌 Project Overview

This project demonstrates an end-to-end ML workflow:
1. **Data** — NLTK Movie Reviews dataset (2,000 labeled documents)
2. **Feature Engineering** — TF-IDF with bigrams and sublinear term frequency scaling
3. **Model** — Logistic Regression with probability outputs
4. **Deployment** — Flask REST API with a JSON `/predict` endpoint
5. **Frontend** — Vanilla JS UI with animated confidence bars

> **Use case relevance**: Sentiment analysis powers Amazon product reviews, Alexa feedback loops, and customer insight systems.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML | scikit-learn (TF-IDF, Logistic Regression) |
| Data | NLTK Movie Reviews corpus |
| Backend | Flask 3.0 |
| Frontend | HTML / CSS / Vanilla JS |

---

## 📂 Project Structure

```
sentiment-analyzer/
├── app.py              # Flask web server & REST API
├── model.py            # ML pipeline definition & prediction logic
├── train.py            # Training script (run once to build model)
├── requirements.txt    # Python dependencies
├── .gitignore
├── README.md
└── templates/
    └── index.html      # Interactive frontend UI
```

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train.py
```
This downloads the NLTK dataset, trains the model, and saves `sentiment_model.pkl`.

Expected output:
```
Accuracy: ~83%
```

### 5. Start the web server
```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🔌 API Reference

### `POST /predict`

**Request:**
```json
{
  "text": "This product exceeded all my expectations!"
}
```

**Response:**
```json
{
  "label": "pos",
  "confidence": 94.37,
  "scores": {
    "neg": 5.63,
    "pos": 94.37
  }
}
```

### `GET /health`
Returns `{ "status": "ok", "model_loaded": true }`

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~83% |
| Precision (pos) | ~84% |
| Recall (pos) | ~82% |
| F1 Score | ~83% |

Trained on 1,600 samples, evaluated on 400 held-out samples.

---

## 🔮 Future Improvements

- [ ] Upgrade to a fine-tuned BERT / DistilBERT model for higher accuracy
- [ ] Add a neutral sentiment class
- [ ] Deploy to AWS EC2 or AWS Lambda for cloud hosting
- [ ] Add batch prediction endpoint for bulk review analysis
- [ ] Integrate with Amazon Product Advertising API for live review analysis

---

## 👤 Author

**Your Name**  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
