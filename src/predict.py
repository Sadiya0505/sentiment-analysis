import torch
import joblib
import re
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def predict_bert(text: str) -> dict:
    """Predict sentiment using BERT."""
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/bert")
    model     = DistilBertForSequenceClassification.from_pretrained("models/bert")
    model.to(device)
    model.eval()

    inputs  = tokenizer(text, truncation=True, padding=True,
                        max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred    = int(probs.argmax())

    return {
        "text":       text[:100] + "..." if len(text) > 100 else text,
        "sentiment":  "positive" if pred == 1 else "negative",
        "confidence": float(round(probs[pred], 4)),
        "scores":     {"negative": float(round(probs[0], 4)),
                       "positive": float(round(probs[1], 4))}
    }

def predict_classical(text: str, model_name: str = "logistic_regression") -> dict:
    """Predict sentiment using classical ML model."""
    vectorizer = joblib.load("data/processed/tfidf_vectorizer.pkl")
    model      = joblib.load(f"models/{model_name}.pkl")

    cleaned  = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred     = model.predict(features)[0]
    proba    = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None

    result = {
        "text":      text[:100] + "..." if len(text) > 100 else text,
        "sentiment": "positive" if pred == 1 else "negative",
        "model":     model_name
    }
    if proba is not None:
        result["confidence"] = float(round(proba[pred], 4))
    return result

if __name__ == "__main__":
    test_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Horrible acting.",
        "It was okay, nothing special but not bad either."
    ]

    print("=" * 50)
    print("BERT PREDICTIONS")
    print("=" * 50)
    for review in test_reviews:
        result = predict_bert(review)
        print(f"\nText      : {result['text']}")
        print(f"Sentiment : {result['sentiment']}")
        print(f"Confidence: {result['confidence']}")

    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION PREDICTIONS")
    print("=" * 50)
    for review in test_reviews:
        result = predict_classical(review)
        print(f"\nText      : {result['text']}")
        print(f"Sentiment : {result['sentiment']}")