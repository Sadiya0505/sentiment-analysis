from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import re
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment of text using BERT or classical ML models",
    version="1.0.0"
)

# ── Load models once at startup ────────────────────────────────
print("Loading models...")

# BERT
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained("models/bert")
bert      = DistilBertForSequenceClassification.from_pretrained("models/bert")
bert.to(device)
bert.eval()

# Classical
vectorizer = joblib.load("data/processed/tfidf_vectorizer.pkl")
lr_model   = joblib.load("models/logistic_regression.pkl")

print("All models loaded!")

# ── Request/Response schemas ───────────────────────────────────
class TextInput(BaseModel):
    text: str
    model: str = "bert"  # "bert" or "logistic_regression"

class PredictionOutput(BaseModel):
    text:       str
    sentiment:  str
    confidence: float
    model_used: str

# ── Helper ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# ── Routes ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!",
            "endpoints": ["/predict", "/health"]}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if input.model == "bert":
        inputs = tokenizer(input.text, truncation=True, padding=True,
                           max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = bert(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred    = int(probs.argmax())
        return PredictionOutput(
            text       = input.text[:100],
            sentiment  = "positive" if pred == 1 else "negative",
            confidence = float(round(probs[pred], 4)),
            model_used = "DistilBERT"
        )

    elif input.model == "logistic_regression":
        cleaned  = clean_text(input.text)
        features = vectorizer.transform([cleaned])
        pred     = lr_model.predict(features)[0]
        proba    = lr_model.predict_proba(features)[0]
        return PredictionOutput(
            text       = input.text[:100],
            sentiment  = "positive" if pred == 1 else "negative",
            confidence = float(round(proba[pred], 4)),
            model_used = "Logistic Regression"
        )

    else:
        raise HTTPException(status_code=400,
                            detail="model must be 'bert' or 'logistic_regression'")