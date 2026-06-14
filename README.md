# Sentiment Analysis — End-to-End ML Project

A complete sentiment analysis system trained on 50,000 IMDB movie reviews.
Covers the full ML lifecycle: data preprocessing, classical ML, deep learning, 
evaluation, and REST API deployment.

## Results

| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| Naive Bayes         | 86.2%    | 86.2%    |
| Linear SVM          | 88.2%    | 88.2%    |
| Logistic Regression | 89.1%    | 89.1%    |
| LSTM                | 74.8%    | -        |
| **DistilBERT**      | **91.4%**| **91.4%**|

## Project Structure

sentiment-analysis/

├── data/

│   ├── raw/          ← IMDB dataset

│   └── processed/    ← TF-IDF vectors

├── src/

│   ├── preprocess.py       ← Text cleaning + TF-IDF

│   ├── train_classical.py  ← LR, Naive Bayes, SVM

│   ├── train_lstm.py       ← PyTorch LSTM

│   ├── train_bert.py       ← DistilBERT fine-tuning

│   ├── evaluate.py         ← Model comparison

│   ├── predict.py          ← Inference scripts

│   └── utils.py            ← Data download + EDA

├── api/

│   └── main.py       ← FastAPI REST API

├── models/           ← Saved model weights

├── requirements.txt

└── Dockerfile

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and preprocess data
```bash
python src/utils.py
python src/preprocess.py
```

### 3. Train classical models
```bash
python src/train_classical.py
```

### 4. Run predictions
```bash
python src/predict.py
```

### 5. Start the API
```bash
uvicorn api.main:app --reload
```

API will be live at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

## API Usage

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was fantastic!", "model": "bert"}'
```

Response:
```json
{
  "text": "This movie was fantastic!",
  "sentiment": "positive",
  "confidence": 0.9981,
  "model_used": "DistilBERT"
}
```

## Tech Stack

- **Data**: HuggingFace Datasets (IMDB, 50k reviews)
- **Classical ML**: scikit-learn (Logistic Regression, Naive Bayes, SVM)
- **Deep Learning**: PyTorch (LSTM), HuggingFace Transformers (DistilBERT)
- **API**: FastAPI + Uvicorn
- **Deployment**: Docker ready