import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_data():
    X_train, y_train = joblib.load("data/processed/train.pkl")
    X_test,  y_test  = joblib.load("data/processed/test.pkl")
    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{name.lower().replace(' ', '_')}.pkl")
    print(f"Saved model to models/{name.lower().replace(' ', '_')}.pkl")

    return acc, f1

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    models = [
        (LogisticRegression(max_iter=1000, C=1.0), "Logistic Regression"),
        (MultinomialNB(alpha=0.1),                 "Naive Bayes"),
        (LinearSVC(max_iter=1000, C=1.0),          "Linear SVM"),
    ]

    results = []
    for model, name in models:
        acc, f1 = train_and_evaluate(model, name, X_train, y_train, X_test, y_test)
        results.append((name, acc, f1))

    print("\n" + "="*45)
    print(f"{'Model':<25} {'Accuracy':>8} {'F1':>8}")
    print("="*45)
    for name, acc, f1 in results:
        print(f"{name:<25} {acc:>8.4f} {f1:>8.4f}")
    print("="*45)