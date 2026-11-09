"""Download the UCI red-wine dataset, train a Random Forest model, and evaluate it."""
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_FILE = Path("wine_quality.csv")
MODEL_FILE = Path("wine_model.pkl")
METRICS_FILE = Path("evaluation_metrics.txt")

def main():
    print("Downloading the UCI red-wine dataset...")
    df = pd.read_csv(DATA_URL, sep=";")
    df.to_csv(DATA_FILE, index=False)
    print("Dataset shape:", df.shape)
    print(df.head())
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)  # 1 = Good; 0 = Needs improvement
    X = df.drop(columns=["quality"])     # Prevent target leakage

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    report = classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Needs improvement", "Good quality"],
        zero_division=0
    )
    metrics_text = (
        "WINE QUALITY MODEL EVALUATION\n"
        f"Accuracy:  {accuracy:.4f}\n"
        f"Precision (GOOD QUALITY): {precision:.4f}\n"
        f"Recall (GOOD QUALITY):    {recall:.4f}\n"
        f"F1-score (GOOD QUALITY):  {f1:.4f}\n\n"
        "Confusion matrix (rows=true, columns=predicted; labels 0,1):\n"
        f"{confusion_matrix(y_test, predictions, labels=[0, 1])}\n\n"
        "Classification report:\n" + report
    )
    print(metrics_text)
    METRICS_FILE.write_text(metrics_text, encoding="utf-8")

    joblib.dump(model, MODEL_FILE)
    # Create a small valid input file for testing the Streamlit upload.
    X.head(5).to_csv("sample_wine_input.csv", index=False)
    print(f"Saved {DATA_FILE}, {MODEL_FILE}, {METRICS_FILE}, and sample_wine_input.csv")

if __name__ == "__main__":
    main()
