"""Train and save multiple models for professional-domain classification."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_profiles.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["age", "education_level", "experience_years", "programming_score",
            "communication_score", "creativity_score", "analytical_score",
            "leadership_score", "design_score", "business_score", "science_score"]
TARGET = "professional_domain"

def train():
    if not DATA.exists():
        from generate_dataset import generate_dataset
        generate_dataset()
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    categorical = ["education_level"]
    numeric = [c for c in FEATURES if c not in categorical]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=220, max_depth=16, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=220, max_depth=18, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced")
    }
    results = {}
    best_name, best_score, best_pipeline = None, -1, None
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = accuracy_score(y_test, pred)
        results[name] = {
            "accuracy": round(float(score), 4),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique())).tolist()
        }
        if score > best_score:
            best_name, best_score, best_pipeline = name, score, pipe
        print(f"{name}: {score:.3f}")
    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")
    joblib.dump(best_pipeline, MODEL_DIR / "career_model.joblib")
    with open(MODEL_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "best_accuracy": best_score,
                   "models": results, "classes": sorted(y.unique())}, f, indent=2)
    print(f"Saved best model: {best_name} ({best_score:.3f})")
    return best_pipeline, results

if __name__ == "__main__":
    train()
