from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"
TARGET_COLUMN = "diagnosis"
DATASET_NAME = "Breast Cancer Wisconsin (Diagnostic)"
DATASET_SOURCE = "UCI Machine Learning Repository"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_model_specs() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=8,
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            random_state=RANDOM_STATE,
        ),
    }


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.rename(TARGET_COLUMN)
    return features, target, list(dataset.target_names)


def prepare_split(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def evaluate_model(model: object, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "mcc": round(float(matthews_corrcoef(y_test, predictions)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            target_names=["malignant", "benign"],
        ),
    }


def save_model_bundle(
    model_name: str,
    fitted_model: object,
    feature_columns: list[str],
    target_names: list[str],
) -> str:
    filename = f"{slugify(model_name)}.joblib"
    output_path = MODEL_DIR / filename
    bundle = {
        "model_name": model_name,
        "dataset_name": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "target_column": TARGET_COLUMN,
        "target_names": target_names,
        "feature_columns": feature_columns,
        "model": fitted_model,
    }
    joblib.dump(bundle, output_path)
    return filename


def build_dataset_profile(features: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    combined = pd.concat([features, target], axis=1)
    class_counts = target.value_counts().sort_index()
    return {
        "dataset_name": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "rows": int(combined.shape[0]),
        "features": int(features.shape[1]),
        "target_column": TARGET_COLUMN,
        "class_distribution": {
            "malignant": int(class_counts.iloc[0]),
            "benign": int(class_counts.iloc[1]),
        },
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    features, target, target_names = load_dataset()
    x_train, x_test, y_train, y_test = prepare_split(features, target)

    test_frame = x_test.copy()
    test_frame[TARGET_COLUMN] = y_test
    test_frame.to_csv(TEST_DATA_PATH, index=False)

    metrics_catalog: dict[str, dict[str, object]] = {}
    comparison_rows: list[dict[str, object]] = []

    for model_name, estimator in build_model_specs().items():
        estimator.fit(x_train, y_train)
        metrics = evaluate_model(estimator, x_test, y_test)
        model_file = save_model_bundle(model_name, estimator, list(features.columns), target_names)
        metrics["model_file"] = model_file
        metrics_catalog[model_name] = metrics
        comparison_rows.append(
            {
                "ML Model Name": model_name,
                "Accuracy": metrics["accuracy"],
                "AUC": metrics["auc"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "MCC": metrics["mcc"],
            }
        )

    comparison_frame = pd.DataFrame(comparison_rows).sort_values(
        by=["Accuracy", "AUC", "F1", "MCC"],
        ascending=False,
    )
    comparison_frame.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)

    payload = {
        "dataset_profile": build_dataset_profile(features, target),
        "split_summary": {
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        "metrics": metrics_catalog,
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))

    print("Saved trained models, metrics, and test_data.csv")


if __name__ == "__main__":
    main()
