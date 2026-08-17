from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
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


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_PATH = ROOT_DIR / "artifacts" / "metrics.json"
COMPARISON_PATH = ROOT_DIR / "artifacts" / "model_comparison.csv"
DEFAULT_TEST_DATA = ROOT_DIR / "test_data.csv"
TARGET_COLUMN = "diagnosis"


st.set_page_config(
    page_title="ML Assignment Multi-label Classification Model",
    layout="wide",
)


@st.cache_data
def load_metrics_payload() -> dict:
    return json.loads(ARTIFACTS_PATH.read_text())


@st.cache_data
def load_comparison_frame() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def load_default_test_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_TEST_DATA)


@st.cache_resource
def load_model_bundle(model_file: str) -> dict:
    return joblib.load(ROOT_DIR / "model" / model_file)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f1e8;
            --card: #fffaf3;
            --ink: #1f2933;
            --muted: #66788a;
            --accent: #c35b3f;
            --accent-soft: #f0c7b8;
            --line: #e4d7c8;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(195, 91, 63, 0.16), transparent 34%),
                radial-gradient(circle at right, rgba(67, 113, 129, 0.18), transparent 28%),
                var(--bg);
            color: var(--ink);
        }
        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem;
            box-shadow: 0 10px 30px rgba(31, 41, 51, 0.06);
        }
        .hero {
            background: linear-gradient(135deg, rgba(255,250,243,0.95), rgba(240,199,184,0.75));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 18px 35px rgba(31, 41, 51, 0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.3rem;
            line-height: 1.1;
        }
        .hero p {
            color: var(--muted);
            font-size: 1rem;
            margin-top: 0.8rem;
            margin-bottom: 0;
        }
        .section-card {
            background: rgba(255, 250, 243, 0.88);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.2rem;
            margin-top: 0.8rem;
        }
        .pill-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .pill {
            background: rgba(31, 41, 51, 0.06);
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_columns(dataframe: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Uploaded CSV is missing required feature columns: "
            + ", ".join(missing_columns[:6])
            + ("..." if len(missing_columns) > 6 else "")
        )
    ordered = dataframe[expected_columns].copy()
    if TARGET_COLUMN in dataframe.columns:
        ordered[TARGET_COLUMN] = dataframe[TARGET_COLUMN]
    return ordered


def evaluate_uploaded_data(model_bundle: dict, data: pd.DataFrame) -> tuple[dict[str, float] | None, pd.DataFrame, list[list[int]] | None]:
    estimator = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    prepared = ensure_columns(data, feature_columns)
    feature_frame = prepared[feature_columns]

    probabilities = estimator.predict_proba(feature_frame)[:, 1]
    predictions = estimator.predict(feature_frame)

    prediction_frame = prepared.copy()
    prediction_frame["predicted_label"] = predictions
    prediction_frame["predicted_class"] = prediction_frame["predicted_label"].map(
        {0: "malignant", 1: "benign"}
    )
    prediction_frame["benign_probability"] = probabilities.round(4)

    if TARGET_COLUMN not in prepared.columns:
        return None, prediction_frame, None

    y_true = prepared[TARGET_COLUMN]
    metric_values = {
        "Accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "AUC": round(float(roc_auc_score(y_true, probabilities)), 4),
        "Precision": round(float(precision_score(y_true, predictions)), 4),
        "Recall": round(float(recall_score(y_true, predictions)), 4),
        "F1": round(float(f1_score(y_true, predictions)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, predictions)), 4),
    }
    prediction_frame["actual_class"] = y_true.map({0: "malignant", 1: "benign"})
    prediction_frame["correct_prediction"] = predictions == y_true
    report = classification_report(
        y_true,
        predictions,
        output_dict=True,
        target_names=["malignant", "benign"],
    )
    prediction_frame.attrs["classification_report"] = pd.DataFrame(report).transpose()
    return metric_values, prediction_frame, confusion_matrix(y_true, predictions).tolist()


def render_confusion_matrix(matrix_values: list[list[int]]) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        matrix_values,
        annot=True,
        fmt="d",
        cmap="OrRd",
        cbar=False,
        xticklabels=["malignant", "benign"],
        yticklabels=["malignant", "benign"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig, clear_figure=True)


def render_metric_cards(metrics: dict[str, float]) -> None:
    columns = st.columns(6)
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, f"{value:.4f}")


def main() -> None:
    apply_theme()
    payload = load_metrics_payload()
    comparison_frame = load_comparison_frame()
    default_test_data = load_default_test_data()
    metrics_lookup = payload["metrics"]
    dataset_profile = payload["dataset_profile"]

    st.markdown(
        f"""
        <div class="hero">
            <h1>ML Assignment 2 Model Studio</h1>
            <p>
                Interactive evaluation lab for the {dataset_profile["dataset_name"]} dataset.
                Upload the provided test CSV, choose a classifier, and inspect performance,
                predictions, and confusion patterns in one place.
            </p>
            <div class="pill-row">
                <div class="pill">{dataset_profile["rows"]} rows</div>
                <div class="pill">{dataset_profile["features"]} features</div>
                <div class="pill">Binary classification</div>
                <div class="pill">5 trained models</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Experiment Controls")
        selected_model = st.selectbox("Choose a trained model", list(metrics_lookup.keys()))
        uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])
        use_uploaded_target = st.toggle("Uploaded file includes true labels", value=True)
        st.caption(
            "The sample test file shipped with this project already contains the diagnosis column, "
            "which lets the app compute metrics and the confusion matrix."
        )

    active_bundle = load_model_bundle(metrics_lookup[selected_model]["model_file"])
    source_data = pd.read_csv(uploaded_file) if uploaded_file else default_test_data.copy()
    if not use_uploaded_target and TARGET_COLUMN in source_data.columns:
        source_data = source_data.drop(columns=[TARGET_COLUMN])

    overview_tab, lab_tab, preview_tab = st.tabs(
        ["Model Comparison", "Interactive Evaluation", "Data Preview"]
    )

    with overview_tab:
        st.subheader("Overall Model Comparison")
        st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
        winner = comparison_frame.iloc[0]
        st.markdown(
            f"""
            <div class="section-card">
                <strong>Current winner:</strong> {winner["ML Model Name"]}<br>
                Best test-set accuracy: {winner["Accuracy"]:.4f} with AUC {winner["AUC"]:.4f}.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with lab_tab:
        st.subheader(f"Interactive Evaluation: {selected_model}")
        try:
            metrics, predictions, matrix_values = evaluate_uploaded_data(active_bundle, source_data)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        if metrics is None:
            st.info(
                "The uploaded file does not contain the true diagnosis labels, so the app is "
                "showing predictions only. Include the diagnosis column to unlock metrics."
            )
            st.dataframe(predictions.head(20), use_container_width=True)
        else:
            render_metric_cards(metrics)
            left_col, right_col = st.columns([1, 1.15])
            with left_col:
                render_confusion_matrix(matrix_values)
            with right_col:
                st.markdown("#### Classification Report")
                st.dataframe(
                    predictions.attrs["classification_report"].round(4),
                    use_container_width=True,
                )

            st.markdown("#### Prediction Snapshot")
            st.dataframe(
                predictions[
                    [
                        TARGET_COLUMN,
                        "actual_class",
                        "predicted_label",
                        "predicted_class",
                        "benign_probability",
                        "correct_prediction",
                    ]
                ].head(20),
                use_container_width=True,
            )

    with preview_tab:
        st.subheader("Preview the Active Evaluation File")
        st.dataframe(source_data.head(20), use_container_width=True)
        st.caption(
            "Tip: keep the feature names unchanged when uploading a custom CSV so the saved models "
            "can evaluate it correctly."
        )


if __name__ == "__main__":
    main()
