"""Model-evaluation report section."""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from dashboard.common import (
    cached_evaluation_report,
    error_message,
    file_token,
    page_header,
    relative_label,
    select_file,
    set_evaluation_context,
)
from dashboard.data_access import (
    evaluation_reports,
    max_upload_bytes,
    prepare_json_upload,
)
from src.utils.errors import GeoAIError
from src.visualization.charts import confusion_matrix_chart, feature_importance_chart

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_macro": "Macro precision",
    "recall_macro": "Macro recall",
    "f1_macro": "Macro F1",
    "precision_weighted": "Weighted precision",
    "recall_weighted": "Weighted recall",
    "f1_weighted": "Weighted F1",
}


def _show_report(path: str, settings: Settings) -> None:
    report = cached_evaluation_report(path, file_token(path))
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        st.error("Evaluation report does not contain a metrics object.")
        return

    columns = st.columns(4)
    for column, key in zip(
        columns,
        ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"),
        strict=True,
    ):
        value = metrics.get(key)
        column.metric(METRIC_LABELS[key], f"{float(value):.4f}" if value is not None else "—")

    st.write(f"**Source:** `{relative_label(path, settings)}`")
    st.write(f"**Held-out samples:** {int(report.get('n_samples', 0)):,}")

    st.dataframe(
        [
            {"metric": METRIC_LABELS.get(key, key), "value": round(float(value), 6)}
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        ],
        use_container_width=True,
        hide_index=True,
    )

    try:
        st.plotly_chart(confusion_matrix_chart(report), use_container_width=True)
    except GeoAIError as exc:
        error_message("Unable to render the confusion matrix", exc)

    per_class = report.get("per_class")
    if isinstance(per_class, list) and per_class:
        st.subheader("Per-class metrics")
        st.dataframe(
            [
                {
                    "class": item.get("name", "—"),
                    "code": item.get("code", "—"),
                    "precision": round(float(item.get("precision", 0.0)), 6),
                    "recall": round(float(item.get("recall", 0.0)), 6),
                    "f1": round(float(item.get("f1", 0.0)), 6),
                    "support": int(item.get("support", 0)),
                }
                for item in per_class
            ],
            use_container_width=True,
            hide_index=True,
        )

    imports = report.get("feature_importances")
    if isinstance(imports, list) and imports:
        try:
            st.plotly_chart(feature_importance_chart(report), use_container_width=True)
        except GeoAIError as exc:
            error_message("Unable to render feature importances", exc)
    else:
        st.info("This evaluation report does not contain feature importances.")

    set_evaluation_context({"source_path": path, "report": report})
    st.caption(
        "Metrics describe predictions on the recorded held-out samples. "
        "They are not evidence of accuracy over every pixel or over new regions."
    )


def render(settings: Settings) -> None:
    page_header(
        "Model Evaluation",
        "Review held-out metrics, confusion matrices, per-class performance, and importances.",
    )
    records = evaluation_reports(settings)
    existing_tab, upload_tab = st.tabs(["Existing reports", "Upload report"])

    with existing_tab:
        selected = select_file(
            records,
            settings,
            label="Evaluation report JSON",
            key="evaluation-existing",
        )
        if selected is not None:
            try:
                _show_report(str(selected.path), settings)
            except GeoAIError as exc:
                error_message("Unable to load the evaluation report", exc)

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload an evaluation report",
            type=["json"],
            key="evaluation-upload",
        )
        if uploaded is not None:
            maximum = max_upload_bytes(settings)
            if uploaded.size > maximum:
                st.error(
                    f"Upload is {uploaded.size / (1024 * 1024):.1f} MB; "
                    f"the limit is {settings.max_upload_mb} MB."
                )
            elif st.button("Save evaluation upload", type="primary", key="save-evaluation"):
                try:
                    path = prepare_json_upload(uploaded.name, uploaded.getvalue(), settings)
                    cached_evaluation_report(str(path), file_token(path))
                    st.success(f"Saved upload to `{relative_label(path, settings)}`.")
                except GeoAIError as exc:
                    error_message("Unable to use the uploaded evaluation report", exc)


__all__ = ["render"]
