"""Portfolio-facing Streamlit app for the DEUCE demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from deuce.demo.artifacts import BUILD_COMMAND, DemoArtifacts, discover_demo_artifacts
from deuce.demo.dataset import summarize_demo_dataset
from deuce.player_db import DEFAULT_DB_PATH, load_db
from deuce.presentation.demo_helpers import (
    accuracy_by_confidence_bucket,
    confidence_label,
    deterministic_explanation,
    grouped_contribution_rows,
    load_csv_rows,
    load_json,
    metric_value,
    percent,
    top_feature_rows,
)
from deuce.training.features import build_player_stats_from_datasets
from deuce.training.predict import load_model, predict_match_probability


ACCENT = "#2F855A"
ROOT = Path(__file__).resolve().parents[2]
DEMO_DATASET_PATH = ROOT / "data" / "demo" / "demo_matches.csv"
DEMO_ARTIFACTS = discover_demo_artifacts(ROOT)


def run() -> None:
    """Render the portfolio demo."""
    st.set_page_config(page_title="DEUCE", page_icon="🎾", layout="wide")
    _apply_css()

    page = _navigation()
    if page == "Overview":
        _overview_page(DEMO_ARTIFACTS)
    elif page == "Match Predictor":
        _match_predictor_page(DEMO_ARTIFACTS)
    elif page == "Backtest Explorer":
        _backtest_page(DEMO_ARTIFACTS)
    else:
        _methodology_page()


def _navigation() -> str:
    st.sidebar.markdown("## DEUCE")
    st.sidebar.caption("A transparent tennis match prediction and backtesting lab.")
    return st.sidebar.radio(
        "Navigation",
        ["Overview", "Match Predictor", "Backtest Explorer", "Methodology"],
        index=0,
    )


def _apply_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: #F8FAF8;
            color: #17201A;
        }}
        .block-container {{
            padding-top: 1.35rem;
            padding-bottom: 2rem;
            max-width: 1160px;
        }}
        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #E4E9E3;
        }}
        div[data-testid="stMetric"] {{
            background: #FFFFFF;
            border: 1px solid #E2E8E1;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            min-height: 104px;
        }}
        div[data-testid="stMetricLabel"] p {{
            white-space: normal;
            line-height: 1.2;
        }}
        .deuce-hero {{
            padding: 1.25rem 0 0.75rem 0;
            border-bottom: 1px solid #E2E8E1;
            margin-bottom: 1.5rem;
        }}
        .deuce-title {{
            font-size: 3rem;
            line-height: 1;
            font-weight: 760;
            letter-spacing: 0;
            color: #102116;
        }}
        .deuce-subtitle {{
            font-size: 1.15rem;
            color: #496353;
            margin-top: 0.35rem;
        }}
        .deuce-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8E1;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }}
        .deuce-result {{
            background: #EDF7F0;
            border: 1px solid #B8D8C1;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin: 0.45rem 0 0.8rem 0;
            color: #173D24;
        }}
        .deuce-result strong {{
            font-size: 1.15rem;
        }}
        h1 {{
            margin-bottom: 0.15rem;
        }}
        h3 {{
            margin-top: 1.25rem;
        }}
        .deuce-accent {{
            color: {ACCENT};
            font-weight: 700;
        }}
        .stButton button {{
            border-radius: 8px;
            border-color: {ACCENT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _artifact_warning(artifacts: DemoArtifacts) -> None:
    st.warning(artifacts.status_message())
    st.code(BUILD_COMMAND, language="powershell")


def _load_demo_summary() -> dict[str, Any]:
    if not DEMO_DATASET_PATH.exists():
        return {}
    return summarize_demo_dataset(DEMO_DATASET_PATH)


def _overview_page(artifacts: DemoArtifacts) -> None:
    st.markdown(
        """
        <div class="deuce-hero">
          <div class="deuce-title">DEUCE</div>
          <div class="deuce-subtitle">A transparent tennis match prediction and backtesting lab.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "DEUCE turns historical tennis match records into pre-match features, trains a calibrated "
        "classification model, and evaluates predictions on later unseen matches."
    )

    dataset = _load_demo_summary()
    metrics = load_json(artifacts.metrics_path)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demo matches", f"{dataset.get('row_count', 'Unavailable')}")
    c2.metric("OOS predictions", f"{metrics.get('n_oos_predictions', 'Unavailable')}")
    c3.metric("Accuracy", percent(metrics.get("accuracy")))
    c4.metric("ROC-AUC", metric_value(metrics.get("roc_auc")))
    st.caption(
        "All evaluation metrics are walk-forward and out of sample. "
        f"Brier score: {metric_value(metrics.get('brier'))} (lower is better)."
    )

    if not artifacts.ready:
        _artifact_warning(artifacts)

    st.markdown("### How It Works")
    steps = [
        ("1", "Build", "Transform historical records into pre-match features."),
        ("2", "Train", "Fit and calibrate a logistic regression model."),
        ("3", "Evaluate", "Test on later matches using walk-forward folds."),
        ("4", "Explain", "Surface the strongest deterministic model signals."),
    ]
    for column, (number, label, detail) in zip(st.columns(4), steps):
        column.markdown(
            f'<div class="deuce-card"><span class="deuce-accent">{number}. {label}</span><br>{detail}</div>',
            unsafe_allow_html=True,
        )

    st.info(
        "DEUCE is an educational sports-analytics project. It is not betting advice "
        "and does not use live odds or market data."
    )


@st.cache_data(show_spinner=False)
def _cached_player_db() -> list[dict[str, Any]]:
    return load_db(str(ROOT / DEFAULT_DB_PATH))


@st.cache_data(show_spinner=False)
def _cached_demo_rows() -> list[dict[str, str]]:
    return load_csv_rows(DEMO_DATASET_PATH)


@st.cache_data(show_spinner=False)
def _cached_catalog() -> list[dict[str, Any]]:
    path = ROOT / "data" / "tournament_catalog.json"
    try:
        from deuce.tournament_catalog import load_catalog

        return load_catalog(str(path))
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def _cached_model(model_path: str) -> tuple[Any, list[str], dict[str, float]]:
    return load_model(model_path)


def _player_options() -> list[str]:
    db = _cached_player_db()
    names_in_db = {str(row.get("player") or "").strip() for row in db if row.get("player")}
    names_in_demo = {
        str(row.get("player_a") or "").strip()
        for row in _cached_demo_rows()
        if row.get("player_a")
    } | {
        str(row.get("player_b") or "").strip()
        for row in _cached_demo_rows()
        if row.get("player_b")
    }
    options = sorted(names_in_db & names_in_demo, key=str.lower)
    if len(options) >= 2:
        return options
    return sorted(names_in_db, key=str.lower)


def _default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    try:
        return options.index(preferred)
    except ValueError:
        return min(fallback, max(0, len(options) - 1))


def _surface_context_name(surface: str, catalog: list[dict[str, Any]]) -> str:
    target = {
        "Hard": ("hard_outdoor", "hard_indoor"),
        "Clay": ("clay",),
        "Grass": ("grass",),
    }[surface]
    for entry in catalog:
        if entry.get("surface") in target:
            return str(entry.get("name") or "")
    return "Wimbledon" if surface == "Grass" else surface


def _match_predictor_page(artifacts: DemoArtifacts) -> None:
    st.title("Match Predictor")
    st.caption("Compare two players with the committed, calibrated demo model. No live data is used.")
    if not artifacts.ready:
        _artifact_warning(artifacts)
        return

    options = _player_options()
    if len(options) < 2:
        st.error("Player data is unavailable. Rebuild demo artifacts and confirm data/player_stats.json exists.")
        return

    default_a = options[_default_index(options, "Carlos Alcaraz", 0)]
    default_b = options[_default_index(options, "Jannik Sinner", 1)]
    st.session_state.setdefault("demo_player_a", default_a)
    st.session_state.setdefault("demo_player_b", default_b)

    def _swap_players() -> None:
        st.session_state["demo_player_a"], st.session_state["demo_player_b"] = (
            st.session_state["demo_player_b"],
            st.session_state["demo_player_a"],
        )

    left, mid, right, surface_col = st.columns([1, 0.35, 1, 0.72])
    with left:
        player_a = st.selectbox("Player A", options, key="demo_player_a")
    with mid:
        st.write("")
        st.write("")
        st.button("Swap", use_container_width=True, on_click=_swap_players)
    with right:
        player_b = st.selectbox("Player B", options, key="demo_player_b")
    with surface_col:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], index=0)

    if not player_a or not player_b or player_a == player_b:
        st.info("Choose two different players to generate a prediction.")
        return

    try:
        model, feature_names, medians = _cached_model(str(artifacts.model_path))
        db = _cached_player_db()
        catalog = _cached_catalog()
        stats_a = build_player_stats_from_datasets(player_a, player_db=db)
        stats_b = build_player_stats_from_datasets(player_b, player_db=db)
        context_name = _surface_context_name(str(surface), catalog)
        result = predict_match_probability(
            stats_a,
            stats_b,
            model_core=model,
            feature_names=feature_names,
            medians=medians,
            tournament_name=context_name,
            catalog=catalog,
        )
    except Exception as exc:
        st.error(f"Prediction could not be generated: {exc}")
        return

    winner = player_a if result.model_prob_a >= result.model_prob_b else player_b
    conf, conf_rule = confidence_label(result.model_prob_a)
    st.markdown(
        f'<div class="deuce-result">Predicted winner: <strong>{winner}</strong></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Player A win probability", percent(result.model_prob_a))
    cols[1].metric("Player B win probability", percent(result.model_prob_b))
    cols[2].metric("Confidence", conf, help=conf_rule)
    cols[3].metric("Data completeness", percent(result.data_completeness, digits=0))

    explanation = deterministic_explanation(
        player_a=player_a,
        player_b=player_b,
        probability_a=result.model_prob_a,
        contributions=result.per_feature_contributions or result.top_contributions,
        warnings=result.warnings,
        feature_values=result.feature_values,
        unavailable_features=set(
            result.diagnostics.get("unavailable_explanation_features") or []
        ),
    )
    st.markdown("### Explanation")
    st.write(explanation)
    st.caption(conf_rule)
    if result.warnings:
        st.caption("Model note: " + " ".join(result.warnings))

    feature_rows = top_feature_rows(
        result.per_feature_contributions or result.top_contributions,
        feature_values=result.feature_values,
        unavailable_features=set(
            result.diagnostics.get("unavailable_explanation_features") or []
        ),
    )
    if feature_rows:
        st.markdown("### Strongest Factors")
        df = pd.DataFrame(feature_rows)
        factor_chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("Contribution:Q", title="Contribution toward Player A (+) or Player B (-)"),
                y=alt.Y("Signal:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.Contribution >= 0,
                    alt.value(ACCENT),
                    alt.value("#64748B"),
                ),
                tooltip=[
                    alt.Tooltip("Signal:N"),
                    alt.Tooltip("Contribution:Q", format=".3f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(factor_chart, use_container_width=True)
    else:
        st.info("Feature contribution details are unavailable for this model artifact.")

    grouped_rows = grouped_contribution_rows(result.grouped_contributions)
    if grouped_rows:
        with st.expander("Model signal groups"):
            gdf = pd.DataFrame(grouped_rows)
            st.dataframe(gdf, use_container_width=True, hide_index=True)
            st.caption(
                "Observed groups exclude unavailable raw factors. Missing-data effects are "
                "shown separately as missing-data indicators."
            )


def _backtest_page(artifacts: DemoArtifacts) -> None:
    st.title("Backtest Explorer")
    st.caption("Walk-forward testing trains on earlier matches, then predicts later unseen matches.")
    if not artifacts.ready:
        _artifact_warning(artifacts)
        return

    metrics = load_json(artifacts.metrics_path)
    report = load_json(artifacts.report_path)
    oos_rows = load_csv_rows(artifacts.oos_predictions_path)
    cols = st.columns(5)
    cols[0].metric("OOS predictions", f"{metrics.get('n_oos_predictions', 'Unavailable')}")
    cols[1].metric("Accuracy", percent(metrics.get("accuracy")))
    cols[2].metric("ROC-AUC", metric_value(metrics.get("roc_auc")))
    cols[3].metric("Brier score", metric_value(metrics.get("brier")), help="Lower is better; measures probability calibration error.")
    cols[4].metric("Log loss", metric_value(metrics.get("log_loss")), help="Lower is better; strongly penalizes confident wrong probabilities.")

    cal = metrics.get("calibration_bins") or []
    cal_rows = [row for row in cal if row.get("avg_predicted") is not None and row.get("observed_rate") is not None]
    bucket_rows = accuracy_by_confidence_bucket(oos_rows)
    bdf = pd.DataFrame(bucket_rows)
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("### Calibration")
        if cal_rows:
            cdf = pd.DataFrame(cal_rows)
            cdf["Predicted probability"] = cdf["avg_predicted"].astype(float)
            cdf["Observed win rate"] = cdf["observed_rate"].astype(float)
            cdf["Perfect calibration"] = cdf["Predicted probability"]
            calibration_long = cdf.melt(
                id_vars=["Predicted probability"],
                value_vars=["Observed win rate", "Perfect calibration"],
                var_name="Series",
                value_name="Win rate",
            )
            calibration_chart = (
                alt.Chart(calibration_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "Predicted probability:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%"),
                    ),
                    y=alt.Y(
                        "Win rate:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%"),
                    ),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(
                            domain=["Observed win rate", "Perfect calibration"],
                            range=[ACCENT, "#94A3B8"],
                        ),
                        legend=alt.Legend(orient="bottom"),
                    ),
                    strokeDash=alt.condition(
                        alt.datum.Series == "Perfect calibration",
                        alt.value([5, 4]),
                        alt.value([1, 0]),
                    ),
                    tooltip=[
                        alt.Tooltip("Series:N"),
                        alt.Tooltip("Predicted probability:Q", format=".1%"),
                        alt.Tooltip("Win rate:Q", format=".1%"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(calibration_chart, use_container_width=True)
            st.caption("Closer to the dashed diagonal indicates better probability calibration.")
        else:
            st.info("Calibration bins are unavailable in the generated metrics file.")

    with chart_right:
        st.markdown("### Accuracy by Confidence")
        if not bdf.empty:
            chart_df = bdf.dropna(subset=["Accuracy"]).copy()
            confidence_chart = (
                alt.Chart(chart_df)
                .mark_bar(color=ACCENT, cornerRadiusEnd=3)
                .encode(
                    x=alt.X(
                        "Accuracy:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%"),
                    ),
                    y=alt.Y("Confidence bucket:N", sort=None, title="Distance from 50%"),
                    tooltip=[
                        alt.Tooltip("Confidence bucket:N"),
                        alt.Tooltip("Rows:Q"),
                        alt.Tooltip("Accuracy:Q", format=".1%"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(confidence_chart, use_container_width=True)
            st.caption("Confidence is the predicted winner's margin above a 50% probability.")
        else:
            st.info("Confidence-bucket results are unavailable.")

    st.markdown("### Predicted Probability Distribution")
    if oos_rows:
        probs = pd.DataFrame({"Predicted probability": [float(row["model_prob_a"]) for row in oos_rows]})
        st.bar_chart(probs["Predicted probability"].value_counts(bins=10, sort=False))
    else:
        st.info("OOS prediction rows are unavailable.")

    folds = report.get("folds") or []
    if folds:
        st.markdown("### Walk-Forward Folds")
        fold_rows = [
            {
                "Fold": fold.get("fold_index"),
                "Train rows": fold.get("n_train_fit"),
                "Calibration rows": fold.get("n_val_cal"),
                "Evaluation rows": fold.get("n_oos"),
                "OOS log loss": fold.get("oos_log_loss"),
                "OOS Brier": fold.get("oos_brier"),
            }
            for fold in folds
            if isinstance(fold, dict)
        ]
        st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)


def _methodology_page() -> None:
    st.title("Methodology")
    st.caption("How the demo is built, evaluated, and bounded.")
    left, right = st.columns(2)
    with left:
        st.markdown("### Data Scope")
        st.write(
            "A deterministic slice of 1,232 committed historical matches from 2018–2025, "
            "covering hard, clay, and grass courts. No live feeds are required."
        )

        st.markdown("### Feature Design")
        st.write(
            "Pre-match ranking, ranking points, surface and event context, draw status, "
            "player attributes, and explicit missing-data indicators. Score-derived fields "
            "are excluded from predictions."
        )

        st.markdown("### Model Family")
        st.write(
            "Standardized logistic regression with Platt calibration, producing a calibrated "
            "estimate of Player A's win probability."
        )

    with right:
        st.markdown("### Validation")
        st.write(
            "Five expanding walk-forward folds split on whole calendar dates, fit on earlier "
            "windows, calibrate on a held-out date range, and evaluate only on strictly later "
            "dates."
        )

        st.markdown("### Reproducibility")
        st.write(
            "Stable SHA-256 event hashing and a committed demo dataset make the build repeatable. "
            "The runtime bundle can be checked without retraining."
        )
        st.code(
            "python -m deuce.cli demo validate\npython -m streamlit run app.py",
            language="powershell",
        )

    st.markdown("### Metrics")
    st.write(
        "Accuracy measures how often the higher-probability player won. ROC-AUC measures ranking "
        "quality across thresholds. Brier score and log loss evaluate the quality of predicted "
        "probabilities, with lower values indicating better probability estimates."
    )

    st.markdown("### Limitations")
    st.write(
        "Historical records can be incomplete or noisy. Player form changes over time, and the "
        "model does not know injuries, withdrawals, tactical changes, travel disruptions, or live "
        "conditions. The demo uses no live inputs and is not a guarantee of future performance."
    )
