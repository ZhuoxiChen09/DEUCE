"""Model operations panel: structured sections beyond a flat metadata table."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from deuce.player_db import DEFAULT_DB_PATH, load_db


def render_model_operations_panel(
    core_ok: bool,
    core_path: Optional[str],
    meta: Optional[Dict[str, Any]],
) -> None:
    """Sections: bundle, training/calibration, data / DB freshness."""
    st.markdown("#### Active bundle summary")
    if not core_ok or not core_path:
        st.warning("Core bundle missing — predictions may use fallback routing.")
    else:
        st.code(str(Path(core_path).resolve()), language="text")

    st.markdown("#### Training & calibration")
    if not meta:
        st.info("No bundle metadata.")
    elif meta.get("error"):
        st.error(meta.get("error"))
    else:
        rows = [
            ("Format", meta.get("format")),
            ("Model tier", meta.get("model_tier")),
            ("Feature set name", meta.get("feature_set_name")),
            ("Feature count", meta.get("n_features")),
            ("Split strategy", meta.get("split_strategy")),
            ("Calibration method", meta.get("calibration_method")),
            ("Val log loss (calibrated)", meta.get("val_log_loss_calibrated")),
            ("Train window", str(meta.get("train_date_window"))),
            ("Validation window", str(meta.get("validation_date_window"))),
            ("Test / holdout window", str(meta.get("test_date_window"))),
            ("Samples (total / train / val / test)", "%s / %s / %s / %s" % (
                meta.get("n_samples_total"),
                meta.get("n_train"),
                meta.get("n_val"),
                meta.get("n_test_holdout"),
            )),
            ("Code version", meta.get("code_version")),
            ("Training timestamp", meta.get("created_at")),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Field", "Value"]), use_container_width=True, hide_index=True)

    st.markdown("#### Feature tier manifest")
    if meta and meta.get("feature_set_name"):
        st.success(f"Bundle declares feature set **{meta.get('feature_set_name')}** (tier aligns with training pipeline).")
    else:
        st.caption("Feature set name not present (legacy bundle).")

    st.markdown("#### Missingness / coverage (from bundle metadata)")
    if meta and not meta.get("error"):
        st.caption(
            "Per-feature missingness is computed at prediction time; see matchup **Developer** tab or PvP diagnostics. "
            "Training rows: see sample counts above."
        )
    else:
        st.caption("—")

    st.markdown("#### Player DB & local data freshness")
    p = Path(DEFAULT_DB_PATH)
    if p.exists():
        mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        try:
            db = load_db(DEFAULT_DB_PATH)
            n = len(db) if db else 0
        except Exception:
            n = "?"
        st.metric("Player DB rows", n)
        st.caption(f"`{p}` · last modified **{mtime}**")
    else:
        st.warning(f"Player DB not found at `{p}` — save ATP data to populate.")

    st.markdown("#### ATP tables in this session")
    bits = [
        f"Rankings: **{'yes' if st.session_state.get('atp_data') else 'no'}**",
        f"Serve: **{'yes' if st.session_state.get('serve_data') else 'no'}**",
        f"Return: **{'yes' if st.session_state.get('return_data') else 'no'}**",
        f"Pressure: **{'yes' if st.session_state.get('pressure_data') else 'no'}**",
    ]
    st.caption(" · ".join(bits))

    if meta:
        with st.expander("Raw bundle JSON"):
            st.json(meta)
