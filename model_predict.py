"""Compatibility shim for old root imports; logic lives in deuce.training.predict."""
from deuce.training.predict import *  # noqa: F401,F403
from deuce.training.predict import (  # noqa: F401
    PredictionResult,
    load_model,
    predict_match_probability,
    get_default_model_path,
    get_default_core_path,
    get_default_enhanced_path,
)
