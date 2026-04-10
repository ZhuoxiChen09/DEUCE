"""Training-time utilities: splits, calibration (no data leakage in API design)."""

from deuce.training.calibration import (
    CalibrationMethod,
    apply_calibration,
    fit_calibration,
)
from deuce.training.splits import (
    SplitStrategy,
    parse_match_date_int,
    split_train_val_indices,
    walk_forward_fold_indices,
)
from deuce.training.features import (
    CORE_FEATURE_NAMES,
    ENHANCED_FEATURE_NAMES,
    FEATURE_NAMES,
    DEFAULT_MEDIANS,
    PlayerStats,
    MatchFeatures,
    build_match_features,
    feature_dict_from_match_features,
    build_player_stats_from_datasets,
)
from deuce.training.train import (
    train_and_save,
    train_from_data_dir,
    load_csv_auto,
    fit_calibrated_lr_fold,
    _PlattCalibratedWrapper,
    _BundleCalibratedWrapper,
    main as train_main,
)
from deuce.training.predict import (
    load_model,
    predict_match_probability,
    get_default_model_path,
    get_default_core_path,
    get_default_enhanced_path,
    PredictionResult,
)

__all__ = [
    # calibration
    "CalibrationMethod",
    "apply_calibration",
    "fit_calibration",
    # splits
    "SplitStrategy",
    "parse_match_date_int",
    "split_train_val_indices",
    "walk_forward_fold_indices",
    # features
    "CORE_FEATURE_NAMES",
    "ENHANCED_FEATURE_NAMES",
    "FEATURE_NAMES",
    "DEFAULT_MEDIANS",
    "PlayerStats",
    "MatchFeatures",
    "build_match_features",
    "feature_dict_from_match_features",
    "build_player_stats_from_datasets",
    # train
    "train_and_save",
    "train_from_data_dir",
    "load_csv_auto",
    "fit_calibrated_lr_fold",
    "_PlattCalibratedWrapper",
    "_BundleCalibratedWrapper",
    "train_main",
    # predict
    "load_model",
    "predict_match_probability",
    "get_default_model_path",
    "get_default_core_path",
    "get_default_enhanced_path",
    "PredictionResult",
]
