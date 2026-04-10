"""Compatibility shim — logic lives in deuce.training.train."""
from deuce.training.train import *  # noqa: F401,F403
from deuce.training.train import (  # noqa: F401
    _PlattCalibratedWrapper,
    _BundleCalibratedWrapper,
    fit_calibrated_lr_fold,
    load_atp_match_csv,
    load_historical_csv,
    load_csv_auto,
    train_and_save,
    train_from_data_dir,
    main,
    CODE_VERSION,
)
