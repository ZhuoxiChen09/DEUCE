import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from deuce.cli.main import main


ROOT = Path(__file__).resolve().parents[1]


def test_structured_cli_app_help_output(capsys):
    assert main(["app"]) == 0
    out = capsys.readouterr().out
    assert "streamlit run app.py" in out


def test_legacy_cli_help_runs():
    proc = subprocess.run(
        [sys.executable, "cli.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Kalshi public market data" in proc.stdout


def test_streamlit_help_runs():
    if importlib.util.find_spec("streamlit") is None:
        pytest.skip("streamlit is not installed in the active Python environment")
    proc = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Run a Python script, piping stderr to Streamlit" in proc.stdout
