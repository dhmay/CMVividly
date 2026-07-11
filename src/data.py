from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def make_example_dataset(output_path: Path | None = None) -> Path:
    """Create a small synthetic dataset for the starter analysis.

    Replace this with a downloader/loader for your real public dataset.
    """
    if output_path is None:
        output_path = RAW_DIR / "example_observations.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-01", periods=18, freq="MS"),
            "group": ["A", "B", "C"] * 6,
            "value": [12, 15, 14, 18, 17, 21, 24, 22, 27, 29, 31, 34, 33, 35, 39, 43, 41, 46],
        }
    )
    df.to_csv(output_path, index=False)
    return output_path


def load_example_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the starter dataset, creating it if needed."""
    if path is None:
        path = RAW_DIR / "example_observations.csv"

    if not path.exists():
        make_example_dataset(path)

    return pd.read_csv(path, parse_dates=["month"])
