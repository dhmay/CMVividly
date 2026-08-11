"""Methods for accessing data"""
from pathlib import Path
import pandas as pd

from cmvividly.data.hamming1_pairs import find_cdr3_hamming1_pairs
from cmvividly.data.manipulation import postprocess_cmv_ecocluster

# Walk up the directory tree to find the project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# CMV ECOcluster dataset locations
CMV_ECOCLUSTER_2026_ZIP_FILENAME = "cmv_ecocluster_2026.tsv.zip"
CMV_ECOCLUSTER_2026_TSV_FILENAME = "cmv_ecocluster_2026.tsv"
CMV_ECOCLUSTER_2026_ZIP_PATH = RAW_DIR / CMV_ECOCLUSTER_2026_ZIP_FILENAME
CMV_ECOCLUSTER_2026_TSV_PATH = RAW_DIR / CMV_ECOCLUSTER_2026_TSV_FILENAME

CMV_ECOCLUSTER_2024_ZIP_FILENAME = "cmv_ecocluster_2024.tsv.zip"
CMV_ECOCLUSTER_2024_TSV_FILENAME = "cmv_ecocluster_2024.tsv"
CMV_ECOCLUSTER_2024_ZIP_PATH = RAW_DIR / CMV_ECOCLUSTER_2024_ZIP_FILENAME
CMV_ECOCLUSTER_2024_TSV_PATH = RAW_DIR / CMV_ECOCLUSTER_2024_TSV_FILENAME

CMV_ECOCLUSTER_2026_CDR3_HAMMING1_PAIRS_TSV_FILENAME = "cmv_ecocluster_2026_cdr3_hamming1_pairs.tsv"
CMV_ECOCLUSTER_2026_CDR3_HAMMING1_PAIRS_TSV_PATH = PROCESSED_DIR / CMV_ECOCLUSTER_2026_CDR3_HAMMING1_PAIRS_TSV_FILENAME

EMERSON_REPERTOIRES_ZIP_FILENAME = "emerson-2017-natgen.zip"
EMERSON_REPERTOIRES_ZIP_PATH = RAW_DIR / EMERSON_REPERTOIRES_ZIP_FILENAME

EMERSON_METADATA_TSV_FILENAME = "emerson_cohorts_sample_metadata.tsv"
EMERSON_METADATA_TSV_PATH = PROCESSED_DIR / EMERSON_METADATA_TSV_FILENAME

CMV_ECOCLUSTER_INTERSECT_EMERSON_PARQUET_FILENAME = "emerson_cmveco_2026_intersection_cdr3_exact.parquet"
CMV_ECOCLUSTER_INTERSECT_EMERSON_PARQUET_PATH = PROCESSED_DIR / CMV_ECOCLUSTER_INTERSECT_EMERSON_PARQUET_FILENAME

def load_cmv_ecocluster_2026(overwrite: bool = False,
                             timeout_seconds: int = 60) -> pd.DataFrame:
    """
    Load the 2026 CMV ECOcluster dataset into a pandas DataFrame.
    Downloads the dataset if it does not exist locally.

    Args:
        overwrite: If True, re-download the file even if it already exists.
        timeout_seconds: Timeout for the download request in seconds.
    Returns:
        A pandas DataFrame containing the 2026 CMV ECOcluster data.
    """
    from cmvividly.data.load_external import download_cmv_ecocluster_2026

    tsv_path = download_cmv_ecocluster_2026(overwrite=overwrite, timeout_seconds=timeout_seconds)
    pdf_raw = load_tsv_pandas(tsv_path)
    pdf_postprocessed = postprocess_cmv_ecocluster(pdf_raw)
    return pdf_postprocessed

def load_cmv_ecocluster_2024(overwrite: bool = False,
                             timeout_seconds: int = 60) -> pd.DataFrame:
    """
    Load the 2024 CMV ECOcluster dataset into a pandas DataFrame.
    Downloads the dataset if it does not exist locally.

    Args:
        overwrite: If True, re-download the file even if it already exists.
        timeout_seconds: Timeout for the download request in seconds.   
    Returns:
        A pandas DataFrame containing the 2024 CMV ECOcluster data.
    """
    from cmvividly.data.load_external import download_cmv_ecocluster_2024

    tsv_path = download_cmv_ecocluster_2024(overwrite=overwrite, timeout_seconds=timeout_seconds)
    pdf_raw = load_tsv_pandas(tsv_path)
    pdf_postprocessed = postprocess_cmv_ecocluster(pdf_raw)
    return pdf_postprocessed


def load_cmv_ecocluster_2026_cdr3_hamming1_pairs(
        overwrite: bool = False,
        timeout_seconds: int = 60) -> pd.DataFrame:
    """
    Load the 2026 CMV ECOcluster and find
    all Hamming-1 CDR3 pairs into a pandas DataFrame.
    If the file already exists and overwite is False,
    just load the file.

    Args:
        overwrite: If True, re-download the file even if it already exists.
        timeout_seconds: Timeout for the download request in seconds.
    Returns:
        A pandas DataFrame containing the 2026 CMV ECOcluster CDR3 Hamming-1 pairs.
    """
    if not overwrite and CMV_ECOCLUSTER_2026_CDR3_HAMMING1_PAIRS_TSV_PATH.exists():
        return load_tsv_pandas(CMV_ECOCLUSTER_2026_CDR3_HAMMING1_PAIRS_TSV_PATH)
    pdf_cmv_ecocluster = load_cmv_ecocluster_2026(overwrite=overwrite, timeout_seconds=timeout_seconds)
    pdf_hamming1_pairs = find_cdr3_hamming1_pairs(pdf_cmv_ecocluster)
    pdf_hamming1_pairs.to_csv(CMV_ECOCLUSTER_2026_CDR3_HAMMING1_PAIRS_TSV_PATH, sep="\t", index=False,
                            mode="w", header=True)
    return pdf_hamming1_pairs

def load_tsv_pandas(tsv_path: Path) -> pd.DataFrame:
    """
    Load a TSV file into a pandas DataFrame.

    Args:
        tsv_path: Path to the TSV file.
    Returns:
        A pandas DataFrame containing the data from the TSV file.
    """
    return pd.read_csv(tsv_path, sep="\t")


def load_emerson_metadata() -> pd.DataFrame:
    """
    Load the Emerson 2017 metadata into a pandas DataFrame.

    Returns:
        A pandas DataFrame containing the Emerson 2017 metadata.
    """
    tsv_path = EMERSON_METADATA_TSV_PATH
    if not tsv_path.exists():
        raise FileNotFoundError(f"Emerson metadata file not found at {tsv_path}.")
    return load_tsv_pandas(tsv_path)

def load_cmv_ecocluster_intersect_emerson() -> pd.DataFrame:
    """
    Load the intersection of the 2026 CMV ECOcluster and Emerson 2017 repertoires (by TCR) into a pandas DataFrame.

    Returns:
        A pandas DataFrame containing the intersection of the 2026 CMV ECOcluster and Emerson 2017 repertoires.
    """
    parquet_path = CMV_ECOCLUSTER_INTERSECT_EMERSON_PARQUET_PATH
    if not parquet_path.exists():
        raise FileNotFoundError(f"CMV ECOcluster intersect Emerson file not found at {parquet_path}.")
    return pd.read_parquet(parquet_path)