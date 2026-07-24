from pathlib import Path
import pandas as pd

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

def load_tsv_pandas(tsv_path: Path) -> pd.DataFrame:
    """
    Load a TSV file into a pandas DataFrame.

    Args:
        tsv_path: Path to the TSV file.
    Returns:
        A pandas DataFrame containing the data from the TSV file.
    """
    return pd.read_csv(tsv_path, sep="\t")

def postprocess_cmv_ecocluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process the CMV ECOcluster DataFrame:
    add hla_class column with values 'ci' or 'cii'
    Args:
        df: A pandas DataFrame containing the CMV ECOcluster data. 
    Returns:
        A pandas DataFrame with the added hla_class column.
    """
    df = df.copy()
    df["hla_class"] = df["hla"].apply(
        lambda x: "cii" if x.startswith("D") else "ci")
    return df

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