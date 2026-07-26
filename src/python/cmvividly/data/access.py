from pathlib import Path
import pandas as pd

from cmvividly.data.hamming1_pairs import find_cdr3_hamming1_pairs

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

def postprocess_cmv_ecocluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process a CMV ECOcluster DataFrame (either 2026 or 2024):
    * add hla_class column with values 'ci' or 'cii'
    * add "cdr3", "vgene" and "jgene" columns by splitting the "tcr" column on "+"
    * add "log10_pgen_eps" column by computing log10(pgen + 1e-50) to avoid log(0)
    Args:
        df: A pandas DataFrame containing the CMV ECOcluster data. 
    Returns:
        A pandas DataFrame with the added hla_class column.
    """
    df = df.copy()
    df["hla_class"] = df["hla"].apply(
        lambda x: "cii" if x.startswith("D") else "ci")
    df[["cdr3", "vgene", "jgene"]] = df["tcr"].str.split("+", expand=True)
    import numpy as np
    df["log10_tcr_pgen_eps"] = (df["tcr_pgen"] + 1e-50).apply(lambda x: np.log10(x))
    # rearrange the columns: tcr, cdr3, vgene, jgene, hla, hla_class, tcr_pgen, log10_tcr_pgen_eps and the rest
    first_cols = ["tcr", "cdr3", "vgene", "jgene", "hla", "hla_class", "tcr_pgen", "log10_tcr_pgen_eps"]
    cols = first_cols + [c for c in df.columns if c not in first_cols]
    return df[cols]

