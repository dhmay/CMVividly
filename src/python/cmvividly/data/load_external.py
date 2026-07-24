from pathlib import Path
from urllib.request import Request, urlopen
import zipfile
from zipfile import ZipFile
import logging

logger = logging.getLogger(__name__)

from cmvividly.data.access import (
    RAW_DIR,
    CMV_ECOCLUSTER_2026_ZIP_PATH,
    CMV_ECOCLUSTER_2026_TSV_PATH,
    CMV_ECOCLUSTER_2024_ZIP_PATH,
    CMV_ECOCLUSTER_2024_TSV_PATH,
)

CMV_ECOCLUSTER_2026_ZIP_URL = (
    "https://www.biorxiv.org/content/biorxiv/early/2026/07/22/"
    "2024.03.26.583354/DC2/embed/media-2.zip?download=true"
)

CMV_ECOCLUSTER_2024_ZIP_URL = (
    "https://www.biorxiv.org/content/biorxiv/early/2024/05/10/"
    "2024.05.08.593237/DC1/embed/media-1.zip?download=true"
)

def download_cmv_ecocluster_2026(overwrite: bool = False,
                                 timeout_seconds: int = 60) -> Path:
    """
    Download the zipped 2026 CMV ECOcluster TSV from bioRxiv,
    unzip it, and return the path to the TSV file.
    Leaves the zipfile behind as detritus.

    Args:
        overwrite: If True, re-download the file even if it already exists.
        timeout_seconds: Timeout for the download request in seconds.
    Returns:
        Path to the downloaded TSV file.
    """
    return download_cmv_ecocluster_oneversion(
        zip_url=CMV_ECOCLUSTER_2026_ZIP_URL,
        zip_path=CMV_ECOCLUSTER_2026_ZIP_PATH,
        tsv_path=CMV_ECOCLUSTER_2026_TSV_PATH,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
    )

def load_all_external_datasets(overwrite: bool = False,
                               timeout_seconds: int = 60) -> None:
    """
    Download all external datasets.

    Args:
        overwrite: If True, re-download the files even if they already exist.
        timeout_seconds: Timeout for the download requests in seconds.
    """
    from cmvividly.data.access import load_cmv_ecocluster_2026, load_cmv_ecocluster_2024
    logger.info("Downloading 2026 CMV ECOcluster dataset...")
    load_cmv_ecocluster_2026(overwrite=overwrite, timeout_seconds=timeout_seconds),
    logger.info("Downloading 2024 CMV ECOcluster dataset...")
    load_cmv_ecocluster_2024(overwrite=overwrite, timeout_seconds=timeout_seconds),
    logger.info("All external datasets downloaded successfully.")

def download_cmv_ecocluster_2024(overwrite: bool = False,
                                 timeout_seconds: int = 60) -> Path:
    """
    Download the zipped 2024 CMV ECOcluster TSV from bioRxiv,
    unzip it, and return the path to the TSV file.
    Leaves the zipfile behind as detritus.

    Args:
        overwrite: If True, re-download the file even if it already exists.
        timeout_seconds: Timeout for the download request in seconds.
    Returns:
        Path to the downloaded TSV file.
    """
    return download_cmv_ecocluster_oneversion(
        zip_url=CMV_ECOCLUSTER_2024_ZIP_URL,
        zip_path=CMV_ECOCLUSTER_2024_ZIP_PATH,
        tsv_path=CMV_ECOCLUSTER_2024_TSV_PATH,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
    )


def download_cmv_ecocluster_oneversion(
    zip_url: str,
    zip_path: Path,
    tsv_path: Path,
    overwrite: bool = False,
    timeout_seconds: int = 60,
) -> Path:
    """
    Download the zipped CMV ECOcluster TSV from bioRxiv,
    unzip it, and return the path to the TSV file.
    Leaves the zipfile behind as detritus.

    Args:
        overwrite: If True, re-download the file even if it already exists.
        timeout_seconds: Timeout for the download request in seconds.
    Returns:
        Path to the downloaded TSV file.
    """
    output_dir = RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if tsv_path.exists() and not overwrite:
        return tsv_path

    zip_path = zip_path

    req = Request(
        zip_url,
        headers={"User-Agent": "CMVividly/1.0"},
    )

    with urlopen(req, timeout=timeout_seconds) as response, open(zip_path, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded file is not a valid zip archive: {zip_path}")
    # unzip it
    cmv_ecocluster_zip = ZipFile(zip_path, "r")
    cmv_ecocluster_zip.extractall(output_dir)

    # rename the extracted file to the expected name
    orig_name = cmv_ecocluster_zip.namelist()[0]
    orig_path = output_dir / orig_name
    Path(orig_path).rename(tsv_path)
    return tsv_path