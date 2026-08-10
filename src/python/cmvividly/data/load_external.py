"""Methods for downloading external datasets"""

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
import socket
import time
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
    EMERSON_REPERTOIRES_ZIP_PATH
)

CMV_ECOCLUSTER_2026_ZIP_URL = (
    "https://www.biorxiv.org/content/biorxiv/early/2026/07/22/"
    "2024.03.26.583354/DC2/embed/media-2.zip?download=true"
)

CMV_ECOCLUSTER_2024_ZIP_URL = (
    "https://www.biorxiv.org/content/biorxiv/early/2024/05/10/"
    "2024.05.08.593237/DC1/embed/media-1.zip?download=true"
)

EMERSON_REPERTOIRES_ZIP_URL = (
    "https://adaptivepublic.blob.core.windows.net/publishedproject-supplements"
    "/emerson-2017-natgen/emerson-2017-natgen.zip"
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

    download_zipfile(zip_url, zip_path, timeout_seconds=timeout_seconds)

    # unzip it
    cmv_ecocluster_zip = ZipFile(zip_path, "r")
    cmv_ecocluster_zip.extractall(output_dir)

    # rename the extracted file to the expected name
    orig_name = cmv_ecocluster_zip.namelist()[0]
    orig_path = output_dir / orig_name
    Path(orig_path).rename(tsv_path)
    return tsv_path


def download_emerson_repertoires(overwrite: bool = False,
                                 timeout_seconds: int = 60) -> Path:
    """Download the "Emerson" CMV repertoires.

    Args:
        overwrite (bool, optional): _description_. Defaults to False.
        timeout_seconds (int, optional): _description_. Defaults to 60.

    Returns:
        Path: Path to the downloaded zip file.
    """
    if EMERSON_REPERTOIRES_ZIP_PATH.exists() and not overwrite:
        return EMERSON_REPERTOIRES_ZIP_PATH
    download_zipfile(
        url=EMERSON_REPERTOIRES_ZIP_URL,
        out_path=EMERSON_REPERTOIRES_ZIP_PATH,
        timeout_seconds=timeout_seconds,
    )
    return EMERSON_REPERTOIRES_ZIP_PATH


def download_zipfile(url: str, out_path: Path, timeout_seconds: int = 60) -> None:
    """Download a file and ensure it's a zip file

    Args:
        url (str): _description_
        out_path (Path): _description_
        timeout_seconds (int, optional): _description_. Defaults to 60.

    Raises:
        ValueError: _description_
    """
    download_file(url, out_path, timeout_seconds=timeout_seconds)
    if not zipfile.is_zipfile(out_path):
        out_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded file is not a valid zip archive: {out_path}")


def download_file(url: str, out_path: Path, timeout_seconds: int = 60,
                  max_retries: int = 3, backoff_seconds: float = 5) -> None:
    """Download a file from url into out_path, retrying on timeout/connection errors.

    Args:
        url (str): _description_
        out_path (Path): _description_
        timeout_seconds (int, optional): _description_. Defaults to 60.
        max_retries (int, optional): Number of attempts before giving up. Defaults to 3.
        backoff_seconds (float, optional): Seconds to wait before retrying, doubling
            after each failed attempt. Defaults to 5.
    """
    req = Request(
        url,
        headers={"User-Agent": "CMVividly/1.0"},
    )

    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(req, timeout=timeout_seconds) as response, open(out_path, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            return
        except (URLError, TimeoutError, socket.timeout) as e:
            if attempt == max_retries:
                raise
            logger.warning(
                "Download attempt %d/%d for %s failed (%s); retrying in %.0fs...",
                attempt, max_retries, url, e, backoff_seconds,
            )
            time.sleep(backoff_seconds)
            backoff_seconds *= 2