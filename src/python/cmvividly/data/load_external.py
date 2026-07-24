from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

BIOARXIV_MEDIA2_ZIP_URL = (
    "https://www.biorxiv.org/content/biorxiv/early/2026/07/22/"
    "2024.03.26.583354/DC2/embed/media-2.zip?download=true"
)

def download_biorxiv_media2_zip(
    output_dir: str | Path,
    filename: str = "media-2.zip",
    overwrite: bool = False,
    timeout_seconds: int = 60,
) -> Path:
    """
    Download the BioRxiv zipped TSV archive and return the local zip path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / filename
    if zip_path.exists() and not overwrite:
        return zip_path

    req = Request(
        BIOARXIV_MEDIA2_ZIP_URL,
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
        raise ValueError(f"Downloaded file is not a valid zip archive: {zip_path}")

    return zip_path