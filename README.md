# Data Analysis Series

A starter repository for an episodic, reproducible Python data analysis project published with Quarto.

## What is included

- A Quarto website with a blog-style post listing
- One starter post in `posts/001-example/index.qmd`
- One exploratory Jupyter notebook in `notebooks/001-example-exploration.ipynb`
- Shared Python code in `src/`
- A starter data generation script in `download_data.py`
- GitHub Pages publishing workflow in `.github/workflows/publish.yml`
- Python environment files for `uv`, `pip`, or Conda
- A devcontainer for GitHub Codespaces or VS Code Dev Containers

## Repository layout

```text
.
├── posts/                  # Published Quarto posts
│   └── 001-example/
│       └── index.qmd
├── notebooks/              # Exploratory notebooks
├── src/                    # Reusable project code
├── data/
│   ├── raw/                # Raw data, usually not committed
│   └── processed/          # Processed data, usually not committed
├── figures/                # Optional exported figures
├── _quarto.yml             # Quarto website config
├── download_data.py        # Replace with real data download logic
├── pyproject.toml          # Python dependencies for uv
├── requirements.txt        # Python dependencies for pip/GitHub Actions
└── environment.yml         # Python dependencies for Conda
```

## Local setup with uv

Install Quarto first: <https://quarto.org/docs/get-started/>

Then run:

```bash
uv sync
uv run python download_data.py
uv run quarto preview
```

## Local setup with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
quarto preview
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Local setup with Conda

```bash
conda env create -f environment.yml
conda activate data-analysis-series
python download_data.py
quarto preview
```

## Render the site

```bash
quarto render
```

The rendered website will appear in `_site/`.

## Publish to GitHub Pages

1. Create a new public GitHub repository.
2. Copy these files into it.
3. Replace placeholder values in `_quarto.yml`:
   - `YOUR-GITHUB-USERNAME`
   - `YOUR-REPO-NAME`
   - `YOUR-LINKEDIN-SLUG`
4. Push to the `main` branch.
5. In GitHub, go to **Settings → Pages**.
6. Under **Build and deployment**, set the source to **GitHub Actions**.
7. Push another commit or manually run the `Publish Quarto site` workflow.

## Suggested workflow for each episode

1. Explore in `notebooks/`.
2. Move reusable loading, cleaning, or plotting code into `src/`.
3. Write the polished narrative in a new folder under `posts/`.
4. Render locally with `quarto preview`.
5. Push to GitHub.
6. Share the published post on LinkedIn.

## Data policy

The template ignores files in `data/raw/` and `data/processed/` by default to avoid accidentally committing large or restricted data.

For public reproducibility, prefer one of these approaches:

- commit a small sample dataset,
- write a download script that fetches the public data,
- document the exact source URL and access date,
- or use GitHub Releases / external storage for large files.

## Next edits to make

- Pick your real public dataset.
- Replace `download_data.py` with a real downloader.
- Replace the starter post with your first analysis.
- Update the site title and About page.
- Add your LinkedIn URL in `_quarto.yml`.
