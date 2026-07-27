<p align="center">
    <img src="https://github.com/dhmay/CMVividly/blob/main/resources/cmvividly_logo.png"
        alt="CMVividly">
</p>

Exploring the public T-cell response to Cytomegalovirus, live, at [cmvividly.com](https://cmvividly.com).

I'm Damon May, a machine learning researcher and computational immunologist. I was most recently at Adaptive
Biotechnologies, in both leadership (Associate Director) and individual contributor (Principal Scientist) roles,
building ML models on immune receptor repertoires.

Here, I'm sharing insights about a new resource my colleagues and I made available in July 2026: the **CMV
ECOcluster**. We mined more than 30,000 T-cell receptor (TCR) repertoires to find **ECOclusters**: groups of TCRs
that tend to occur in the same people. Each ECOcluster putatively represents humanity's collective response to a
virus, bacterium, or other prevalent exposure
([explainer](https://globalforum.diaglobal.org/issue/january-2025/#ecoclusters) in DIA Global Forum). We described
ECOclusters in [this preprint](https://www.biorxiv.org/content/10.1101/2024.03.26.583354), with a major update in
July 2026.

The **CMV ECOcluster**
([direct download](https://www.biorxiv.org/content/10.1101/2024.03.26.583354v3.supplementary-material)) is a
sensitive, specific, quantitative biomarker of response to Cytomegalovirus (CMV). Each of its 52,447 TCRs is
associated with the Human Leukocyte Antigen (HLA) allele that presents the CMV peptide it binds. I'll post new
analyses at [cmvividly.com](https://cmvividly.com) and announce them on my
[LinkedIn](https://www.linkedin.com/in/damonhmay).

So, those 52,447 TCRs: let's **see 'em, vividly**.

## Run notebooks / code in a Codespace

The fastest way to explore this code (no local setup required) is
[GitHub Codespaces](https://github.com/features/codespaces):

1. On this repo page, click the green **Code** button, open the **Codespaces** tab, and click
   **Create codespace on main**.
2. Wait for the codespace to build. The first launch runs `uv sync` automatically, installing all dependencies and
   the `cmvividly` package itself — this takes a minute or two.
3. In the file explorer, open the example notebook, `notebooks/cmvividly_example.ipynb`.
4. Two options for starting a notebook:
    * Click **Select Kernel** in the top right, choose **Python Environments**, and pick the `.venv` interpreter
      that `uv sync` created (not the default system Python).
    * Open a terminal in the codespace and run `uv run jupyter lab --no-browser`, then click **Open in Browser**
      in the notification that pops up.

## Reproduce this environment on your machine

1. Clone this repo and install Python and [uv](https://docs.astral.sh/uv/).
2. From the repository root, two options for setting up the environment:
    * with **uv**: `uv sync`, then start Jupyter with `uv run jupyter lab`
    * with **venv** and **pip**:
      ```bash
      python -m venv .venv
      source .venv/bin/activate
      pip install -r requirements.txt
      ```

The code used by the Quarto posts lives in `src/python`; put that directory on your `PYTHONPATH` to use it the same
way the posts do.

To render the Quarto site itself, run `quarto render` (or `uv run quarto render`) from the repo root.

More details are on the [About page](https://cmvividly.com/about.html).
