<p align="center">
    <img src="https://github.com/dhmay/CMVividly/blob/main/resources/cmvividly_logo.png"
        alt="CMVividly">
</p>

Exploring public T-cell receptors (TCRs) that respond to Cytomegalorvirus.

I'm a lead author of a [preprint](https://www.biorxiv.org/content/10.1101/2024.03.26.583354) describing "ECOclusters":
groups of T-cell receptors (TCRs) that represent humanity's collective T-cell responses to viruses, bacteria, etc. We
constructed ECOclusters by finding groups of T-cell receptors (TCRs) that tend to occur in the same people, across >30,000 people.

The “CMV ECOcluster” is our compilation of humanity’s collective public TCR response to Cytomegalovirus (CMV): 52,447 TCRs, each associated with the HLA allele that presents the CMV peptide it binds ([available here](https://www.biorxiv.org/content/10.1101/2024.03.26.583354v3.supplementary-material)). Here, I'm exploring the CMV ECOcluster to help [the AIRR community](https://www.antibodysociety.org/the-airr-community/)
and others make the best use of this new public resource ([more details here](about.qmd)). I'll add updates here, publish them to [cmvividly.com](https://cmvividly.com), and
publicize them on my [LinkedIn](https://www.linkedin.com/in/damonhmay). 

So, those 52,447 CMV TCRs: let's **see 'em, vividly**.

## Replicating my environment and running my code locally

I did package management with [uv](https://docs.astral.sh/uv/), so you should be able to:

1. Install Python (e.g., 3.12.x, such as 3.12.13).
2. Install **uv** (e.g., curl ... | sh or your preferred installer method).
3. Clone this repo.
4. `cd` to the repo root.
5. Create/sync the environment and install dependencies: `uv sync`

The code used by my Quarto posts is stored in `src/python`, so if you put that directory on your PYTHONPATH you should be able to
do things like I'm doing in that Python environment (e.g., via `uv run jupyter lab`).

## Running my code with GitHub Codespaces

This repository includes a [.devcontainer](.devcontainer) configuration, so you can open it with GitHub Codespaces
and work with the `cmvividly` package and all its dependencies:

1. In this github repo, choose **Code** -> **Codespaces** -> **Create codespace on main**.
2. Wait for the container to build and for `uv sync --all-groups` to finish.
3. Open a terminal or notebook and use the preinstalled Python and package environment.

The first startup may take a few minutes because it installs the project dependencies inside the container.
