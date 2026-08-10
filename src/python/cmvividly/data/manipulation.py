# Methods for manipulating dataframes

import numpy as np
import pandas as pd

def extract_hlacoclusters_pdf(pdf_cmv_ecocluster: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a dataframe with one row per HLA-COcluster
    from a CMV ECOcluster dataframe, dropping TCR-level
    information and adding a "n_tcrs" column

    Args:
        pdf_cmv_ecocluster (pd.DataFrame): ECOcluster dataframe
           with one row per TCR

    Returns:
        pd.DataFrame: _description_
    """
    hlacocluster_level_cols = [
        'hla', 'hla_class',
        'hla_cocluster_npos_hlamatch',
        'hla_cocluster_nneg_hlamatch',
        'hla_cocluster_auroc_hlaaware',
        'hla_cocluster_auroc_hlaunaware']
    # make compatible with 2024 CMV ECOcluster
    hlacocluster_level_cols = [
        c for c in hlacocluster_level_cols
        if c in pdf_cmv_ecocluster.columns
    ]
    agg_dict = {"tcr": "count"}
    for col in hlacocluster_level_cols:
        agg_dict[col] = "first"
    pdf_hla_coclusters = pdf_cmv_ecocluster.groupby("hla_cocluster").agg(agg_dict).reset_index()
    pdf_hla_coclusters = pdf_hla_coclusters.rename(columns={
        "tcr": "n_tcrs"})
    # make n_tcrs the second column, after hla_cocluster, and before the other columns
    pdf_hla_coclusters = pdf_hla_coclusters[["hla_cocluster", "n_tcrs"] + hlacocluster_level_cols]
    return pdf_hla_coclusters


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

    first_cols = ["tcr", "cdr3", "vgene", "jgene", "hla", "hla_class"]
    if "tcr_pgen" in df.columns:
        df["log10_tcr_pgen_eps"] = (df["tcr_pgen"] + 1e-50).apply(lambda x: np.log10(x))
        first_cols += ["tcr_pgen", "log10_tcr_pgen_eps"]
    # rearrange the columns: tcr, cdr3, vgene, jgene, hla, hla_class, tcr_pgen, log10_tcr_pgen_eps and the rest
    cols = first_cols + [c for c in df.columns if c not in first_cols]
    return df[cols]