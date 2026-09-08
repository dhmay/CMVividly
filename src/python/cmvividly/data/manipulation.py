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


def combine_2024_2026_cmv_ecoclusters(pdf_cmv_ecocluster_2024: pd.DataFrame,
                                      pdf_cmv_ecocluster_2026: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the 2024 and 2026 CMV ECOclusters into one dataframe, deduplicating
    on TCR (cdr3 + vgene + jgene) and annotating where each TCR came from
    (2024 only, 2026 only, or shared). When HLA disagrees, use the 2026
    one.

    Args:
        pdf_cmv_ecocluster_2024: 2024 CMV ECOcluster dataframe.
        pdf_cmv_ecocluster_2026: 2026 CMV ECOcluster dataframe.
    Returns:
        Combined dataframe with one row per TCR, annotated with
        which version(s) it came from and whether HLA class agrees.
    """
    # the 2024 ECOcluster has some duplicate TCRs, so dedup on TCR first.
    # I'm essentially grabbing one row at random. This could probably be improved upon,
    # but I'd have to figure out some way of deciding which row to keep. 
    pdf_cmv_ecocluster_2024_dedup = pdf_cmv_ecocluster_2024.sort_values("hla_cocluster").drop_duplicates(subset=["tcr"])
 
    # outer join on tcr, to build a dataframe that contains all TCRs in both ECOcluster versions
    pdf_2024_merge = pdf_cmv_ecocluster_2024_dedup.copy()
    pdf_2024_merge["source_2024"] = True
    pdf_2026_merge = pdf_cmv_ecocluster_2026.copy()
    pdf_2026_merge["source_2026"] = True
    
    pdf_cmv_ecocluster_merged = pdf_2024_merge.merge(
        pdf_2026_merge,
        on=["tcr", "cdr3", "vgene", "jgene"],
        how="outer",
        suffixes=("_2024", "_2026"),
        indicator=True,
    )

    # annotate each row with whether HLA class agrees and which version(s) the TCR came from
    pdf_cmv_ecocluster_merged["hla_agree_2024_2026"] = (
        pdf_cmv_ecocluster_merged["hla_2024"] == pdf_cmv_ecocluster_merged["hla_2026"]
    )
    pdf_cmv_ecocluster_merged["hla_class_agree_2024_2026"] = (
        pdf_cmv_ecocluster_merged["hla_class_2024"] == pdf_cmv_ecocluster_merged["hla_class_2026"]
    )
    pdf_cmv_ecocluster_merged["ecocluster_source"] = pdf_cmv_ecocluster_merged["_merge"].map(
        {"left_only": "2024_only", "right_only": "2026_only", "both": "2024_and_2026"})

    # assign the "hla" column" as hla_2024 if only hla_2024 is present, otherwise hla_2026.
    # I.e., 2026 wins ties. That's pretty arbitrary.
    pdf_cmv_ecocluster_merged["hla"] = pdf_cmv_ecocluster_merged.apply(
        lambda row: row["hla_2024"] if row["ecocluster_source"] == "2024_only"
        else row["hla_2026"],
        axis=1
    )
    pdf_cmv_ecocluster_merged["hla_class"] = pdf_cmv_ecocluster_merged.hla.apply(
        lambda x: "cii" if x.startswith("D") else "ci")

    # assign hla_cocluster analogously to hla
    pdf_cmv_ecocluster_merged["hla_cocluster"] = pdf_cmv_ecocluster_merged.apply(
        lambda row: row["hla_cocluster_2024"] if row["ecocluster_source"] == "2024_only"
        else row["hla_cocluster_2026"],
        axis=1
    )
    
    pdf_cmv_ecocluster_merged["source_2026"] = pdf_cmv_ecocluster_merged.source_2026.fillna(False).astype(bool)
    pdf_cmv_ecocluster_merged["source_2024"] = pdf_cmv_ecocluster_merged.source_2024.fillna(False).astype(bool)

    drop_cols = ["_merge", "source_2024", "source_2026"]
    pdf_cmv_ecocluster_merged = pdf_cmv_ecocluster_merged.drop(columns=drop_cols)
    first_cols = ["tcr", "cdr3", "vgene", "jgene", "hla", "hla_class", "hla_cocluster", "ecocluster_source"]
    return pdf_cmv_ecocluster_merged[first_cols + [col for col in pdf_cmv_ecocluster_merged.columns if col not in first_cols]]
