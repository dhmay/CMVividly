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