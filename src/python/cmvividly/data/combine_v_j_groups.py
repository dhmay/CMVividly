# This module contains functions to combine indistinguishable V and J genes in a dataframe of TCRs.
# These indistinguishable genes were unearthed in post #4.

import pandas as pd

# Mapping from a group name for indistinguishable V genes to the members of the group
VGENE_COMBINATION_MEMBERS_MAP = {
    "TCRBV20-X": {'TCRBV20-01', 'TCRBV20-X'},
    "TCRBV07-X": {'TCRBV07-06', 'TCRBV07-07', 'TCRBV07-X'},
    "TCRBV06-02/06-03": {'TCRBV06-02', 'TCRBV06-02/06-03'},
    "TCRBV07-02/07-03": {'TCRBV07-02', 'TCRBV07-03'},
    "TCRBV11-X": {'TCRBV11-01', 'TCRBV11-02', 'TCRBV11-X'},
    "TCRBV04-X": {'TCRBV04-01', 'TCRBV04-02', 'TCRBV04-03'},
    "TCRBV06-X": {'TCRBV06-01', 'TCRBV06-05', 'TCRBV06-06', 'TCRBV06-X'}
}

# Mapping from a group name for indistinguishable J genes to the members of the group
JGENE_COMBINATION_MEMBERS_MAP = {
    "J02-03/05": {'J02-03', 'J02-05'}
}


pdf_vgene_combination_members = pd.DataFrame(
    VGENE_COMBINATION_MEMBERS_MAP.items(),
    columns=["combined_vgene", "vgene"]).explode("vgene").reset_index(drop=True)

def combine_indistinguishable_vgenes(pdf_tcrs, drop_original_col=True):
    """
    Combine indistinguishable V genes in a dataframe of TCRs using the VGENE_COMBINATION_MEMBERS_MAP.
    If a V gene is not in the combination map, it will remain unchanged.
    Parameters:
        pdf_tcrs (pd.DataFrame): DataFrame containing TCRs with a 'vgene' column.
        drop_original_col (bool): Whether to drop the 'vgene_original' column after combining.
    """
    pdf_tcrs_vcombined = pdf_tcrs.merge(pdf_vgene_combination_members, how="left", on="vgene")
    # If a vgene is not in the combination map, then combined_vgene will be NaN. Fill those with the original vgene.
    pdf_tcrs_vcombined["combined_vgene"] = pdf_tcrs_vcombined["combined_vgene"].fillna(pdf_tcrs_vcombined["vgene"])
    pdf_tcrs_vcombined = pdf_tcrs_vcombined.rename(columns={"vgene": "vgene_original", "combined_vgene": "vgene"})
    if drop_original_col:
        pdf_tcrs_vcombined = pdf_tcrs_vcombined.drop(columns=["vgene_original"])
    return pdf_tcrs_vcombined

def combine_indistinguishable_jgenes(pdf_tcrs, drop_original_col=True):
    """
    Combine indistinguishable J genes in a dataframe of TCRs using the JGENE_COMBINATION_MEMBERS_MAP.
    If a J gene is not in the combination map, it will remain unchanged.
    Parameters:
        pdf_tcrs (pd.DataFrame): DataFrame containing TCRs with a 'jgene' column.
        drop_original_col (bool): Whether to drop the 'jgene_original' column after combining.
    """
    pdf_tcrs_jcombined = pdf_tcrs.merge(
        pd.DataFrame(JGENE_COMBINATION_MEMBERS_MAP.items(), columns=["combined_jgene", "jgene"]).explode("jgene"),
        how="left", on="jgene")
    # If a jgene is not in the combination map, then combined_jgene will be NaN. Fill those with the original jgene.
    pdf_tcrs_jcombined["combined_jgene"] = pdf_tcrs_jcombined["combined_jgene"].fillna(pdf_tcrs_jcombined["jgene"])
    pdf_tcrs_jcombined = pdf_tcrs_jcombined.rename(columns={"jgene": "jgene_original", "combined_jgene": "jgene"})
    if drop_original_col:
        pdf_tcrs_jcombined = pdf_tcrs_jcombined.drop(columns=["jgene_original"])
    return pdf_tcrs_jcombined

def rebuild_tcr_from_cdr3_v_j(pdf_tcrs, retain_original_tcr=False):
    """
    Rebuild the 'tcr' column in a dataframe of TCRs from the 'cdr3', 'vgene', and 'jgene' columns.
    Parameters:
        retain_original_tcr (bool): Whether to retain the original 'tcr' column in "tcr_original"
    """
    if retain_original_tcr:
        pdf_tcrs["tcr_original"] = pdf_tcrs.tcr
    pdf_tcrs["tcr"] = pdf_tcrs.apply(lambda x: f"{x.cdr3}+{x.vgene}+{x.jgene}", axis=1)
    return pdf_tcrs


def combine_indistinguishable_genes(pdf_tcrs, drop_original_gene_cols=True,
                                   retain_original_tcr=False):
    """
    Combine indistinguishable V and J genes in a dataframe of TCRs.
    pdf_tcrs must have columns 'cdr3', 'vgene', and 'jgene'. 
    Replace indistinguishable vgene and jgene names with the names representing groups of
    indistinguishable genes. Rebuild the 'tcr' column from the new vgene and jgene names.

    Parameters:
    pdf_tcrs (pd.DataFrame): DataFrame containing TCRs with columns 'cdr3', 'vgene', and 'jgene'.
    drop_original_gene_cols (bool): Whether to drop the 'vgene_original' and 'jgene_original' columns after combining.
    retain_original_tcr (bool): Whether to retain the original 'tcr' column in "tcr_original"

    Returns:
    pd.DataFrame: DataFrame with combined V and J genes, and a new 'tcr' column.
    """
    pdf_tcrs_combined_v = combine_indistinguishable_vgenes(pdf_tcrs, drop_original_col=drop_original_gene_cols)
    pdf_tcrs_combined_vj = combine_indistinguishable_jgenes(pdf_tcrs_combined_v, drop_original_col=drop_original_gene_cols)
    pdf_result = rebuild_tcr_from_cdr3_v_j(pdf_tcrs_combined_vj, retain_original_tcr=retain_original_tcr)
    return pdf_result

