# Methods for manipulating dataframes

import numpy as np
import pandas as pd

#: P group -> its member alleles, the full list from the ECOclusters paper's supplementary
#: material. Several of these never fire on the CMV ECOcluster -- `DRB3*01:01P` and `A*23:01P`
#: have no member allele anywhere in the 2024 release -- but the mapping is kept complete so it
#: is reusable on other datasets built from the same models.
#:
#: ⚠️ `DRB5*02:02P` is worth knowing about: `DRB5*02:02` appears 556 times in the raw 2024
#: release and the P group name appears **nowhere** in 2026, so applying the mapping relabels
#: those rows to a name the 2026 ECOcluster never uses. That is a faithful application of the
#: paper's grouping, not a bug, but it means a `DRB5*02:02P` row has no 2026 counterpart to
#: agree with. Only `DRB1*14:54` and `DRB4*01:03` are actually replaced across the two releases.
HLA_P_GROUPS = {
    "A*23:01P": ["A*23:01", "A*23:17"],
    "DRB1*14:01P": ["DRB1*14:01", "DRB1*14:54"],
    "DRB3*01:01P": ["DRB3*01:62", "DRB3*01:01"],
    "DRB4*01:01P": ["DRB4*01:01", "DRB4*01:03"],
    "DRB5*02:02P": ["DRB5*02:02", "DRB5*02:21"],
}
#: member allele -> the P group containing it.
ALLELE_TO_P_GROUP = {allele: p_group
                     for p_group, alleles in HLA_P_GROUPS.items()
                     for allele in alleles}
#: The columns `apply_hla_p_groups` rewrites. `hla_cocluster*` are cluster ids, not alleles.
HLA_P_GROUP_COLUMNS = ["hla", "hla_2024", "hla_2026"]


def apply_hla_p_groups(pdf: pd.DataFrame) -> pd.DataFrame:
    """Rename alleles in HLA_P_GROUPS to their P group, and recompute hla_agree_2024_2026.

    The 2024 ECOcluster models individual alleles; the 2026 one models some of them as a single
    P group. Collapsing 2024's members to the group is what makes the two releases comparable.

    Each part of a heterodimer (e.g. DQA1*05:01+DQB1*02:01) is renamed separately, and only exact
    allele matches are renamed, so a name that is already a P group is left alone. The
    `hla_cocluster*` columns are cluster ids and are not touched.

    Args:
        pdf: the combined ECOcluster, carrying HLA_P_GROUP_COLUMNS and hla_agree_2024_2026.

    Returns:
        A copy of pdf with those columns renamed and hla_agree_2024_2026 recomputed as
        hla_2024 == hla_2026.
    """
    pdf = pdf.copy()
    for col in HLA_P_GROUP_COLUMNS:
        # Rename the unique values, then map back to rows.
        values = pd.Series(pdf[col].dropna().unique())
        renamed = values.str.split("+").map(
            lambda parts: "+".join(ALLELE_TO_P_GROUP.get(part, part) for part in parts))
        pdf[col] = pdf[col].map(dict(zip(values, renamed)))
        parts = pdf[col].dropna().str.split("+").explode()
        assert not parts.isin(ALLELE_TO_P_GROUP).any(), f"{col} still has an allele in a P group"
    pdf["hla_agree_2024_2026"] = pdf["hla_2024"] == pdf["hla_2026"]
    return pdf


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
                                      pdf_cmv_ecocluster_2026: pd.DataFrame,
                                      apply_p_groups: bool = True) -> pd.DataFrame:
    """
    Combine the 2024 and 2026 CMV ECOclusters into one dataframe, deduplicating
    on TCR (cdr3 + vgene + jgene) and annotating where each TCR came from
    (2024 only, 2026 only, or shared). When HLA disagrees, use the 2026
    one.

    Args:
        pdf_cmv_ecocluster_2024: 2024 CMV ECOcluster dataframe.
        pdf_cmv_ecocluster_2026: 2026 CMV ECOcluster dataframe.
        apply_p_groups: collapse 2024's individual alleles onto the P groups the 2026 release
            models them as, via `apply_hla_p_groups`, and recompute hla_agree_2024_2026 from the
            collapsed names. On by default: without it the two releases disagree on HLA for
            TCRs that in fact agree, because only one of them names the P group.
            ⚠️ Pass **False** to study the difference between the releases itself -- a
            comparison of 2024 against 2026 wants the names each release actually used.
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
    if apply_p_groups:
        pdf_cmv_ecocluster_merged = apply_hla_p_groups(pdf_cmv_ecocluster_merged)
    first_cols = ["tcr", "cdr3", "vgene", "jgene", "hla", "hla_class", "hla_cocluster", "ecocluster_source"]
    return pdf_cmv_ecocluster_merged[first_cols + [col for col in pdf_cmv_ecocluster_merged.columns if col not in first_cols]]
