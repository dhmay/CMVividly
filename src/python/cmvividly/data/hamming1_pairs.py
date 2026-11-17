"""CMV-ECOcluster-specific handling of Hamming-1 CDR3 pairs.

⚠️ **The generic machinery is no longer here.** Finding Hamming-1 pairs and building graphs over
them lives in `portmanteau.sequence`, which this package now depends on; the copies that used to
sit in this module and in `hamming1_graph.py` were removed in favour of it. The equivalents:

===================================================  =======================================================
was                                                  now
===================================================  =======================================================
``find_cdr3_hamming1_pairs(pdf, seq_col)``           ``portmanteau.sequence.hamming_pairs.find_hamming1_pairs(pdf, seq_col)``
``find_hamming1_pairs_same_vj(pdf)``                 ``find_hamming1_pairs(pdf, "tcr", group_cols=["vgene", "jgene"])``
``hamming1_graph.build_seq_ham1_graph``              ``portmanteau.sequence.hamming_n_graph.build_seq_hamming_graph``
``hamming1_graph.build_seq_ham1_graph_and_extract_ccs``  ``...hamming_n_graph.build_seq_hamming_graph_and_extract_ccs``
``hamming1_graph.extract_connected_components``      ``...hamming_n_graph.extract_connected_components`` (same signature)
``hamming1_graph.plot_seq_ham1_graph``               ``...hamming_n_graph.plot_seq_graph``
``hamming1_graph.prepend_other_column_to_seq``       ``...hamming_n_graph.prepend_other_column_to_seq`` (same signature)
===================================================  =======================================================

Two behaviour changes the replacements bring, both improvements, both checked on the full
ECOcluster (66,024 rows, 50,348 same-V/J pairs — identical pair sets and identical
``differing_position``/``seq_length`` on every one):

- ⚠️ **The old ``find_hamming1_pairs_same_vj`` emitted all-NaN ``cdr3_i``/``cdr3_j`` columns**,
  an artifact of calling the CDR3 helper with ``seq_col="tcr"``. portmanteau does not.
- ⚠️ **With grouping, a portmanteau graph node is a tuple** ``(sequence, *group values)``, not a
  bare string, so ``G.subgraph(...)`` and node lookups need tuples. Its
  ``extract_connected_components`` unpacks them back into columns.

What stays here is what is about *this dataset* rather than about sequences in general.
"""

from __future__ import annotations

import pandas as pd


def populate_hamming1_pairs_othercols(pdf_all_ham1_pairs: pd.DataFrame,
                                      pdf_original: pd.DataFrame) -> pd.DataFrame:
    """Populate additional columns in a dataframe of Hamming-1 pairs.

    ECOcluster-specific: it names this dataset's columns and derives the ``same_*`` flags the
    posts compare on, so it is not a candidate for portmanteau.

    Args:
        pdf_all_ham1_pairs: DataFrame containing Hamming-1 pairs with columns
            ``row_i`` and ``row_j`` that are index labels of the original dataframe.
        pdf_original: Original dataframe from which the pairs were derived.

    Returns:
        DataFrame with additional columns populated from the original dataframe.
    """
    pdf_original_withrow = pdf_original.copy()
    pdf_original_withrow["row"] = range(len(pdf_original_withrow))
    keep = ["row", "tcr", "cdr3", "vgene", "jgene", "hla_cocluster", "hla"]
    pdf_hamming1_pairs_allcols = (
        pdf_all_ham1_pairs
        .drop(columns=["cdr3_i", "cdr3_j"])
        .merge(pdf_original_withrow[keep], left_on="row_i", right_on="row")
        .rename(columns={"tcr": "tcr_i",
                         "vgene": "vgene_i",
                         "jgene": "jgene_i",
                         "cdr3": "cdr3_i",
                         "hla_cocluster": "hla_cocluster_i",
                         "hla": "hla_i",
                         })
        .merge(pdf_original_withrow[keep], left_on="row_j", right_on="row")
        .rename(columns={"tcr": "tcr_j",
                         "vgene": "vgene_j",
                         "jgene": "jgene_j",
                         "cdr3": "cdr3_j",
                         "hla_cocluster": "hla_cocluster_j",
                         "hla": "hla_j",
                         })
    )
    pdf_hamming1_pairs_allcols["same_vgene"] = (
        pdf_hamming1_pairs_allcols["vgene_i"] == pdf_hamming1_pairs_allcols["vgene_j"])
    pdf_hamming1_pairs_allcols["same_jgene"] = (
        pdf_hamming1_pairs_allcols["jgene_i"] == pdf_hamming1_pairs_allcols["jgene_j"])
    pdf_hamming1_pairs_allcols["same_hlacocluster"] = (
        pdf_hamming1_pairs_allcols["hla_cocluster_i"]
        == pdf_hamming1_pairs_allcols["hla_cocluster_j"])
    pdf_hamming1_pairs_allcols["same_hla"] = (
        pdf_hamming1_pairs_allcols["hla_i"] == pdf_hamming1_pairs_allcols["hla_j"])
    return pdf_hamming1_pairs_allcols
