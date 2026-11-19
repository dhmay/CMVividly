"""Giving a component one peptide, and the HLA that peptide implies."""
import pandas as pd
import pytest

from cmvividly.data.ecocluster_graph import HLA_CROSSABLE_COLUMN
from cmvividly.data.propagation import (
    ASSIGNED,
    CONFIDENCE_COLUMN,
    HIGH,
    LOW,
    HLA_AMBIGUOUS,
    HLA_FROM_COMPONENT,
    HLA_FROM_PMIDS,
    HLA_UNIQUE,
    PEPTIDE_TIE,
    PROPAGATED_ARTIFACT,
    add_peptide_confidence,
    hla_for_peptide,
    load_propagated_ecocluster,
    peptide_allele_support,
    propagate_component_annotations,
)

TRB = ("trb_vfamily", "trb_cdr3")


def alleles(rows=(("NLVPMVATV", "HLA-A*02:01", 12),)):
    return pd.DataFrame(rows, columns=["peptide", "allele", "n_pmids"])


def members(component_id="c1", hlas=("HLA-A*02:01",)):
    return pd.DataFrame({"component_id": [component_id] * len(hlas), "hla_std": list(hlas)})


def annotated(votes, component_id="c1", resolved="HLA-A*02:01"):
    """One row per voter; `votes` is a list of (cdr3, peptide)."""
    return pd.DataFrame({
        "component_id": [component_id] * len(votes),
        "trb_vfamily": ["TRBV1"] * len(votes),
        "trb_cdr3": [c for c, _ in votes],
        "top_antigen": [p for _, p in votes],
        "hla_resolved": [resolved] * len(votes),
    })


# --- voting ------------------------------------------------------------------------------------


def test_a_unique_plurality_wins():
    out = propagate_component_annotations(
        members(), annotated([("A", "NLVPMVATV"), ("B", "NLVPMVATV"), ("C", "NEGVKAAW")]),
        alleles([("NLVPMVATV", "HLA-A*02:01", 12), ("NEGVKAAW", "HLA-B*44:03", 3)]),
        trb_columns=TRB)
    row = out.iloc[0]
    assert row.peptide == "NLVPMVATV" and row.votes == 2 and row.runner_up_votes == 1
    assert row.status == ASSIGNED and row.hla == "HLA-A*02:01"


def test_one_vote_per_trb_not_per_annotation():
    """⚠️ A TRB appearing on several rows must not outvote the others."""
    rows = annotated([("A", "NLVPMVATV"), ("A", "NLVPMVATV"), ("A", "NLVPMVATV"),
                      ("B", "NEGVKAAW"), ("C", "NEGVKAAW")])
    out = propagate_component_annotations(
        members(), rows,
        alleles([("NLVPMVATV", "HLA-A*02:01", 12), ("NEGVKAAW", "HLA-B*44:03", 3)]),
        trb_columns=TRB)
    assert out.iloc[0].peptide == "NEGVKAAW", "A voted three times"
    assert out.iloc[0].voters == 3


def test_only_rows_with_a_resolved_hla_vote():
    """Rule 1: an annotation the HLA check could not corroborate does not get a say."""
    rows = annotated([("A", "NLVPMVATV"), ("B", "NEGVKAAW"), ("C", "NEGVKAAW")])
    rows.loc[rows.trb_cdr3.isin(["B", "C"]), "hla_resolved"] = None
    out = propagate_component_annotations(
        members(), rows,
        alleles([("NLVPMVATV", "HLA-A*02:01", 12), ("NEGVKAAW", "HLA-B*44:03", 3)]),
        trb_columns=TRB)
    assert out.iloc[0].peptide == "NLVPMVATV" and out.iloc[0].voters == 1


def test_a_component_with_no_voting_rows_is_absent():
    rows = annotated([("A", "NLVPMVATV")])
    rows["hla_resolved"] = None
    assert propagate_component_annotations(members(), rows, alleles(), trb_columns=TRB).empty


def test_a_tie_whose_candidates_all_fit_the_components_hla_is_left_unannotated():
    """A tie the component's own HLA call cannot break is not guessed at.

    Both peptides sit on A*02:01 and the component says A*02:01, so it has nothing to say.
    """
    out = propagate_component_annotations(
        members(), annotated([("A", "NLVPMVATV"), ("B", "VLEETSVML")]),
        alleles([("NLVPMVATV", "HLA-A*02:01", 12), ("VLEETSVML", "HLA-A*02:01", 3)]),
        trb_columns=TRB)
    row = out.iloc[0]
    assert row.peptide is None and row.status == PEPTIDE_TIE and row.hla is None


def test_a_tie_neither_candidate_fits_is_also_left_unannotated():
    out = propagate_component_annotations(
        members(hlas=("HLA-B*35:01",)), annotated([("A", "NLVPMVATV"), ("B", "NEGVKAAW")]),
        alleles([("NLVPMVATV", "HLA-A*02:01", 12), ("NEGVKAAW", "HLA-B*44:03", 3)]),
        trb_columns=TRB)
    assert out.iloc[0].status == PEPTIDE_TIE


def test_a_tie_is_broken_when_only_one_candidate_fits_the_components_own_hla():
    """The component says A*02:01, so of two tied peptides only NLVPMVATV can be the one."""
    out = propagate_component_annotations(
        members(hlas=("HLA-A*02:01", "HLA-A*02:01")),
        annotated([("A", "NLVPMVATV"), ("B", "NEGVKAAW")]),
        alleles([("NLVPMVATV", "HLA-A*02:01", 12), ("NEGVKAAW", "HLA-B*44:03", 3)]),
        trb_columns=TRB)
    assert out.iloc[0].peptide == "NLVPMVATV" and out.iloc[0].status == ASSIGNED


# --- the HLA a peptide implies -------------------------------------------------------------------


def test_one_recorded_allele_is_used_directly():
    assert hla_for_peptide("NLVPMVATV", frozenset(), alleles()) == ("HLA-A*02:01", HLA_UNIQUE)


def test_the_components_own_call_picks_between_two_alleles():
    """CRVLCCYVL is recorded on both A*02:01 and C*07:02; the component says C*07:02."""
    pdf = alleles([("CRVLCCYVL", "HLA-A*02:01", 1), ("CRVLCCYVL", "HLA-C*07:02", 1)])
    assert hla_for_peptide("CRVLCCYVL", frozenset({"HLA-C*07:02"}), pdf) == (
        "HLA-C*07:02", HLA_FROM_COMPONENT)


def test_pmids_decide_when_the_component_cannot():
    pdf = alleles([("CRVLCCYVL", "HLA-A*02:01", 1), ("CRVLCCYVL", "HLA-C*07:02", 9)])
    assert hla_for_peptide("CRVLCCYVL", frozenset({"HLA-B*07:02"}), pdf) == (
        "HLA-C*07:02", HLA_FROM_PMIDS)


def test_pmids_decide_between_the_two_the_component_allows():
    """⚠️ When the component spans a crossable pair it allows both; PMIDs break that."""
    pdf = alleles([("CRVLCCYVL", "HLA-B*07:02", 2), ("CRVLCCYVL", "HLA-C*07:02", 9)])
    assert hla_for_peptide(
        "CRVLCCYVL", frozenset({"HLA-B*07:02", "HLA-C*07:02"}), pdf)[0] == "HLA-C*07:02"


def test_everything_tied_leaves_the_hla_unannotated():
    pdf = alleles([("CRVLCCYVL", "HLA-A*02:01", 4), ("CRVLCCYVL", "HLA-C*07:02", 4)])
    assert hla_for_peptide("CRVLCCYVL", frozenset(), pdf) == (None, None)


def test_an_unresolvable_hla_is_reported_as_such_not_silently_dropped():
    pdf = alleles([("NLVPMVATV", "HLA-A*02:01", 4), ("NLVPMVATV", "HLA-C*07:02", 4)])
    out = propagate_component_annotations(
        members(hlas=("HLA-B*35:01",)), annotated([("A", "NLVPMVATV")]), pdf, trb_columns=TRB)
    row = out.iloc[0]
    assert row.peptide == "NLVPMVATV", "the peptide still won its vote"
    assert row.hla is None and row.status == HLA_AMBIGUOUS


def test_a_peptide_with_no_recorded_allele_is_not_invented():
    out = propagate_component_annotations(
        members(), annotated([("A", "MYSTERY")]), alleles(), trb_columns=TRB)
    assert out.iloc[0].hla is None and out.iloc[0].status == HLA_AMBIGUOUS


# --- support table ------------------------------------------------------------------------------


def test_pmids_are_unioned_across_rows_not_summed():
    """One paper reporting three TRBs is one source."""
    pdf_long = pd.DataFrame({
        "antigen": ["NLVPMVATV"] * 3,
        "mhc_a_std": ["HLA-A*02:01"] * 3,
        "pmids": ["1|2", "1|2", "3"],
    })
    out = peptide_allele_support(pdf_long)
    assert out.n_pmids.tolist() == [3], "distinct PMIDs 1, 2, 3"


def test_each_allele_of_a_joined_cell_gets_its_own_row():
    pdf_long = pd.DataFrame({"antigen": ["NLVPMVATV"],
                             "mhc_a_std": ["HLA-A*02|HLA-A*02:01"], "pmids": ["7"]})
    out = peptide_allele_support(pdf_long)
    assert sorted(out.allele) == ["HLA-A*02", "HLA-A*02:01"]


def test_rows_with_no_allele_are_dropped():
    pdf_long = pd.DataFrame({"antigen": ["X", "Y"], "mhc_a_std": ["", None], "pmids": ["1", "2"]})
    assert peptide_allele_support(pdf_long).empty


# --- the convenience loader --------------------------------------------------------------------


def propagated_table(rows=(("TRBV1", "CASSA", "A*02:01", "NLVPMVATV", "HLA-A*02:01"),)):
    """A stand-in for the artifact, with the two columns that collide with the ECOcluster."""
    return pd.DataFrame({
        "trb_vfamily": [v for v, *_ in rows],
        "trb_cdr3": [c for _, c, *_ in rows],
        HLA_CROSSABLE_COLUMN: [g for _, _, g, *_ in rows],
        "component_id": [f"{c}|{v}|{g}" for v, c, g, *_ in rows],
        "component_size": [2] * len(rows),
        "hla_std": ["HLA-A*02:01"] * len(rows),
        "peptide": [p for *_, p, _ in rows],
        "hla": [h for *_, h in rows],
        "hla_from": [HLA_FROM_COMPONENT] * len(rows),
        "status": [ASSIGNED] * len(rows),
        "votes": [1] * len(rows), "runner_up_votes": [0] * len(rows), "voters": [1] * len(rows),
    })


def ecocluster(rows=(("TRBV1", "CASSA", "A*02:01"), ("TRBV1", "CASSB", "B*07:02"))):
    """The ECOcluster's own shape: `hla` and `hla_std` are ITS columns, and must survive."""
    return pd.DataFrame({
        "vfamily_imgt": [v for v, *_ in rows], "cdr3": [c for _, c, *_ in rows],
        "hla": [h for *_, h in rows],
        "hla_std": [f"HLA-{h}" for *_, h in rows],
        "hla_class": ["ci"] * len(rows),
    })


def written(tmp_path, pdf=None):
    (tmp_path / PROPAGATED_ARTIFACT).write_text(
        (propagated_table() if pdf is None else pdf).to_csv(sep="\t", index=False))
    return tmp_path


def test_the_ecoclusters_own_hla_columns_survive_the_join(tmp_path):
    out = load_propagated_ecocluster(written(tmp_path), pdf_eco=ecocluster())
    assert list(out.hla) == ["A*02:01", "B*07:02"]
    assert list(out.hla_std) == ["HLA-A*02:01", "HLA-B*07:02"]
    assert "hla_propagated" in out.columns and "hla_from" not in out.columns


def test_an_unreached_row_is_kept_with_nulls(tmp_path):
    out = load_propagated_ecocluster(written(tmp_path), pdf_eco=ecocluster())
    assert len(out) == 2
    assert out.peptide.tolist()[0] == "NLVPMVATV"
    assert pd.isna(out.peptide.tolist()[1]) and pd.isna(out.component_id.tolist()[1])


def test_propagated_only_drops_the_unreached(tmp_path):
    out = load_propagated_ecocluster(written(tmp_path), pdf_eco=ecocluster(),
                                     propagated_only=True)
    assert len(out) == 1 and out.peptide.tolist() == ["NLVPMVATV"]


def test_class_ii_is_refused_rather_than_returning_nothing(tmp_path):
    with pytest.raises(ValueError, match="class I only"):
        load_propagated_ecocluster(written(tmp_path), pdf_eco=ecocluster(), arm="cii")


def test_a_duplicated_node_key_is_refused_rather_than_fanning_rows_out(tmp_path):
    doubled = pd.concat([propagated_table(), propagated_table()], ignore_index=True)
    with pytest.raises(AssertionError, match="not unique"):
        load_propagated_ecocluster(written(tmp_path, doubled), pdf_eco=ecocluster())


def test_a_colliding_column_is_refused_rather_than_suffixed(tmp_path):
    pdf = propagated_table().assign(hla_class="ci")  # the ECOcluster's own column name
    with pytest.raises(AssertionError, match="would overwrite"):
        load_propagated_ecocluster(written(tmp_path, pdf), pdf_eco=ecocluster())


# --- confidence in a propagated peptide ----------------------------------------------------------


def judged(rows):
    """`rows` is a list of (peptide, votes, component_size)."""
    pdf = pd.DataFrame(rows, columns=["peptide", "votes", "component_size"])
    return add_peptide_confidence(pdf)[CONFIDENCE_COLUMN].tolist()


def test_two_votes_are_enough_at_any_size():
    assert judged([("NLVPMVATV", 2, 208)]) == [HIGH]


def test_one_vote_in_a_large_cluster_is_low():
    assert judged([("TPRVTGGGAM", 1, 40)]) == [LOW]


def test_one_vote_is_enough_when_it_is_most_of_a_small_cluster():
    assert judged([("TPRVTGGGAM", 1, 2), ("TPRVTGGGAM", 1, 10)]) == [HIGH, HIGH]


def test_the_fraction_boundary_is_inclusive():
    # 1/10 is exactly CONFIDENCE_MIN_FRACTION, 1/11 is below it.
    assert judged([("A", 1, 10), ("A", 1, 11)]) == [HIGH, LOW]


def test_an_unannotated_cluster_is_NA_not_low():
    """'no peptide' and 'a peptide on thin evidence' must not be the same filter."""
    assert pd.isna(judged([(None, None, 7)])[0])


def test_the_thresholds_are_parameters_not_baked_in():
    pdf = pd.DataFrame([("A", 1, 40)], columns=["peptide", "votes", "component_size"])
    assert add_peptide_confidence(pdf, min_fraction=0.02)[CONFIDENCE_COLUMN].tolist() == [HIGH]
    assert add_peptide_confidence(pdf, min_votes=1)[CONFIDENCE_COLUMN].tolist() == [HIGH]


def test_a_missing_column_is_an_error_rather_than_a_silent_NA():
    with pytest.raises(ValueError, match="component_size"):
        add_peptide_confidence(pd.DataFrame({"peptide": ["A"], "votes": [2]}))
