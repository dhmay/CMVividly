"""Joining ECOcluster TRBs to peptide annotations, with MHC class as a hard filter."""
import pandas as pd
import pytest

from cmvividly.data.ecocluster_annotation import (
    AGREE,
    AGREED,
    AGREEMENT_ORDER,
    ANNOTATION_HLA_COLUMN,
    CONFLICT,
    CONFUSABLE_ECO_ALLELES,
    FROM_ANNOTATION,
    FROM_ECOCLUSTER,
    RESOLVED_COLUMN,
    RESOLVED_FROM_COLUMN,
    TRUSTED_COLUMN,
    UNTRUSTED_ECO_ALLELES,
    NO_ALLELE,
    NOT_COMPARABLE,
    VERDICT_COLUMN,
    VERDICT_FIELDS_COLUMN,
    add_annotation_trb_columns,
    annot_hla_agreement,
    annotate_ecocluster,
    join_annotations,
    resolve_hla,
)

TRB_KEY = ("trb_vfamily", "trb_cdr3")


@pytest.fixture
def eco():
    """Three ECOcluster rows: two class II, one class I, one CDR3 shared across classes."""
    return pd.DataFrame({
        "tcr": ["t1", "t2", "t3"],
        "cdr3": ["CASSLGQAYEQYF", "CASSPARNTEAFF", "CASSLGQAYEQYF"],
        "vfamily_imgt": ["TRBV20", "TRBV4", "TRBV20"],
        "hla_class": ["cii", "ci", "ci"],
        "hla_std": ["HLA-DRB1*07:01", "HLA-A*02:01", "HLA-B*07:02"],
    })


@pytest.fixture
def annotations():
    return pd.DataFrame({
        "trb_vfamily": ["TRBV20", "TRBV4", "TRBV9"],
        "trb_cdr3": ["CASSLGQAYEQYF", "CASSPARNTEAFF", "CASSNOMATCHF"],
        "hla_class": ["ci", "ci", "ci"],
        "antigen.epitope": ["NLVPMVATV", "RPHERNGFTVL", "TPRVTGGGAM"],
        "top_antigen_mhc_a_std": ["HLA-B*07:02", "HLA-A*02:01", "HLA-A*01:01"],
    })


def test_adds_the_annotation_column_names_without_losing_the_originals(eco):
    out = add_annotation_trb_columns(eco)
    assert out.trb_cdr3.equals(eco.cdr3) and out.trb_vfamily.equals(eco.vfamily_imgt)
    assert "cdr3" in out.columns and "vfamily_imgt" in out.columns


def test_rejects_a_frame_that_is_not_the_ecocluster():
    with pytest.raises(ValueError, match="missing"):
        add_annotation_trb_columns(pd.DataFrame({"something_else": [1]}))


def test_class_filter_drops_the_cross_class_match(eco, annotations):
    """t1 is class II and its CDR3 matches a class I annotation: a coincidence, not a hit."""
    joined = join_annotations(eco, annotations, trb_key=TRB_KEY, verbose=False)
    assert set(joined.tcr) == {"t2", "t3"}
    assert (joined.hla_class == "ci").all()


def test_without_the_filter_the_cross_class_match_comes_back(eco, annotations):
    joined = join_annotations(eco, annotations, trb_key=TRB_KEY,
                              require_same_class=False, verbose=False)
    assert set(joined.tcr) == {"t1", "t2", "t3"}


def test_one_cdr3_in_both_classes_is_not_an_ambiguity(eco, annotations):
    """t1 and t3 share a CDR3 and differ only in class; the row-wise filter keeps the right one."""
    joined = join_annotations(eco, annotations, trb_key=TRB_KEY, verbose=False)
    shared = joined[joined.trb_cdr3 == "CASSLGQAYEQYF"]
    assert len(shared) == 1 and shared.iloc[0].tcr == "t3"


def test_unclassified_annotations_are_dropped_not_matched(eco, annotations):
    """McPAS rows with no MHC and IEDB's MR1/CD1 rows cannot belong to either class."""
    unknown = annotations.assign(hla_class=[None, None, None])
    joined = join_annotations(eco, unknown, trb_key=TRB_KEY, verbose=False)
    assert len(joined) == 0


def test_unclassified_ecocluster_rows_are_dropped_too(eco, annotations):
    joined = join_annotations(eco.assign(hla_class=None), annotations,
                              trb_key=TRB_KEY, verbose=False)
    assert len(joined) == 0


def test_requires_the_class_column_when_filtering(eco, annotations):
    with pytest.raises(ValueError, match="hla_class"):
        join_annotations(eco, annotations.drop(columns="hla_class"),
                         trb_key=TRB_KEY, verbose=False)


def test_annotate_keeps_every_ecocluster_row_and_fills_what_matched(eco, annotations):
    out = annotate_ecocluster(eco, annotations, trb_key=TRB_KEY, verbose=False)
    assert len(out) == len(eco)
    assert out.tcr.tolist() == eco.tcr.tolist(), "row order is preserved"
    # t1 is class II against a class I annotation, so it matches nothing; t3 shares t1's CDR3
    # but is class I, so it does.
    matched = out.set_index("tcr")["antigen.epitope"]
    assert pd.isna(matched["t1"])
    assert matched["t2"] == "RPHERNGFTVL" and matched["t3"] == "NLVPMVATV"
    assert "trb_cdr3" not in out.columns, "the join scaffolding must not collide with a cell frame"


def test_annotate_without_the_class_filter_matches_the_class_ii_row_too(eco, annotations):
    out = annotate_ecocluster(eco, annotations, trb_key=TRB_KEY, require_same_class=False,
                              verbose=False)
    assert len(out) == len(eco)
    assert out.set_index("tcr").loc["t1", "antigen.epitope"] == "NLVPMVATV"
    assert "hla_class_annotation" in out.columns, "the annotation's own class is kept, suffixed"


def test_annotate_refuses_an_annotation_table_that_would_fan_rows_out(eco):
    """The long table has several rows per TRB, so it is not what this step takes."""
    long = pd.DataFrame({
        "trb_vfamily": ["TRBV4", "TRBV4"],
        "trb_cdr3": ["CASSPARNTEAFF", "CASSPARNTEAFF"],
        "hla_class": ["ci", "ci"],
        "antigen.epitope": ["NLVPMVATV", "RPHERNGFTVL"],
        "top_antigen_mhc_a_std": ["HLA-A*02:01", "HLA-A*02:01"],
    })
    with pytest.raises(ValueError, match="more than one annotation row"):
        annotate_ecocluster(eco, long, trb_key=TRB_KEY, verbose=False)


def test_annotate_keeps_integer_counts_as_integers(eco):
    annotations = pd.DataFrame({
        "trb_vfamily": ["TRBV4"], "trb_cdr3": ["CASSPARNTEAFF"], "hla_class": ["ci"],
        "n_antigens": [3], "top_antigen_mhc_a_std": ["HLA-A*02:01"],
    })
    out = annotate_ecocluster(eco, annotations, trb_key=TRB_KEY, verbose=False)
    assert out.n_antigens.dtype == "Int64", "an unmatched row must not float the counts"
    assert out.set_index("tcr").loc["t2", "n_antigens"] == 3


# --- the allele dimension ------------------------------------------------------------------

P_GROUPS = {"HLA-A*23:01P": ["HLA-A*23:01", "HLA-A*23:17"]}


def verdict(eco_hla, annotation_hla):
    """One row's (verdict, resolution), with a fixed P group table so no ECOcluster is loaded."""
    pdf = pd.DataFrame({"hla_std": [eco_hla], "top_antigen_mhc_a_std": [annotation_hla]})
    row = annot_hla_agreement(pdf, p_groups=P_GROUPS).iloc[0]
    return row[VERDICT_COLUMN], row[VERDICT_FIELDS_COLUMN]


def test_the_same_allele_on_both_sides_agrees_at_full_resolution():
    assert verdict("HLA-A*02:01", "HLA-A*02:01") == (AGREE, 2)


def test_one_of_several_annotation_alleles_is_enough_to_agree():
    assert verdict("HLA-B*07:02", "HLA-A*02:01|HLA-B*07:02") == (AGREE, 2)


def test_a_one_field_annotation_allele_agrees_but_only_at_one_field():
    """HLA-A*02 cannot be told from the ECOcluster's A*02:01 or its A*02:06; the caller decides."""
    assert verdict("HLA-A*02:01", "HLA-A*02") == (AGREE, 1)


def test_the_deepest_agreement_is_the_one_reported():
    """A mixed-resolution annotation agrees twice over; the full-resolution match is the evidence."""
    assert verdict("HLA-A*02:01", "HLA-A*02|HLA-A*02:01") == (AGREE, 2)


def test_two_fully_resolved_different_alleles_conflict():
    assert verdict("HLA-B*07:02", "HLA-A*02:01") == (CONFLICT, 2)


def test_two_alleles_of_the_same_group_conflict_at_full_resolution():
    """A*02:01 and A*02:06 are different proteins, not two resolutions of one."""
    assert verdict("HLA-A*02:01", "HLA-A*02:06") == (CONFLICT, 2)


def test_a_conflict_reports_the_deepest_comparison_that_was_made():
    """One shallow annotation allele must not make the whole comparison look shallow."""
    assert verdict("HLA-A*02:01", "HLA-B*07|HLA-C*08:02") == (CONFLICT, 2)


def test_a_one_field_annotation_allele_of_another_gene_still_conflicts():
    assert verdict("HLA-A*02:01", "HLA-B*07") == (CONFLICT, 1)


def test_a_p_group_is_expanded_to_its_members_before_comparing():
    """A P group names a set, so it can never equal an allele name."""
    assert verdict("HLA-A*23:01P", "HLA-A*23:17") == (AGREE, 2)
    assert verdict("HLA-A*23:01P", "HLA-A*24:02") == (CONFLICT, 2)


def test_a_missing_allele_on_either_side_is_not_a_conflict():
    assert verdict("HLA-A*02:01", "") == (NO_ALLELE, pd.NA)
    assert verdict(None, "HLA-A*02:01") == (NO_ALLELE, pd.NA)


def test_a_name_with_no_fields_at_all_is_not_scored_as_a_conflict():
    """IEDB records B2M and bare class names in the MHC column; neither has a resolution."""
    assert verdict("HLA-A*02:01", "B2M") == (NO_ALLELE, pd.NA)


def test_a_class_ii_heterodimer_is_not_comparable():
    assert verdict("HLA-DQA1*05:01+HLA-DQB1*02:01", "HLA-DQA1*05:01") == (NOT_COMPARABLE, pd.NA)


def test_a_class_ii_dr_allele_is_not_comparable_either():
    """⚠️ The regression that matters: DRB1*07:01 carries no '+', so a separator-based guard
    misses it and it would be scored against the invariant DRA*01:01 as a conflict."""
    assert verdict("HLA-DRB1*07:01", "HLA-DRA*01:01") == (NOT_COMPARABLE, pd.NA)


def test_requires_both_sides_hla_columns():
    with pytest.raises(ValueError, match="missing"):
        annot_hla_agreement(pd.DataFrame({"hla_std": ["HLA-A*02:01"]}), p_groups=P_GROUPS)


def test_the_verdict_order_puts_an_agreeing_annotation_ahead_of_a_conflicting_one():
    """A TRB's best verdict is what a per-TRB summary reports, so CONFLICT must sort last."""
    assert AGREEMENT_ORDER.index(AGREE) == 0
    assert AGREEMENT_ORDER.index(CONFLICT) == len(AGREEMENT_ORDER) - 1


def test_the_columns_are_added_to_the_frame_it_was_given_without_reordering_it():
    pdf = pd.DataFrame({"hla_std": ["HLA-A*02:01", "HLA-B*07:02"],
                        "top_antigen_mhc_a_std": ["HLA-A*02:01", "HLA-A*02:01"]},
                       index=[7, 9])
    out = annot_hla_agreement(pdf, p_groups=P_GROUPS)
    assert out.index.tolist() == [7, 9]
    assert out[VERDICT_COLUMN].tolist() == [AGREE, CONFLICT]
    assert list(out.columns[:2]) == list(pdf.columns), "the input columns come first, unchanged"
    assert VERDICT_COLUMN not in pdf.columns, "the input frame is not mutated"


def test_the_fields_column_is_nullable_integer_not_float():
    pdf = pd.DataFrame({"hla_std": ["HLA-A*02:01", None],
                        "top_antigen_mhc_a_std": ["HLA-A*02:01", "HLA-A*02:01"]})
    assert annot_hla_agreement(pdf, p_groups=P_GROUPS)[VERDICT_FIELDS_COLUMN].dtype == "Int64"


# --- the verdict as a default of the join ----------------------------------------------------


def test_the_join_scores_the_allele_by_default(eco, annotations):
    joined = join_annotations(eco, annotations, trb_key=TRB_KEY, verbose=False)
    assert joined.set_index("tcr")[VERDICT_COLUMN].to_dict() == {"t2": AGREE, "t3": AGREE}


def test_the_join_can_be_asked_not_to(eco, annotations):
    joined = join_annotations(eco, annotations, trb_key=TRB_KEY, add_verdict=False, verbose=False)
    assert VERDICT_COLUMN not in joined.columns


def test_a_class_ii_row_reaching_the_join_is_labelled_not_scored(eco, annotations):
    """Without the class filter a class II ECOcluster row can match; it must not become a
    conflict, which is what comparing a DR beta chain against the databases' mhc.a would give."""
    joined = join_annotations(eco, annotations, trb_key=TRB_KEY,
                              require_same_class=False, verbose=False)
    assert joined.set_index("tcr").loc["t1", VERDICT_COLUMN] == NOT_COMPARABLE


def test_annotate_carries_the_verdict_through(eco, annotations):
    out = annotate_ecocluster(eco, annotations, trb_key=TRB_KEY, verbose=False).set_index("tcr")
    assert out.loc["t2", VERDICT_COLUMN] == AGREE
    assert out.loc["t2", VERDICT_FIELDS_COLUMN] == 2
    assert out[VERDICT_FIELDS_COLUMN].dtype == "Int64", "an unmatched row must not float it"


def test_an_unmatched_ecocluster_row_has_no_verdict_rather_than_no_allele(eco, annotations):
    """t1 matched nothing, so no allele comparison was attempted - that is not NO_ALLELE."""
    out = annotate_ecocluster(eco, annotations, trb_key=TRB_KEY, verbose=False).set_index("tcr")
    assert pd.isna(out.loc["t1", VERDICT_COLUMN])
    assert pd.isna(out.loc["t1", VERDICT_FIELDS_COLUMN])


# --- resolving the allele ----------------------------------------------------------------------

GROUPS = (frozenset({"HLA-B*07:02", "HLA-C*07:02"}),)


def resolved(eco_hla, annotation_hla=None, verdict=None):
    """One row's (trusted, allele, provenance). No annotation at all when annotation_hla is None."""
    data = {"hla_std": [eco_hla]}
    if annotation_hla is not None:
        data[ANNOTATION_HLA_COLUMN] = [annotation_hla]
        data[VERDICT_COLUMN] = [verdict]
    row = resolve_hla(pd.DataFrame(data), confusable_groups=GROUPS).iloc[0]
    return row[TRUSTED_COLUMN], row[RESOLVED_COLUMN], row[RESOLVED_FROM_COLUMN]


def test_a_trusted_allele_with_no_annotation_resolves_to_itself():
    """The 11,140-row common case: nothing contradicts the ECOcluster, so believe it."""
    assert resolved("HLA-A*02:01") == (True, "HLA-A*02:01", FROM_ECOCLUSTER)


def test_an_untrusted_allele_with_no_annotation_resolves_to_nothing():
    """Rule 1 bites hardest here: 835 class I rows with a condemned call and no replacement."""
    assert resolved("HLA-B*07:02") == (False, pd.NA, pd.NA)
    assert resolved("HLA-C*07:02") == (False, pd.NA, pd.NA)


def test_the_trust_flag_does_not_need_an_annotation_to_exist():
    """A propagation step must be able to refuse to propagate out of an untrusted row."""
    pdf = pd.DataFrame({"hla_std": ["HLA-A*02:01", "HLA-B*07:02", None]})
    out = resolve_hla(pdf, confusable_groups=GROUPS)
    assert out[TRUSTED_COLUMN].tolist()[:2] == [True, False]
    assert pd.isna(out[TRUSTED_COLUMN].iloc[2]), "no call is not an untrusted call"
    assert out[TRUSTED_COLUMN].dtype == "boolean"


def test_an_untrusted_allele_confirmed_by_the_annotation_is_believed():
    """Rule 1 says do not believe the ECOcluster alone; the database saying so is not alone."""
    assert resolved("HLA-C*07:02", "HLA-C*07:02", AGREE) == (False, "HLA-C*07:02", AGREED)


def test_a_confusable_conflict_takes_the_annotation_allele():
    """Rule 2, both directions: the label swaps, the error does not."""
    assert resolved("HLA-C*07:02", "HLA-B*07:02", CONFLICT) == (False, "HLA-B*07:02",
                                                                FROM_ANNOTATION)
    assert resolved("HLA-B*07:02", "HLA-C*07:02", CONFLICT) == (False, "HLA-C*07:02",
                                                                FROM_ANNOTATION)


def test_any_other_conflict_resolves_to_nothing_rather_than_to_either_side():
    """Asserting the ECOcluster's allele would assert something netMHCpan contradicts, and
    overriding generally is a policy that was deliberately not adopted."""
    assert resolved("HLA-A*01:01", "HLA-A*03:01", CONFLICT) == (True, pd.NA, pd.NA)
    assert resolved("HLA-C*07:02", "HLA-A*02:01", CONFLICT) == (False, pd.NA, pd.NA)


def test_agreement_keeps_the_more_resolved_name():
    """An ECOcluster B*07:02 against a one-field HLA-B*07 must not lose a field."""
    assert resolved("HLA-B*07:02", "HLA-B*07", AGREE) == (False, "HLA-B*07:02", AGREED)
    assert resolved("HLA-A*02:01", "HLA-A*02|HLA-A*02:01", AGREE) == (True, "HLA-A*02:01", AGREED)


def test_an_annotation_that_cannot_say_which_member_it_is_does_not_override():
    """Two candidates in the group leave nothing to replace the ECOcluster's call *with*.

    Unreachable with today's two-member group - sharing a member would make the verdict AGREE -
    so this is what the groups are for: it becomes reachable the moment one has three members,
    which is exactly when a flat untrusted set would start overriding on the wrong evidence.
    """
    three = (frozenset({"HLA-B*07:02", "HLA-C*07:02", "HLA-B*07:05"}),)
    pdf = pd.DataFrame({"hla_std": ["HLA-B*07:05"],
                        ANNOTATION_HLA_COLUMN: ["HLA-B*07:02|HLA-C*07:02"],
                        VERDICT_COLUMN: [CONFLICT]})
    row = resolve_hla(pdf, confusable_groups=three).iloc[0]
    assert row[TRUSTED_COLUMN] is False or not row[TRUSTED_COLUMN]
    assert pd.isna(row[RESOLVED_COLUMN]) and pd.isna(row[RESOLVED_FROM_COLUMN])

    one = (frozenset({"HLA-B*07:02", "HLA-B*07:05"}),)
    assert resolve_hla(pdf, confusable_groups=one).iloc[0][RESOLVED_COLUMN] == "HLA-B*07:02"


def test_an_uncomparable_or_allele_less_annotation_falls_back_to_the_ecocluster():
    assert resolved("HLA-A*02:01", "", NO_ALLELE) == (True, "HLA-A*02:01", FROM_ECOCLUSTER)
    assert resolved("HLA-DRB1*07:01", "", NOT_COMPARABLE)[1] == "HLA-DRB1*07:01"


def test_an_empty_policy_trusts_every_ecocluster_call():
    pdf = pd.DataFrame({"hla_std": ["HLA-B*07:02"]})
    row = resolve_hla(pdf, confusable_groups=()).iloc[0]
    assert (row[TRUSTED_COLUMN], row[RESOLVED_COLUMN]) == (True, "HLA-B*07:02")


def test_the_untrusted_set_is_derived_from_the_groups():
    assert UNTRUSTED_ECO_ALLELES == frozenset().union(*CONFUSABLE_ECO_ALLELES)


def test_resolve_needs_the_ecocluster_column():
    with pytest.raises(ValueError, match="hla_std"):
        resolve_hla(pd.DataFrame({"something_else": [1]}))


def test_annotate_resolves_every_ecocluster_row_including_the_unmatched(eco, annotations):
    """⭐ The contract the propagation step depends on: hla_resolved is NA only where the allele
    is genuinely unknown, not merely where no database knew the TRB."""
    out = annotate_ecocluster(eco, annotations, trb_key=TRB_KEY, verbose=False).set_index("tcr")
    assert pd.isna(out.loc["t1", VERDICT_COLUMN]), "t1 matched nothing"
    assert out.loc["t1", RESOLVED_COLUMN] == "HLA-DRB1*07:01", "but its own call still stands"
    assert out.loc["t1", RESOLVED_FROM_COLUMN] == FROM_ECOCLUSTER
    assert out.loc["t3", RESOLVED_FROM_COLUMN] == AGREED
