"""Collapsing 2024's HLA alleles onto the P groups the 2026 release models them as."""
import pandas as pd

from cmvividly.data.manipulation import (
    ALLELE_TO_P_GROUP,
    HLA_P_GROUPS,
    apply_hla_p_groups,
)


def pdf_hlas():
    return pd.DataFrame(
        {
            "hla": ["DRB4*01:01P", "DRB1*14:54", "DQA1*05:01+DQB1*02:01",
                    "DRB3*01:01+DRB5*02:21", None],
            "hla_2024": [None, "DRB1*14:54", "DQA1*05:01+DQB1*02:01",
                         "DRB3*01:01+DRB5*02:21", "DRB4*01:03"],
            "hla_2026": ["DRB4*01:01P", "DRB1*14:01P", "DQA1*05:01+DQB1*02:01", None, None],
            "hla_agree_2024_2026": [False, False, True, False, False],
            "hla_cocluster": ["h-DRB4_01_01P-47", "feature1_DRB1_14_54", "x", "y", None],
        }
    )


def test_members_become_their_group_and_existing_groups_are_left_alone():
    out = apply_hla_p_groups(pdf_hlas())
    assert out["hla"].fillna("<NA>").tolist() == [
        "DRB4*01:01P", "DRB1*14:01P", "DQA1*05:01+DQB1*02:01",
        "DRB3*01:01P+DRB5*02:02P", "<NA>",
    ]
    assert out["hla_2024"].fillna("<NA>").tolist() == [
        "<NA>", "DRB1*14:01P", "DQA1*05:01+DQB1*02:01",
        "DRB3*01:01P+DRB5*02:02P", "DRB4*01:01P",
    ]


def test_each_half_of_a_heterodimer_is_mapped_separately():
    """DRB3*01:01+DRB5*02:21 has a member on each side and both must move."""
    out = apply_hla_p_groups(pdf_hlas())
    assert out.loc[3, "hla"] == "DRB3*01:01P+DRB5*02:02P"


def test_agreement_is_recomputed_from_the_collapsed_names():
    """The point of the collapse: DRB1*14:54 in 2024 is DRB1*14:01P in 2026, so they agree."""
    out = apply_hla_p_groups(pdf_hlas())
    assert out["hla_agree_2024_2026"].tolist() == [False, True, True, False, False]


def test_cluster_ids_and_the_input_frame_are_untouched():
    pdf = pdf_hlas()
    out = apply_hla_p_groups(pdf)
    assert out["hla_cocluster"].equals(pdf["hla_cocluster"])
    assert pdf.loc[1, "hla_2024"] == "DRB1*14:54", "the input must not be mutated"


def test_no_member_allele_survives():
    out = apply_hla_p_groups(pdf_hlas())
    for col in ("hla", "hla_2024", "hla_2026"):
        assert not out[col].dropna().str.split("+").explode().isin(ALLELE_TO_P_GROUP).any()


def test_the_mapping_is_the_papers_full_list():
    """Kept complete even where an entry never fires on the CMV ECOcluster, so the code is
    reusable on other datasets built from the same models."""
    assert set(HLA_P_GROUPS) == {
        "A*23:01P", "DRB1*14:01P", "DRB3*01:01P", "DRB4*01:01P", "DRB5*02:02P"}
    assert all(group.endswith("P") for group in HLA_P_GROUPS)
    assert all(not allele.endswith("P") for allele in ALLELE_TO_P_GROUP)


def test_combine_applies_p_groups_by_default_and_can_be_asked_not_to():
    """⚠️ A 2024-vs-2026 comparison wants the names each release actually used, so the switch
    has to exist; everything else wants the releases made comparable, so it defaults on."""
    from cmvividly.data.manipulation import combine_2024_2026_cmv_ecoclusters

    pdf_2024 = pd.DataFrame({"tcr": ["t1"], "cdr3": ["CASSX"], "vgene": ["V1"], "jgene": ["J1"],
                             "hla": ["DRB1*14:54"], "hla_class": ["cii"],
                             "hla_cocluster": ["feature1_DRB1_14_54"]})
    pdf_2026 = pd.DataFrame({"tcr": ["t1"], "cdr3": ["CASSX"], "vgene": ["V1"], "jgene": ["J1"],
                             "hla": ["DRB1*14:01P"], "hla_class": ["cii"],
                             "hla_cocluster": ["h-DRB1_14_01P-1"]})
    collapsed = combine_2024_2026_cmv_ecoclusters(pdf_2024, pdf_2026)
    assert collapsed.loc[0, "hla_2024"] == "DRB1*14:01P"
    assert bool(collapsed.loc[0, "hla_agree_2024_2026"]) is True

    raw = combine_2024_2026_cmv_ecoclusters(pdf_2024, pdf_2026, apply_p_groups=False)
    assert raw.loc[0, "hla_2024"] == "DRB1*14:54", "the release's own name must survive"
    assert bool(raw.loc[0, "hla_agree_2024_2026"]) is False
