"""Hamming-1 graphs over the ECOcluster, and which HLA boundaries an edge may cross."""
import networkx as nx
import pandas as pd
import pytest

from cmvividly.data.ecocluster_graph import (
    CROSSABLE_HLA_PAIRS,
    HLA_CROSSABLE_COLUMN,
    LD_HLA_PAIRS,
    SUPERTYPE_HLA_PAIRS,
    UNTRUSTED_HLA_PAIRS,
    add_hla_crossable_key,
    ecocluster_components,
    ecocluster_hamming_graph,
    hla_crossable_key,
)


def pdf_tcrs():
    """Four CDR3s, each one substitution from the next, spread over four HLAs."""
    return pd.DataFrame({
        "cdr3": ["CASSLGQAYEQYF", "CASSLGQAYEQYW", "CASSLGQAYEQYY", "CASSLGQAYEQYH"],
        "hla": ["B*07:02", "C*07:02", "A*02:01", "B*08:01"],
    })


# --- the constants ----------------------------------------------------------------------------


def test_crossable_is_the_union_of_the_two_reasons():
    assert set(CROSSABLE_HLA_PAIRS) == set(LD_HLA_PAIRS) | set(SUPERTYPE_HLA_PAIRS)
    assert not set(LD_HLA_PAIRS) & set(SUPERTYPE_HLA_PAIRS), "a pair has one reason, not both"


def test_ld_pairs_are_cross_locus_and_supertype_pairs_are_not():
    """Two alleles at one locus cannot share a haplotype, so a same-locus pair is never LD."""
    locus = lambda allele: allele.split("*")[0]
    for pair in LD_HLA_PAIRS:
        assert len({locus(a) for a in pair}) == 2, f"{sorted(pair)} is same-locus, not LD"
    for pair in SUPERTYPE_HLA_PAIRS:
        assert len({locus(a) for a in pair}) == 1, f"{sorted(pair)} is cross-locus"


def test_untrusted_is_a_strict_subset_of_crossable():
    """⚠️ Distrust blanks an HLA; crossing only adds an edge. The bars are not the same."""
    assert set(UNTRUSTED_HLA_PAIRS) < set(CROSSABLE_HLA_PAIRS)


def test_alleles_are_spelled_the_way_the_ecocluster_spells_them():
    """Bare, no `HLA-` prefix, so the constants match the `hla` column directly."""
    for pair in CROSSABLE_HLA_PAIRS:
        for allele in pair:
            assert not allele.startswith("HLA-") and "*" in allele


# --- the key ----------------------------------------------------------------------------------


def test_members_of_a_pair_share_a_key_and_everything_else_keeps_its_own():
    assert hla_crossable_key("B*07:02") == hla_crossable_key("C*07:02")
    assert hla_crossable_key("A*02:01") == "A*02:01"
    assert hla_crossable_key("B*07:02") != hla_crossable_key("B*08:01")


def test_the_key_is_deterministic_and_readable():
    assert hla_crossable_key("C*07:02") == "B*07:02~C*07:02"


def test_a_missing_hla_has_no_key():
    assert hla_crossable_key(None) is None
    assert hla_crossable_key(pd.NA) is None


def test_an_empty_policy_gives_every_allele_its_own_key():
    assert hla_crossable_key("B*07:02", crossable=()) == "B*07:02"


def test_add_key_leaves_the_original_column_alone():
    pdf = pdf_tcrs()
    out = add_hla_crossable_key(pdf)
    assert out.hla.equals(pdf.hla)
    assert out[HLA_CROSSABLE_COLUMN].tolist() == [
        "B*07:02~C*07:02", "B*07:02~C*07:02", "A*02:01", "B*08:01~C*07:01"]


def test_add_key_needs_the_column():
    with pytest.raises(ValueError, match="hla"):
        add_hla_crossable_key(pd.DataFrame({"something_else": [1]}))


# --- the graph --------------------------------------------------------------------------------


def test_an_edge_crosses_a_crossable_pair_and_nothing_else():
    """All four CDR3s are mutual Hamming-1 neighbours, so only the HLA decides."""
    graph = ecocluster_hamming_graph(pdf_tcrs())
    joined = {frozenset((a[0], b[0])) for a, b in graph.edges()}
    assert frozenset({"CASSLGQAYEQYF", "CASSLGQAYEQYW"}) in joined, "B*07:02~C*07:02 may cross"
    assert len(joined) == 1, f"no other pair of HLAs may cross: {joined}"


def test_without_a_policy_no_edge_crosses_an_hla():
    assert ecocluster_hamming_graph(pdf_tcrs(), crossable=()).number_of_edges() == 0


def test_ignoring_the_hla_connects_everything():
    graph = ecocluster_hamming_graph(pdf_tcrs(), hla_column=None)
    assert graph.number_of_edges() == 6, "4 mutual neighbours = 6 edges"


def test_further_group_cols_are_also_respected():
    pdf = pdf_tcrs().assign(vfamily=["TRBV1", "TRBV2", "TRBV1", "TRBV1"])
    assert ecocluster_hamming_graph(pdf, group_cols=["vfamily"]).number_of_edges() == 0


def test_edge_filter_can_drop_edges():
    """⭐ The seam for anything tighter than single linkage."""
    pdf = pdf_tcrs().assign(hla="B*07:02")
    assert ecocluster_hamming_graph(pdf).number_of_edges() == 6
    # Either endpoint, because which one is i and which is j is the graph's business.
    graph = ecocluster_hamming_graph(
        pdf, edge_filter=lambda e: e.seq_i.str.endswith("W") | e.seq_j.str.endswith("W"))
    assert graph.number_of_edges() == 3, "only the edges touching the W sequence survive"
    assert isinstance(graph, nx.Graph)


def test_edge_filter_returning_the_wrong_length_is_an_error():
    with pytest.raises(ValueError, match="edge_filter returned"):
        ecocluster_hamming_graph(pdf_tcrs().assign(hla="B*07:02"),
                                 edge_filter=lambda e: [True])


# --- components -------------------------------------------------------------------------------


def test_component_ids_are_stable_across_rebuilds():
    """⚠️ portmanteau's extract_connected_components numbers them arbitrarily, which cannot be
    used as a key in a saved table. These ids come from the content."""
    first = ecocluster_components(pdf_tcrs())
    second = ecocluster_components(pdf_tcrs().iloc[::-1].reset_index(drop=True))
    assert set(first.component_id) == set(second.component_id)
    assert first.set_index("cdr3").component_id.to_dict() == \
        second.set_index("cdr3").component_id.to_dict()


def test_the_id_is_the_smallest_member():
    out = ecocluster_components(pdf_tcrs())
    joined = out[out.component_size == 2]
    assert joined.component_id.nunique() == 1
    assert joined.component_id.iloc[0] == min(
        f"{cdr3}|{key}" for cdr3, key in zip(joined.cdr3, joined[HLA_CROSSABLE_COLUMN]))


def test_every_node_gets_a_component_and_sizes_add_up():
    out = ecocluster_components(pdf_tcrs())
    assert len(out) == 4 and out.component_id.notna().all()
    assert out.component_size.tolist() == [2, 2, 1, 1]
    assert out.drop_duplicates("component_id").component_size.sum() == len(out)


def test_one_sequence_under_two_hlas_is_two_nodes():
    pdf = pd.DataFrame({"cdr3": ["CASSX", "CASSX"], "hla": ["A*02:01", "B*08:01"]})
    out = ecocluster_components(pdf)
    assert len(out) == 2 and out.component_id.nunique() == 2
