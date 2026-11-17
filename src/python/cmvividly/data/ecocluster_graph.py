"""Hamming-1 neighbour graphs and sequence clusters over the CMV ECOcluster.

Two TCRs one substitution apart usually share a binding solution, so the Hamming-1 graph is the
natural unit for carrying an annotation from a TCR that has one to a TCR that does not. What this
module adds over `portmanteau.sequence.hamming_n_graph` is the ECOcluster-specific part: **which
HLA boundaries an edge may cross**.

⚠️ **Condition edges on the HLA, not on the HLA cocluster.** The combined ECOcluster is an
amalgam of the 2024 and 2026 releases and the two label coclusters in **disjoint** namespaces —
2024 writes `feature10607_B_18_01`, 2026 writes `h-A_01_01-169`, and the class I name sets share
no name at all. Three quarters of rows come from only one release, so conditioning on the
cocluster severs most of the graph for a reason that is an artifact of the merge. Measured on the
class I arm: cocluster conditioning keeps 7,658 of 10,653 edges, HLA conditioning keeps 10,217.

Two alleles are allowed to connect anyway when the ECOcluster's own attribution cannot tell them
apart — see `CROSSABLE_HLA_PAIRS`.

Nothing here decides how to *use* a cluster. `ecocluster_components` returns plain connected
components; `edge_filter` is the seam for anything tighter, because single-linkage over
unweighted Hamming-1 edges chains distinct specificities together and the largest class I
component runs to 184 members.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import networkx as nx
import pandas as pd
from portmanteau.sequence.hamming_n_graph import build_seq_hamming_graph

#: Allele pairs in **linkage disequilibrium**: different loci, carried together on a haplotype,
#: so a co-occurrence-based HLA attribution cannot say which of the two a TCR belongs to.
#:
#: Each was measured against a permutation null that shuffles the HLA across TCRs, destroying any
#: sequence-HLA association while preserving exactly how common each allele is. Cross-HLA edges
#: are 4.1% of the class I graph against 96% under that null, and these three carry far more
#: than frequency predicts: `B*08:01`~`C*07:01` 9.6x, `B*14:02`~`C*08:02` 6.6x,
#: `B*07:02`~`C*07:02` 2.8x.
#:
#: ⭐ All three are HLA-B~HLA-C, which is what the genetics predicts: those genes sit about 85 kb
#: apart, the tightest linkage in the class I region, while HLA-A is far away — and the one A~B
#: pair in the data is *depleted* (0.63x) despite a high raw count. Enrichment tracks physical
#: distance. ⚠️ Rank pairs by enrichment, never by raw edge count.
#:
#: ⚠️ External support is uneven. `Cw7-B8` and `Cw7-B7` are documented common haplotypes, but
#: only at **serotype** level, and no D', r² or allele-level haplotype frequency was obtainable
#: for any pair. `B*14:02`~`C*08:02` has no external support found at all and rests on its
#: internal enrichment alone.
LD_HLA_PAIRS: tuple[frozenset[str], ...] = (
    frozenset({"B*08:01", "C*07:01"}),
    frozenset({"B*14:02", "C*08:02"}),
    frozenset({"B*07:02", "C*07:02"}),
)

#: Allele pairs at the **same locus**, so not linkage disequilibrium at all — two HLA-B alleles
#: cannot sit on one haplotype, they co-occur only in a heterozygote. `B*18:01` and `B*44:03` are
#: both assigned to the **B44 supertype** and share a Glu-at-P2 anchor preference, so an edge
#: between them may be genuine shared specificity rather than a misattribution.
#:
#: ⚠️ **That is a different claim from the LD pairs**, with a different consequence: crossing an
#: LD pair says "we cannot tell which allele this is", crossing a supertype pair says "these two
#: alleles may really present the same peptide". Kept separate so the distinction survives.
#: Internal evidence: 53 edges at 2.8x the frequency null.
SUPERTYPE_HLA_PAIRS: tuple[frozenset[str], ...] = (
    frozenset({"B*18:01", "B*44:03"}),
)

#: Pairs a Hamming-1 edge may cross. Permissive and cheap to widen: an edge that should not exist
#: adds a neighbour, it does not destroy anything.
CROSSABLE_HLA_PAIRS: tuple[frozenset[str], ...] = LD_HLA_PAIRS + SUPERTYPE_HLA_PAIRS

#: Pairs where the ECOcluster's HLA call should **not be believed on its own**. ⚠️ Deliberately
#: much narrower than `CROSSABLE_HLA_PAIRS`, and a strict subset of it, because the consequences
#: are not symmetric: permitting an edge is reversible, while distrusting an allele blanks the
#: HLA on every row carrying it. Only `B*07:02`~`C*07:02` has arbitration evidence behind it —
#: netMHCpan puts TPRVTGGGAM on `B*07:02` at %Rank_EL 0.044 against 12.1 on `C*07:02`, showing
#: the ECOcluster's call wrong on the rows where the two disagree. Widening this to the other
#: crossable pairs was measured and would cost 1,220 class I rows their resolved HLA while
#: relabelling none.
UNTRUSTED_HLA_PAIRS: tuple[frozenset[str], ...] = (
    frozenset({"B*07:02", "C*07:02"}),
)

#: The column `add_hla_crossable_key` writes.
HLA_CROSSABLE_COLUMN = "hla_crossable"


def hla_crossable_key(hla: str | None,
                      crossable: Sequence[frozenset[str]] = CROSSABLE_HLA_PAIRS) -> str | None:
    """The grouping label for one HLA: its crossable set's name, or the allele itself.

    Two alleles in the same crossable pair get the same label, which is what lets a Hamming-1
    edge join them while still refusing every other cross-HLA edge.

    Args:
        hla: an allele name as the ECOcluster writes it, without the `HLA-` prefix
            (`B*07:02`). ⚠️ A caller holding IMGT-standard names (`HLA-B*07:02`) must either
            strip the prefix or pass a `crossable` spelled its way.
        crossable: pairs (or larger sets) whose members share a label.

    Returns:
        A label, or None where `hla` is missing. The label of a set is its members sorted and
        joined with `~`, so it is deterministic and readable.
    """
    if hla is None or pd.isna(hla):
        return None
    for group in crossable:
        if hla in group:
            return "~".join(sorted(group))
    return str(hla)


def add_hla_crossable_key(pdf: pd.DataFrame, hla_column: str = "hla",
                          out_column: str = HLA_CROSSABLE_COLUMN,
                          crossable: Sequence[frozenset[str]] = CROSSABLE_HLA_PAIRS
                          ) -> pd.DataFrame:
    """Add the crossable-group label for each row's HLA. The original column is untouched."""
    if hla_column not in pdf.columns:
        raise ValueError(f"frame has no {hla_column!r} column")
    lookup = {allele: "~".join(sorted(group)) for group in crossable for allele in group}
    return pdf.assign(**{out_column: pdf[hla_column].map(lambda h: lookup.get(h, h))})


def ecocluster_hamming_graph(pdf: pd.DataFrame, *, seq_column: str = "cdr3",
                             group_cols: Sequence[str] = (),
                             hla_column: str | None = "hla",
                             crossable: Sequence[frozenset[str]] = CROSSABLE_HLA_PAIRS,
                             edge_filter: Callable[[pd.DataFrame], Iterable[bool]] | None = None,
                             **kwargs) -> nx.Graph:
    """Build the Hamming-1 graph over ECOcluster sequences, within an HLA.

    Args:
        pdf: ECOcluster rows.
        seq_column: the sequence to compare. `cdr3` or `tcr`.
        group_cols: further columns an edge may not cross. ⚠️ **V-gene-family conditioning is
            the usual choice and needs a family column this package does not build** — the
            ECOcluster's `vgene` is in Adaptive format and is family-only on some rows, so a
            caller that has converted to IMGT passes its own column here.
        hla_column: the HLA to condition on, or None to ignore the HLA entirely.
        crossable: pairs a Hamming-1 edge may cross. See `CROSSABLE_HLA_PAIRS`.
        edge_filter: given a frame of candidate edges with `node_i`, `node_j`, `seq_i` and
            `seq_j`, return one boolean per row; False drops the edge. ⭐ **The seam for
            anything tighter than single-linkage** — scoring each edge with a same-peptide model
            and dropping the weak ones, for instance. Single-linkage over unweighted Hamming-1
            edges chains distinct specificities together.
        **kwargs: passed to `portmanteau.sequence.hamming_n_graph.build_seq_hamming_graph`
            (`distance`, `exact`, `attribute_columns`, `max_pairs`, ...).

    Returns:
        A networkx graph. ⚠️ **With any grouping, a node is a tuple** `(sequence, *group values)`
        rather than the bare sequence, which is what `build_seq_hamming_graph` does; node lookups
        and `subgraph` calls need the tuple.
    """
    cols = list(group_cols)
    if hla_column is not None:
        pdf = add_hla_crossable_key(pdf, hla_column=hla_column, crossable=crossable)
        cols.append(HLA_CROSSABLE_COLUMN)
    kwargs.setdefault("distance", 1)
    kwargs.setdefault("exact", True)
    graph = build_seq_hamming_graph(pdf, seq_column=seq_column,
                                    group_cols=cols or None, **kwargs)
    if edge_filter is None:
        return graph
    edges = list(graph.edges())
    if not edges:
        return graph
    pdf_edges = pd.DataFrame({
        "node_i": [a for a, _ in edges], "node_j": [b for _, b in edges],
        "seq_i": [a[0] if isinstance(a, tuple) else a for a, _ in edges],
        "seq_j": [b[0] if isinstance(b, tuple) else b for _, b in edges],
    })
    keep = list(edge_filter(pdf_edges))
    if len(keep) != len(edges):
        raise ValueError(f"edge_filter returned {len(keep)} values for {len(edges)} edges")
    graph.remove_edges_from([e for e, k in zip(edges, keep) if not k])
    return graph


def ecocluster_components(pdf: pd.DataFrame, *, seq_column: str = "cdr3",
                          group_cols: Sequence[str] = (),
                          hla_column: str | None = "hla",
                          crossable: Sequence[frozenset[str]] = CROSSABLE_HLA_PAIRS,
                          edge_filter: Callable[[pd.DataFrame], Iterable[bool]] | None = None,
                          **kwargs) -> pd.DataFrame:
    """Label each sequence with its connected component, using a **stable** component id.

    ⚠️ `portmanteau`'s `extract_connected_components` numbers components with arbitrary integers
    that change whenever the input does, which is fine in a notebook and unusable as a key in a
    saved table. Here the id is the lexicographically smallest member, so it is derived from the
    content: the same component gets the same id across rebuilds, and two tables can be joined on
    it.

    Args and `**kwargs` are `ecocluster_hamming_graph`'s.

    Returns:
        One row per node, carrying `seq_column`, each of `group_cols`, the crossable HLA label
        where one was used, plus `component_id` and `component_size`. A sequence appearing under
        two HLAs is two rows, because it is two nodes.
    """
    cols = list(group_cols)
    if hla_column is not None:
        pdf = add_hla_crossable_key(pdf, hla_column=hla_column, crossable=crossable)
        cols.append(HLA_CROSSABLE_COLUMN)
    graph = ecocluster_hamming_graph(pdf, seq_column=seq_column, group_cols=group_cols,
                                     hla_column=hla_column, crossable=crossable,
                                     edge_filter=edge_filter, **kwargs)
    pdf_nodes = pdf[[seq_column, *cols]].drop_duplicates().reset_index(drop=True)
    label_by_node = {}
    for component in nx.connected_components(graph):
        name = min("|".join(str(part) for part in node) if isinstance(node, tuple) else str(node)
                   for node in component)
        for node in component:
            label_by_node[node] = name

    def node_of(row):
        return tuple(row) if cols else row[0]

    ids = [label_by_node[node_of(row)]
           for row in pdf_nodes[[seq_column, *cols]].itertuples(index=False, name=None)]
    out = pdf_nodes.assign(component_id=ids)
    return out.assign(component_size=out.component_id.map(out.component_id.value_counts()))
