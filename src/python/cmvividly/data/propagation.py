"""Giving a whole sequence component one peptide, and the HLA that peptide implies.

Few class I ECOcluster TRBs carry a public-database annotation, but a Hamming-1 component is a
set of near-identical betas presumed to share a binding solution. So an annotation on one member
is evidence about all of them, and propagating it is how the annotated set grows.

⭐ **One peptide per component.** A parsimony assumption, not a biological claim: a component
could genuinely respond to two CMV peptides, but assuming it does not keeps every downstream
operation single-valued.

The rules, in order:

1. **Only HLA-corroborated annotations vote.** A row votes only if `resolve_hla` gave it an
   allele - either both sides agreed, or the annotation overrode a confusable ECOcluster call.
   ⭐ This filter does most of the work in removing peptide disagreements, because a disagreeing
   annotation is usually also the one that disagrees on HLA.
2. **One vote per TRB**, for its best-supported antigen. ⚠️ Deliberately *not* one vote per
   annotation: a few annotated TRBs carry two antigens themselves, and counting both would
   let one ambiguous TRB outvote an unambiguous one. (Detecting a conflict wants every antigen;
   deciding one wants a single vote per voter. Different questions, different unit.)
3. **A unique plurality wins.** A tie leaves the component unannotated rather than guessing,
   unless the component's own HLA call breaks it - see `hla_for_peptide`.

⚠️ **The vote may decide little, and that is expected.** After rule 1 very few components still
carry two peptides. The rule earns its place by being fixed before the annotated set grows, not by
resolving a backlog.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmvividly.data.ecocluster_annotation import add_annotation_trb_columns
from cmvividly.data.ecocluster_graph import HLA_CROSSABLE_COLUMN, add_hla_crossable_key

#: What `propagate_component_annotations` can say about a component.
ASSIGNED = "assigned"
PEPTIDE_TIE = "peptide tie"
HLA_AMBIGUOUS = "hla ambiguous"
#: Where a winning peptide's HLA came from.
HLA_UNIQUE = "only allele for the peptide"
HLA_FROM_COMPONENT = "component's own HLA call"
HLA_FROM_PMIDS = "best attested by PMIDs"

#: ⭐ **How much annotation a propagated peptide rests on.** A cluster is trusted when more than
#: one TCR voted for its peptide, or when a lone vote is a large enough share of a small cluster.
#: ⚠️ **This is about the public databases being NOISY, not about the cluster being wrong**: one
#: annotation can simply be incorrect, and nothing inside the cluster can tell. Two annotations
#: have to be wrong the same way.
#:
#: ⚠️ Deliberately **not** graph distance to the nearest annotated member. That measures chaining,
#: which is a different failure - a TCR one substitution from a single *wrong* annotation is
#: maximally wrong, and a hop-based rule would call it confident.
CONFIDENCE_MIN_VOTES = 2
CONFIDENCE_MIN_FRACTION = 0.10
CONFIDENCE_COLUMN = "propagated_peptide_confidence"
HIGH = "high"
LOW = "low"

#: The default filename of the propagated table, one row per class I component node. Class II is
#: refused: see `load_propagated_ecocluster`.
PROPAGATED_ARTIFACT = "cmveco_ci_trb_propagated.tsv"
#: The artifact's key, which is also `cmveco_ci_trb_components.tsv`'s.
NODE_KEY = ["trb_vfamily", "trb_cdr3", HLA_CROSSABLE_COLUMN]
#: ⚠️ The artifact's `hla` and `hla_std` would both collide with the ECOcluster's own columns,
#: which has caused the same silent-suffix bug twice. Renamed on load; `hla_std` is dropped
#: because it is only an aggregation of the ECOcluster's own.
PROPAGATED_RENAMES = {
    "hla": "hla_propagated", "hla_from": "hla_propagated_from", "status": "propagation_status",
    "votes": "peptide_votes", "runner_up_votes": "peptide_runner_up_votes",
    "voters": "component_voters",
}


def load_propagated_ecocluster(
    processed_dir,
    *,
    pdf_eco: pd.DataFrame,
    artifact: str = PROPAGATED_ARTIFACT,
    arm: str = "ci",
    propagated_only: bool = False,
) -> pd.DataFrame:
    """The CMV ECOcluster with its component, and the peptide and HLA that component was given.

    The one call that turns "public TCRBs with an HLA association" into "TCRBs with a peptide":
    it takes the ECOcluster, attaches `component_id` and `component_size`, and left-joins the
    propagated table so `peptide` and `hla_propagated` are the propagation's, not any single
    row's own annotation.

    ⚠️ **`hla_propagated` is a separate column from `hla` and `hla_std`, deliberately.** The
    ECOcluster's own call stays untouched, and on the `B*07:02`/`C*07:02` pair the two disagree
    on purpose - that pair is exactly the one whose ECOcluster calls are not believed.

    ⛔ **This join is NOT purely additive - it drops some directly annotated class I rows.** Few
    are vote losers. Most are rows whose own HLA conflict is not adjudicable, so `resolve_hla`
    left them with no allele, so they never voted and nothing replaced them; the rest sit in a
    component whose peptide vote tied. ⚠️ Many are in SINGLETON components, where propagation
    cannot help by construction. A caller who wants those rows back has to join the annotations
    itself - see `cmvividly.data.ecocluster_annotation.join_annotations`.

    Args:
        processed_dir: the directory holding `artifact`.
        pdf_eco: the CMV ECOcluster, already loaded, with IMGT-form `vfamily_imgt`, `cdr3`,
            `hla_class` and the standardized `hla_std`. Filtered by `arm` here.
        artifact: the propagated table's filename.
        arm: the `hla_class` to load. ⛔ Class II is refused, not unimplemented: the annotation
            side records only `mhc.a`, so every class II peptide would resolve to the invariant
            DR alpha chain and the table would look confident while saying nothing. Class II
            needs an HLA comparison that handles heterodimers.
        propagated_only: keep only the rows a peptide reached. The default keeps every
            ECOcluster row, because ⭐ the null set is the work list for widening coverage.

    Returns:
        One row per ECOcluster row of the arm - the loader's own columns untouched - plus
        `component_id`, `component_size`, `peptide`, `hla_propagated`, `hla_propagated_from`,
        `propagation_status`, `peptide_votes`, `peptide_runner_up_votes`, `component_voters` and
        `propagated_peptide_confidence`. All ten are null on a row in a component no annotation
        reached - including the confidence, which is NA rather than `low` there.
    """
    if arm != "ci":
        raise ValueError(f"propagation is class I only; {arm!r} is refused, not unimplemented - "
                         "the annotation side records only the alpha chain of the MHC")
    pdf_propagated = pd.read_csv(Path(processed_dir) / artifact, sep="\t")
    if pdf_propagated.duplicated(NODE_KEY).any():
        raise AssertionError(f"{artifact} is not unique on {NODE_KEY}; a join would fan rows out")
    pdf_propagated = pdf_propagated.drop(columns=["hla_std"]).rename(columns=PROPAGATED_RENAMES)

    pdf_arm = add_hla_crossable_key(add_annotation_trb_columns(pdf_eco[pdf_eco.hla_class.eq(arm)]))

    collisions = (set(pdf_arm.columns) & set(pdf_propagated.columns)) - set(NODE_KEY)
    if collisions:
        raise AssertionError(f"{artifact} would overwrite ECOcluster columns {sorted(collisions)}")
    pdf_joined = pdf_arm.merge(pdf_propagated, on=NODE_KEY, how="left")
    if len(pdf_joined) != len(pdf_arm):
        raise AssertionError(f"join fanned {len(pdf_arm):,} rows out to {len(pdf_joined):,}")
    return pdf_joined[pdf_joined.peptide.notna()] if propagated_only else pdf_joined


def propagate_component_annotations(
    pdf_members: pd.DataFrame,
    pdf_annotated: pd.DataFrame,
    pdf_peptide_alleles: pd.DataFrame,
    *,
    trb_columns: tuple[str, ...] = ("trb_vfamily", "trb_cdr3"),
    peptide_column: str = "top_antigen",
    resolved_column: str = "hla_resolved",
) -> pd.DataFrame:
    """Decide one peptide, and one HLA, for every component carrying an annotation.

    Args:
        pdf_members: every component member, with `component_id` and `hla_std`. Supplies the
            component's own HLA call, which breaks ties the annotations cannot.
        pdf_annotated: the annotated rows, with `component_id`, the TRB columns, the peptide and
            `resolved_column`. ⚠️ Rows whose resolved HLA is null do not vote - see rule 1.
        pdf_peptide_alleles: one row per (peptide, allele) with `n_pmids`, as
            `peptide_allele_support` returns it. Says which HLA a peptide implies.
        trb_columns: the columns identifying one TRB, which is one voter.
        peptide_column: the per-TRB peptide to vote for.
        resolved_column: the column `resolve_hla` wrote.

    Returns:
        One row per annotated component: `component_id`, `voters`, `peptide`, `votes`,
        `runner_up_votes`, `hla`, `hla_from` and `status`. `peptide` is null on a tie and `hla`
        is null when the peptide implies no single allele; `status` says which.
    """
    voting = pdf_annotated[pdf_annotated[resolved_column].notna()]
    hla_by_component = (pdf_members.groupby("component_id").hla_std
                        .agg(lambda s: frozenset(s.dropna())))
    rows = []
    for component_id, group in voting.groupby("component_id"):
        votes = (group.drop_duplicates(list(trb_columns))[peptide_column]
                 .value_counts(dropna=True))
        if votes.empty:
            continue
        component_hlas = hla_by_component.get(component_id, frozenset())
        peptide, status = _winner(votes, component_hlas, pdf_peptide_alleles)
        hla, hla_from = (None, None)
        if peptide is not None:
            hla, hla_from = hla_for_peptide(peptide, component_hlas, pdf_peptide_alleles)
            if hla is None:
                status = HLA_AMBIGUOUS
        rows.append({
            "component_id": component_id, "voters": int(votes.sum()),
            "peptide": peptide, "votes": int(votes.iloc[0]),
            "runner_up_votes": int(votes.iloc[1]) if len(votes) > 1 else 0,
            "hla": hla, "hla_from": hla_from, "status": status,
        })
    return pd.DataFrame(rows, columns=["component_id", "voters", "peptide", "votes",
                                       "runner_up_votes", "hla", "hla_from", "status"])


def add_peptide_confidence(
    pdf: pd.DataFrame,
    *,
    votes_column: str = "votes",
    size_column: str = "component_size",
    peptide_column: str = "peptide",
    min_votes: int = CONFIDENCE_MIN_VOTES,
    min_fraction: float = CONFIDENCE_MIN_FRACTION,
) -> pd.DataFrame:
    """Add `propagated_peptide_confidence`: whether a propagated peptide rests on enough evidence.

    `high` when **either** at least `min_votes` TCRs voted for the winning peptide, **or** a
    smaller number of votes is at least `min_fraction` of the cluster. The second arm is what
    keeps a 1-of-2 cluster trusted while a 1-of-40 cluster is not.

    ⚠️ **The numerator is votes, not annotations.** A TCR whose own annotation *lost* the vote is
    not support for the winner - if 13 TCRs in a cluster are annotated and 12 voted for the
    winning peptide, the evidence is 12.

    ⚠️ **A cluster with no propagated peptide gets NA, not `low`.** "Not annotated" and "annotated
    on thin evidence" are different states and a consumer filtering on `low` should not catch the
    first.

    Args:
        pdf: a frame carrying the vote count, the cluster size and the peptide. One row per
            component or one per member both work - the rule is a function of the row.
        votes_column: TCRs that voted for the winning peptide.
        size_column: TCRs in the cluster, the denominator.
        peptide_column: used only to find the rows that have no peptide at all.
        min_votes: votes that make a cluster trusted whatever its size.
        min_fraction: share of the cluster that makes fewer votes trusted anyway.

    Returns:
        A copy of `pdf` with `CONFIDENCE_COLUMN` appended, as a nullable `string`.
    """
    for column in (votes_column, size_column, peptide_column):
        if column not in pdf.columns:
            raise ValueError(f"frame has no {column!r} column to judge confidence from")
    votes = pd.to_numeric(pdf[votes_column], errors="coerce")
    size = pd.to_numeric(pdf[size_column], errors="coerce")
    trusted = votes.ge(min_votes) | votes.div(size).ge(min_fraction)
    verdict = trusted.map({True: HIGH, False: LOW})
    return pdf.assign(**{
        CONFIDENCE_COLUMN: verdict.where(pdf[peptide_column].notna()).astype("string")})


def hla_for_peptide(peptide: str, component_hlas: frozenset[str],
                    pdf_peptide_alleles: pd.DataFrame) -> tuple[str | None, str | None]:
    """The allele a winning peptide implies, and how it was chosen.

    Three rules, in order:

    1. One allele recorded for the peptide - use it.
    2. Several, and the **component's own HLA call** is one of them - use that, because the
       ECOcluster independently says this is the allele these TCRs are restricted by.
    3. Several, and the component cannot choose between them - use the **best attested by
       distinct PMIDs**. Still tied, and nothing is returned: an unannotated HLA beats a guess.
    """
    alleles = pdf_peptide_alleles[pdf_peptide_alleles.peptide.eq(peptide)]
    if alleles.empty:
        return None, None
    distinct = sorted(set(alleles.allele))
    if len(distinct) == 1:
        return distinct[0], HLA_UNIQUE
    agreeing = sorted(set(distinct) & set(component_hlas))
    if len(agreeing) == 1:
        return agreeing[0], HLA_FROM_COMPONENT
    by_pmids = (alleles[alleles.allele.isin(agreeing or distinct)]
                .groupby("allele").n_pmids.max().sort_values(ascending=False))
    if len(by_pmids) and (len(by_pmids) == 1 or by_pmids.iloc[0] > by_pmids.iloc[1]):
        return by_pmids.index[0], HLA_FROM_PMIDS
    return None, None


def peptide_allele_support(pdf_long: pd.DataFrame, *,
                           list_separator: str = "|") -> pd.DataFrame:
    """One row per (peptide, allele) with the count of distinct PMIDs behind it.

    ⚠️ A **union** of PMIDs, never a sum: one paper reports many TRBs and two databases cite the
    same paper, so summing counts one source many times.

    Args:
        pdf_long: the long annotation table, one row per (TRB, antigen, database), with
            `antigen`, `mhc_a_std` and `pmids`.
    """
    exploded = pdf_long.assign(allele=pdf_long.mhc_a_std.fillna("").str.split(list_separator))
    exploded = exploded.explode("allele")
    exploded = exploded[exploded.allele.fillna("").ne("")]
    return (exploded.groupby(["antigen", "allele"]).pmids
            .agg(lambda s: len({p for v in s.dropna()
                                for p in str(v).split(list_separator) if p}))
            .rename("n_pmids").reset_index().rename(columns={"antigen": "peptide"}))


def _winner(votes: pd.Series, component_hlas: frozenset[str],
            pdf_peptide_alleles: pd.DataFrame) -> tuple[str | None, str]:
    """The peptide with a unique plurality, or the one the component's HLA call singles out."""
    if len(votes) == 1 or votes.iloc[0] > votes.iloc[1]:
        return votes.index[0], ASSIGNED
    tied = [p for p, n in votes.items() if n == votes.iloc[0]]
    compatible = [p for p in tied
                  if set(pdf_peptide_alleles.loc[pdf_peptide_alleles.peptide.eq(p), "allele"])
                  & set(component_hlas)]
    if len(compatible) == 1:
        return compatible[0], ASSIGNED
    return None, PEPTIDE_TIE
