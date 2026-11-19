"""Joining CMV ECOcluster TRBs to the public databases' peptide annotations.

The ECOcluster says *this TRB is CMV-associated, restricted by this HLA* but names no peptide.
The public databases name peptides. Joining them on the TRB is how the ECOcluster gets a putative
antigen.

Two entry points, for the two shapes that are wanted:

- **`annotate_ecocluster`** - one row per ECOcluster row, annotation columns filled where the
  public databases had something and NA where they did not. This is the ECOcluster *with* an
  antigen guess, and is what downstream work reads.
- **`join_annotations`** - the intersection alone, one row per matched pair. Use it to measure
  the match: how many TRBs, under which class, at what cost from the class filter.

**MHC class is a hard filter here.** A class I ECOcluster TRB matched to a class II annotation is
wrong whatever the CDR3 says: the two sides are describing different receptors that happen to
share a sequence, or one of them is mislabelled. Both sides already carry `hla_class` spelled the
same way (`ci`/`cii`) — see `portmanteau.tcr.hla` — so the filter is an equality condition on the
join rather than a post-hoc check.

**Rows whose class is unknown are dropped, not passed through.** In the public databases these
are a handful of rows: McPAS rows with no MHC recorded at all, and IEDB rows naming MR1 or CD1,
which are class I-*like* but not HLA. Silently matching them to either class would be worse than
losing them.

⚠️ A CDR3 appearing under both classes is **not** an ambiguity for this join. A few
ECOcluster CDR3s, and a VDJdb TRB, do this; each individual *row* has a definite class, and a
row-wise equality filter does the right thing with all of them.

**Class is the coarse filter; the allele is the fine one, and it is measured but not imposed.**
`annot_hla_agreement` labels each joined row `agree` / `conflict` / `no allele` /
`not comparable` against the allele the annotation's top antigen was actually recorded on, and
says in `hla_verdict_fields` how deep the comparison went. `join_annotations` and
`annotate_ecocluster` run it by default.

It deliberately does not drop anything. A one-field annotation allele (`HLA-A*02`) agrees with the
ECOcluster's `A*02:01` *and* with its `A*02:06`, so it is an `agree` at 1 field — and **that
counts as a match**. So `hla_verdict == AGREE` is the whole test, and
`hla_verdict_fields` says how deep the comparison went without gating on it.
"""
from __future__ import annotations

import pandas as pd

from cmvividly.data.ecocluster_graph import UNTRUSTED_HLA_PAIRS
from cmvividly.data.manipulation import HLA_P_GROUPS
from cmvividly.data.publicdb import load_cmv_trb_antigens
from portmanteau.tcr.publicdb.common import LIST_SEPARATOR

from portmanteau.tcr.hla import (
    CLASS_I,
    expand_p_group,
    hla_class,
    hla_field_count,
    normalize_hla,
)

#: `HLA_P_GROUPS` in standard IMGT form, for expanding a P group to its members at join time. A P
#: group names a set of alleles, so an ECOcluster `hla_std` can hold a name no allele equals.
HLA_P_GROUPS_STD = {
    normalize_hla(p_group): [normalize_hla(allele) for allele in alleles]
    for p_group, alleles in HLA_P_GROUPS.items()
}

#: How the two sides name the same thing.
ECO_TRB_COLUMNS = {"vfamily_imgt": "trb_vfamily", "cdr3": "trb_cdr3"}
#: The column both sides carry, spelled identically, that the filter keys on.
CLASS_COLUMN = "hla_class"
DEFAULT_TRB_KEY = ("trb_vfamily", "trb_cdr3")
#: Carries each ECOcluster row's identity through the intersection and back. Dropped on the way out.
ROW_ID_COLUMN = "_ecocluster_row"

#: Where each side's HLA allele lives on a joined frame. The ECOcluster's own standardized name,
#: and the alleles the *top antigen* was recorded on - see
#: `portmanteau.tcr.publicdb.combined.top_antigens` for why the allele rides with the antigen rather than with the TRB.
ECO_HLA_COLUMN = "hla_std"
ANNOTATION_HLA_COLUMN = "top_antigen_mhc_a_std"

#: The two columns `annot_hla_agreement` adds.
VERDICT_COLUMN = "hla_verdict"
VERDICT_FIELDS_COLUMN = "hla_verdict_fields"
#: The three `resolve_hla` adds.
TRUSTED_COLUMN = "hla_eco_trusted"
RESOLVED_COLUMN = "hla_resolved"
RESOLVED_FROM_COLUMN = "hla_resolved_from"

#: ⚠️ Sets of alleles the ECOcluster's co-occurrence-based HLA attribution **cannot tell apart**,
#: so its call naming any of them carries no information about which member it is. Grouped rather
#: than flat because the two rules differ: membership alone condemns a call, but only a database
#: allele from the **same** group may replace it.
#:
#: ⭐ **The list itself is CMVividly's `UNTRUSTED_HLA_PAIRS`**, an ECOcluster fact rather than an
#: assay decision, and is deliberately much narrower than its `CROSSABLE_HLA_PAIRS`: permitting a
#: clustering edge is reversible, while distrusting an allele blanks the HLA on every row
#: carrying it. Only translated here, from the ECOcluster's bare spelling to the IMGT-standard
#: one `hla_std` uses — the same pattern as `HLA_P_GROUPS_STD`.
CONFUSABLE_ECO_ALLELES = tuple(
    frozenset(normalize_hla(allele) for allele in pair) for pair in UNTRUSTED_HLA_PAIRS)
#: Never believe an ECOcluster HLA call naming one of these. Derived, so there is one thing to edit.
UNTRUSTED_ECO_ALLELES = frozenset().union(*CONFUSABLE_ECO_ALLELES)

#: Where a resolved allele came from. `AGREED` is both sides; `FROM_ANNOTATION` means the
#: ECOcluster's call was overridden; `FROM_ECOCLUSTER` means nothing contradicted it.
AGREED = "agreed"
FROM_ANNOTATION = "annotation"
FROM_ECOCLUSTER = "ecocluster"

#: What `annot_hla_agreement` can say about one joined row.
AGREE = "agree"
CONFLICT = "conflict"
NO_ALLELE = "no allele"
NOT_COMPARABLE = "not comparable"
#: Best to worst, so a per-TRB summary can take a TRB's best verdict. `CONFLICT` is last: a TRB
#: with any agreeing annotation is not in conflict, it has a weaker annotation that disagrees.
AGREEMENT_ORDER = [AGREE, NO_ALLELE, NOT_COMPARABLE, CONFLICT]


def annotate_ecocluster(
    pdf_eco: pd.DataFrame,
    pdf_annotations: pd.DataFrame | None = None,
    *,
    trb_key: tuple[str, ...] = DEFAULT_TRB_KEY,
    require_same_class: bool = True,
    add_verdict: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return the ECOcluster with public-database antigen columns attached, one row per input row.

    The intersection from `join_annotations` is left-joined back onto the ECOcluster, so no row
    is lost: an ECOcluster TRB no public database knows keeps all its own columns and gets NA for
    the annotation ones. That is the difference from `join_annotations`, which returns only what
    matched.

    This is one row per ECOcluster row because the annotation side is one row per
    (TRB, MHC class) - see `portmanteau.tcr.publicdb.combined.top_antigens`. Passing the long
    table instead would fan rows out, so that is rejected rather than silently allowed.

    Args:
        pdf_eco: the CMV ECOcluster, with IMGT-form `vfamily_imgt`, `cdr3` and the standardized
            `hla_std` columns.
        pdf_annotations: the flattened per-TRB table; `load_cmv_trb_antigens(trb_key)` by default.
        trb_key: the columns defining one TRB.
        require_same_class: match MHC class as well as TRB. See `join_annotations`.
        add_verdict: carry `hla_verdict` and `hla_verdict_fields` through onto the result, and
            run `resolve_hla` over **every** ECOcluster row. See `annot_hla_agreement`.
        verbose: print how much of the ECOcluster got annotated.

    Returns:
        `pdf_eco`'s own columns plus every annotation column, the latter suffixed `_annotation`
        where the name collides, plus the two verdict columns unless `add_verdict` is False.
        Integer annotation columns come back as nullable `Int64` rather than floated by the
        unmatched rows, and so do the verdict fields. Row count and row order match `pdf_eco`.

        ⚠️ An **unmatched** row gets NA for the verdict, which is not `NO_ALLELE`: no public
        database knew this TRB at all, so no allele comparison was ever attempted. It still gets
        a trust flag and a resolved allele, because `resolve_hla` runs over the whole frame and
        an ECOcluster call nothing contradicts is still the best available answer.

        The TRB key columns are *not* added. They duplicate `cdr3`/`vfamily_imgt` under the names
        a per-cell scRNA frame uses for its own chains, so leaving them on would make a later join to
        such a frame suffix both sides' `trb_cdr3` and break every downstream reference to it.
    """
    if pdf_annotations is None:
        pdf_annotations = load_cmv_trb_antigens(trb_key)
    left = pdf_eco if set(trb_key) <= set(pdf_eco.columns) else add_annotation_trb_columns(pdf_eco)
    tagged = left.assign(**{ROW_ID_COLUMN: range(len(left))})

    pdf_matched = join_annotations(tagged, pdf_annotations, trb_key=trb_key,
                                   require_same_class=require_same_class,
                                   add_verdict=add_verdict, verbose=verbose)
    on = [*trb_key, CLASS_COLUMN] if require_same_class else list(trb_key)
    added = [column for column in pdf_annotations.columns if column not in on]
    # The verdicts are computed on the intersection, not present on `pdf_annotations`, so they
    # have to be named explicitly or they would be dropped on the way back out. The *resolution*
    # is deliberately NOT carried over: it is recomputed below on the whole ECOcluster, because
    # an unannotated row still has a trust flag and often still has a usable allele.
    added += [VERDICT_COLUMN, VERDICT_FIELDS_COLUMN] if add_verdict else []

    fanned = int(pdf_matched[ROW_ID_COLUMN].duplicated().sum())
    if fanned:
        raise ValueError(
            f"{fanned:,} ECOcluster rows matched more than one annotation row, so the result "
            f"would not be one row per ECOcluster row; pdf_annotations must be keyed on "
            f"{on}, as load_cmv_trb_antigens returns it"
        )

    pdf_annotated = tagged.merge(pdf_matched[[ROW_ID_COLUMN, *added]], on=ROW_ID_COLUMN,
                                 how="left", suffixes=("", "_annotation"))
    # An unmatched row makes every added column nullable, which silently floats the counts.
    counts = {column: "Int64" for column in added
              if column in pdf_annotations.columns
              and pd.api.types.is_integer_dtype(pdf_annotations[column])}
    if add_verdict:
        counts[VERDICT_FIELDS_COLUMN] = "Int64"
    pdf_annotated = pdf_annotated.astype(counts)
    if add_verdict:
        pdf_annotated = resolve_hla(pdf_annotated)
    scaffolding = [column for column in trb_key if column not in pdf_eco.columns]
    if verbose:
        annotated = pdf_annotated[added[0]].notna().sum() if added else 0
        print(f"{annotated:,} of {len(pdf_annotated):,} ECOcluster rows annotated "
              f"({100 * annotated / len(pdf_annotated):.1f}%)")
    return pdf_annotated.drop(columns=[ROW_ID_COLUMN, *scaffolding])


def join_annotations(
    pdf_eco: pd.DataFrame,
    pdf_annotations: pd.DataFrame,
    *,
    trb_key: tuple[str, ...] = DEFAULT_TRB_KEY,
    require_same_class: bool = True,
    add_verdict: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Inner-join ECOcluster TRBs to peptide annotations, matching MHC class.

    Args:
        pdf_eco: the CMV ECOcluster, with IMGT-form `vfamily_imgt`, `cdr3` and the standardized
            `hla_std` columns. Converted with
            `add_annotation_trb_columns` if it does not already carry the TRB column names.
        pdf_annotations: one row per (TRB, antigen), as `load_cmv_trb_antigens()` returns it,
            keyed the same way as `trb_key`.
        trb_key: the columns defining one TRB. Must be present on the annotation side and
            derivable on the ECOcluster side.
        require_same_class: the point of this function. False joins on the TRB alone, which is
            only useful for measuring what the class filter costs.
        add_verdict: run `annot_hla_agreement` and `resolve_hla` on the result, adding
            `hla_verdict`, `hla_verdict_fields`, `hla_eco_trusted`, `hla_resolved` and
            `hla_resolved_from`. On by default because class agreement without allele agreement
            is a half-answer, and the half that is missing is the one that is wrong. Needs both
            sides' HLA columns; pass False for a frame that has neither.
        verbose: print what was dropped and why.

    Returns:
        One row per (ECOcluster row, matching annotation), with the ECOcluster columns suffixed
        `_eco` where they collide. Unclassified rows on either side never appear. Plus the two
        verdict columns unless `add_verdict` is False.
    """
    left = pdf_eco if set(trb_key) <= set(pdf_eco.columns) else add_annotation_trb_columns(pdf_eco)

    on = list(trb_key)
    if require_same_class:
        for name, frame in (("ECOcluster", left), ("annotations", pdf_annotations)):
            if CLASS_COLUMN not in frame.columns:
                raise ValueError(f"{name} frame has no {CLASS_COLUMN!r} column to filter on")
        unknown_left = left[CLASS_COLUMN].isna().sum()
        unknown_right = pdf_annotations[CLASS_COLUMN].isna().sum()
        left = left[left[CLASS_COLUMN].notna()]
        pdf_annotations = pdf_annotations[pdf_annotations[CLASS_COLUMN].notna()]
        on.append(CLASS_COLUMN)
        if verbose and (unknown_left or unknown_right):
            print(f"dropped rows with no MHC class: {unknown_left:,} ECOcluster, "
                  f"{unknown_right:,} annotation")

    pdf_joined = left.merge(pdf_annotations, on=on, how="inner", suffixes=("_eco", ""))
    if verbose:
        matched = pdf_joined.drop_duplicates(list(trb_key))
        print(f"{len(pdf_joined):,} (ECOcluster row, annotation) pairs over "
              f"{len(matched):,} distinct TRBs")
        if require_same_class:
            print(f"  by class: {pdf_joined[CLASS_COLUMN].value_counts().to_dict()}")
    if add_verdict:
        pdf_joined = resolve_hla(annot_hla_agreement(pdf_joined))
        if verbose:
            print(f"  by HLA verdict: {pdf_joined[VERDICT_COLUMN].value_counts().to_dict()}")
            print(f"  resolved from: "
                  f"{pdf_joined[RESOLVED_FROM_COLUMN].value_counts(dropna=False).to_dict()}")
    return pdf_joined


def annot_hla_agreement(
    pdf_joined: pd.DataFrame,
    *,
    eco_hla_column: str = ECO_HLA_COLUMN,
    annotation_hla_column: str = ANNOTATION_HLA_COLUMN,
    p_groups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Add `hla_verdict` and `hla_verdict_fields`: do the two sides agree on the HLA allele?

    The class filter in `join_annotations` is not enough: an annotation can place a peptide on an
    allele no public database ever records it under, and those score at untested levels, so they
    are errors rather than edge cases. This **labels and does not drop** - whether a verdict is
    disqualifying is a join policy the caller owns.

    `hla_verdict`, worst-last in `AGREEMENT_ORDER`:

    - `AGREE` - some ECOcluster allele and some annotation allele match on every field *both* of
      them resolve. P groups on the ECOcluster side are expanded to their members first, because
      a P group names a set and can never equal an allele name.
    - `CONFLICT` - no pair matches that far.
    - `NO_ALLELE` - one side recorded none, so there was nothing to compare.
    - `NOT_COMPARABLE` - the ECOcluster HLA is not class I. ⚠️ The annotation side is `mhc.a`
      alone, so a class II DR row would otherwise be compared against the invariant `HLA-DRA*01:01`
      and called a conflict on every row, and a `+`-joined DQ/DP heterodimer would never match
      anything. Class II needs `mhc_b_std`, which this does not use.

    `hla_verdict_fields` is **the resolution the verdict was reached at**, 1 or 2, NA for the two
    non-judgements. It is `min(fields)` over the pair that decided the verdict - the deepest
    agreeing pair for `AGREE`, the deepest comparison made for `CONFLICT`. So `HLA-A*02:01`
    against a bare `HLA-A*02` is `AGREE` at **1 field**, and against `HLA-A*02:06` it is
    `CONFLICT` at **2**: the column never claims two different proteins match.

    A one-field agreement - `HLA-A*02` cannot be told from the ECOcluster's `A*02:01` *or* its
    `A*02:06` - **counts as a match**, so a filter is on `hla_verdict` alone
    and this column is reporting. `resolve_hla` still keeps the deeper of the two names.

    Args:
        pdf_joined: any frame carrying both columns - `join_annotations` output or
            `annotate_ecocluster` output, neither of which suffixes either name.
        eco_hla_column: the ECOcluster's standardized HLA, one name per row.
        annotation_hla_column: the annotation side's alleles, LIST_SEPARATOR-joined.
        p_groups: P group -> member alleles; `HLA_P_GROUPS_STD` by default.

    Returns:
        A copy of `pdf_joined` with `VERDICT_COLUMN` (string) and `VERDICT_FIELDS_COLUMN` (Int64)
        appended. Row count and order are unchanged.
    """
    if p_groups is None:
        p_groups = HLA_P_GROUPS_STD
    missing = [c for c in (eco_hla_column, annotation_hla_column) if c not in pdf_joined.columns]
    if missing:
        raise ValueError(f"joined frame is missing {missing}; expected a frame carrying both "
                         f"sides' HLA, and an annotation table carrying the allele columns")

    eco = pdf_joined[eco_hla_column].fillna("").astype(str)
    annotation = pdf_joined[annotation_hla_column].fillna("").astype(str)
    pairs = pd.Series(list(zip(eco, annotation)), index=pdf_joined.index)
    decided = {pair: _hla_verdict(*pair, p_groups) for pair in set(pairs)}
    return pdf_joined.assign(**{
        VERDICT_COLUMN: pairs.map({pair: v for pair, (v, _) in decided.items()}).astype("string"),
        VERDICT_FIELDS_COLUMN: pairs.map(
            {pair: f for pair, (_, f) in decided.items()}).astype("Int64"),
    })


def resolve_hla(
    pdf: pd.DataFrame,
    *,
    eco_hla_column: str = ECO_HLA_COLUMN,
    annotation_hla_column: str = ANNOTATION_HLA_COLUMN,
    confusable_groups: tuple[frozenset[str], ...] = CONFUSABLE_ECO_ALLELES,
) -> pd.DataFrame:
    """Add `hla_eco_trusted`, `hla_resolved` and `hla_resolved_from`: which allele to actually use.

    `annot_hla_agreement` describes a disagreement. This decides one, under a policy that is a
    parameter rather than a hidden rule. ⭐ **`hla_resolved` is NA exactly where the allele is
    genuinely unknown**, which makes the null set the work list for anything that fills gaps -
    annotation propagation above all.

    Two rules, both about `CONFUSABLE_ECO_ALLELES`:

    1. An ECOcluster call naming an untrusted allele is **never** believed on its own, even where
       the annotation agrees with it and even where there is no annotation at all. `hla_eco_trusted`
       carries this and depends **only** on `eco_hla_column`, so it is defined on unannotated rows
       too and a propagation step can refuse to propagate *out of* one.
    2. A conflict is overridden by the annotation **only** when the annotation's allele sits in the
       same confusable group as the ECOcluster's. ⚠️ Any other conflict resolves to **NA, not to
       either side**: the annotation may well be right, but asserting the ECOcluster's call would
       be asserting something the annotation contradicts, and overriding on a conflict in general
       is not attempted here.

    On agreement the **more resolved** of the two names wins, so an ECOcluster `HLA-B*07:02`
    against a one-field `HLA-B*07` resolves to `HLA-B*07:02` rather than losing a field.

    Args:
        pdf: a frame carrying `eco_hla_column`; `annotation_hla_column` and `VERDICT_COLUMN` where
            there is an annotation. A row with no verdict is treated as unannotated, which is what
            `annotate_ecocluster` produces for an ECOcluster TRB no database knows.
        eco_hla_column: the ECOcluster's standardized HLA.
        annotation_hla_column: the annotation side's alleles, LIST_SEPARATOR-joined.
        confusable_groups: the policy. An empty tuple trusts every ECOcluster call.

    Returns:
        A copy of `pdf` with the three columns appended; `hla_eco_trusted` is `boolean` and the
        other two are `string`. Row count and order are unchanged.
    """
    if eco_hla_column not in pdf.columns:
        raise ValueError(f"frame has no {eco_hla_column!r} column to resolve from")
    untrusted = frozenset().union(*confusable_groups) if confusable_groups else frozenset()

    eco = pdf[eco_hla_column].fillna("").astype(str)
    if annotation_hla_column in pdf.columns:
        annotation = pdf[annotation_hla_column].fillna("").astype(str)
        verdict = (pdf[VERDICT_COLUMN].fillna("") if VERDICT_COLUMN in pdf.columns
                   else pd.Series("", index=pdf.index))
    else:
        annotation = pd.Series("", index=pdf.index)
        verdict = pd.Series("", index=pdf.index)

    keys = pd.Series(list(zip(eco, annotation, verdict.astype(str))), index=pdf.index)
    decided = {key: _resolve_one(*key, untrusted, confusable_groups) for key in set(keys)}
    return pdf.assign(**{
        TRUSTED_COLUMN: eco.map(lambda name: bool(name) and name not in untrusted
                                ).where(eco.ne(""), pd.NA).astype("boolean"),
        RESOLVED_COLUMN: keys.map({k: v for k, (v, _) in decided.items()}).astype("string"),
        RESOLVED_FROM_COLUMN: keys.map({k: f for k, (_, f) in decided.items()}).astype("string"),
    })


def add_annotation_trb_columns(pdf_eco: pd.DataFrame) -> pd.DataFrame:
    """Add `trb_vfamily`/`trb_cdr3` to an ECOcluster frame, so it joins to an annotation table.

    The originals are kept: callers may read `cdr3` and
    `vfamily_imgt`, and renaming in place would break them.
    """
    missing = [column for column in ECO_TRB_COLUMNS if column not in pdf_eco.columns]
    if missing:
        raise ValueError(f"ECOcluster frame is missing {missing}; expected IMGT-form "
                         "`vfamily_imgt` and `cdr3` columns")
    return pdf_eco.assign(**{new: pdf_eco[old] for old, new in ECO_TRB_COLUMNS.items()})


def _hla_verdict(
    eco_name: str, annotation_names: str, p_groups: dict[str, list[str]]
) -> tuple[str, int | None]:
    """The (verdict, resolution) for one (ECOcluster HLA, joined annotation alleles) pair.

    See `annot_hla_agreement` for what each verdict means and why the resolution is a `min`.
    """
    if not eco_name:
        return NO_ALLELE, pd.NA
    if hla_class(eco_name) != CLASS_I:
        return NOT_COMPARABLE, pd.NA

    eco_alleles = _with_field_counts(expand_p_group(eco_name, p_groups))
    annotation_alleles = _with_field_counts(annotation_names.split(LIST_SEPARATOR))
    if not eco_alleles or not annotation_alleles:
        return NO_ALLELE, pd.NA

    agreeing = [min(eco_fields, annotation_fields)
                for eco_allele, eco_fields in eco_alleles
                for annotation_allele, annotation_fields in annotation_alleles
                if _agrees_as_far_as_both_resolve(eco_allele, annotation_allele)]
    if agreeing:
        return AGREE, max(agreeing)
    return CONFLICT, max(min(eco_fields, annotation_fields)
                         for _, eco_fields in eco_alleles
                         for _, annotation_fields in annotation_alleles)


def _with_field_counts(names: list[str]) -> list[tuple[str, int]]:
    """Keep the names that resolve to a field count, paired with it.

    A name with no fields at all - IEDB records `B2M` and bare class names - cannot be compared
    at a resolution, so it is dropped here rather than silently counted as a conflict.
    """
    counted = [(name, hla_field_count(name)) for name in names if name]
    return [(name, fields) for name, fields in counted if fields is not None]


def _agrees_as_far_as_both_resolve(left: str, right: str) -> bool:
    """Whether two allele names match on every field *both* of them resolve.

    Equal resolution makes this plain equality. Unequal resolution compares only the fields the
    shallower name reached: `HLA-A*02:01` agrees with `HLA-A*02` and not with `HLA-B*07`.
    """
    left_fields, right_fields = left.split(":"), right.split(":")
    shared = min(len(left_fields), len(right_fields))
    return left_fields[:shared] == right_fields[:shared]


def _resolve_one(eco_name: str, annotation_names: str, verdict: str, untrusted: frozenset,
                 confusable_groups: tuple[frozenset[str], ...]) -> tuple[object, object]:
    """The (allele, provenance) for one row. See `resolve_hla` for the policy."""
    if not eco_name:
        return pd.NA, pd.NA
    trusted = eco_name not in untrusted
    annotation_alleles = {name for name in annotation_names.split(LIST_SEPARATOR) if name}

    if verdict == AGREE:
        agreeing = [name for name in annotation_alleles
                    if _agrees_as_far_as_both_resolve(eco_name, name)]
        return max([eco_name, *agreeing], key=_field_count), AGREED
    if verdict == CONFLICT:
        replacement = _same_group_allele(eco_name, annotation_alleles, confusable_groups)
        if replacement is not None:
            return replacement, FROM_ANNOTATION
        # Contested and not adjudicable under this policy: assert neither side.
        return pd.NA, pd.NA
    # No annotation, or one that could not be compared: the ECOcluster's call is all there is.
    return (eco_name, FROM_ECOCLUSTER) if trusted else (pd.NA, pd.NA)


def _same_group_allele(eco_name: str, annotation_alleles: set[str],
                       confusable_groups: tuple[frozenset[str], ...]) -> str | None:
    """The annotation allele confusable with this ECOcluster call, if there is exactly one.

    Two in the same group would mean the annotation itself cannot say which member it is, so
    there is nothing to replace the ECOcluster's call *with*.
    """
    for group in confusable_groups:
        if eco_name not in group:
            continue
        candidates = sorted(annotation_alleles & (group - {eco_name}))
        if len(candidates) == 1:
            return candidates[0]
    return None


def _field_count(name: str) -> int:
    """How many fields a normalized allele name resolves to; 1 for `HLA-A*02`, 2 for `HLA-A*02:01`."""
    return name.count(":") + 1
