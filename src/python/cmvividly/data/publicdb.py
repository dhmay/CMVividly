"""CMV records from the public TCR-antigen databases: VDJdb, McPAS-TCR and IEDB.

The reading, reshaping and caching live in `portmanteau.tcr.publicdb`, which is deliberately
species-agnostic: it takes a `SpeciesPreset` naming the species in each database rather than
hardcoding one. What belongs here is the CMV half of that — the three spellings, and thin views
that bind them.

⚠️ **The three databases do not name a species the same way, and the obvious approach is wrong.**
A `cmv|cytomegalo` pattern looks like it should work and does not: it catches **LCMV** in McPAS,
which is a different virus entirely, and murine CMV in both McPAS and IEDB. So two of the three
are matched exactly and IEDB is matched on a prefix, because IEDB appends strain and host detail
to the organism name.

Three views, each cached, each built from the one above:

- **`load_cmv_rows()`** - every CMV record from every source, one row as the source wrote it.
  The only one that keeps **gene-level pairing**: which alpha chain, with which V and J, sat
  beside which beta. That is lost the moment rows are grouped.
- **`load_cmv_annotations()`** - one row per (TRB, antigen, database), keeping every peptide,
  reference and paired alpha.
- **`load_cmv_trb_antigens()`** - the flattened view, one row per (TRB, MHC class) with its
  best-supported antigen and runner-up. Lossy by design, and what an equality join against a
  beta-chain table wants.

⚠️ **Where the files land is not this module's choice.** `portmanteau.tcr.publicdb.cache`
defaults to the OS cache directory and reads `PORTMANTEAU_CACHE` on every call, so a consumer
whose cache directory is the wrong filesystem sets that variable and nothing here changes.
"""

from __future__ import annotations

import pandas as pd
from portmanteau.tcr.publicdb import combined
from portmanteau.tcr.publicdb.common import DEFAULT_TRB_KEY
from portmanteau.tcr.publicdb.species import SpeciesPreset

__all__ = ["CMV", "load_cmv_annotations", "load_cmv_rows", "load_cmv_trb_antigens"]

#: Human cytomegalovirus, as each of the three databases spells it.
#:
#: - VDJdb: `antigen.species` is a short controlled value, matched exactly.
#: - McPAS: `Pathology` is matched exactly. ⚠️ A pattern here also catches LCMV.
#: - IEDB: `antigen.species` is matched on a **prefix**, because both human CMV spellings begin
#:   with this and strain detail follows. Matching "cytomegalo" instead would also catch
#:   "Murid betaherpesvirus 1 (Mouse cytomegalovirus 1)".
#:
#: `IE-1` appears on 31 VDJdb rows against `IE1`'s 14,183, so the two are folded together;
#: without that, one antigen is split in two.
CMV = SpeciesPreset(
    slug="cmv",
    vdjdb="CMV",
    mcpas="Cytomegalovirus (CMV)",
    iedb="Human herpesvirus 5",
    gene_aliases={"IE-1": "IE1"},
)


def load_cmv_rows(*, refresh: bool = False) -> pd.DataFrame:
    """Every CMV record from every source, one row each. See `combined.load_rows`."""
    return combined.load_rows(CMV, refresh=refresh)


def load_cmv_annotations(
    trb_key: tuple[str, ...] = DEFAULT_TRB_KEY, *, refresh: bool = False
) -> pd.DataFrame:
    """The long CMV table, one row per (TRB, antigen, database).

    Args:
        trb_key: the columns defining one TRB. A dataset carrying family-only V calls - Adaptive
            output typically does - wants `TRB_KEY_V_FAMILY`.
        refresh: rebuild even if a current cached table exists.

    See `combined.load_annotations`.
    """
    return combined.load_annotations(CMV, trb_key, refresh=refresh)


def load_cmv_trb_antigens(
    trb_key: tuple[str, ...] = DEFAULT_TRB_KEY, *, refresh: bool = False
) -> pd.DataFrame:
    """The flattened per-TRB CMV view, one row per (TRB, MHC class).

    See `combined.load_trb_antigens`.
    """
    return combined.load_trb_antigens(CMV, trb_key, refresh=refresh)
