"""The CMV species preset: the three spellings, and the traps that fix each match mode."""
import pandas as pd
from portmanteau.tcr.publicdb import iedb, mcpas, vdjdb

from cmvividly.data.publicdb import CMV


def test_the_preset_names_cmv_in_all_three_databases():
    assert CMV.slug == "cmv"
    assert (CMV.vdjdb, CMV.mcpas, CMV.iedb) == (
        "CMV", "Cytomegalovirus (CMV)", "Human herpesvirus 5")


def test_the_vdjdb_gene_alias_folds_ie1():
    """IE-1 is on 31 VDJdb rows against IE1's 14,183; unfolded, one antigen becomes two."""
    pdf = pd.DataFrame({"antigen.species": ["CMV", "CMV"], "antigen.gene": ["IE-1", "IE1"]})
    out = vdjdb.filter_species(pdf, CMV)
    assert set(out["antigen.gene"]) == {"IE1"}


def test_mcpas_matches_the_pathology_exactly_and_so_excludes_lcmv():
    """⛔ A `cmv|cytomegalo` pattern catches LCMV, which is a different virus."""
    pdf = pd.DataFrame({"Pathology": ["Cytomegalovirus (CMV)",
                                      "Lymphocytic choriomeningitis virus (LCMV)"]})
    out = mcpas.filter_species(pdf, CMV)
    assert len(out) == 1
    assert out.loc[0, "Pathology"] == "Cytomegalovirus (CMV)"


def test_iedb_matches_a_prefix_so_strain_detail_is_kept_and_murine_cmv_is_not():
    """⚠️ Exact would drop the strain-qualified rows; "cytomegalo" would catch the murid virus."""
    pdf = pd.DataFrame({"antigen.species": [
        "Human herpesvirus 5",
        "Human herpesvirus 5 (Human cytomegalovirus)",
        "Murid betaherpesvirus 1 (Mouse cytomegalovirus 1)",
        "Influenza A virus",
    ]})
    out = iedb.filter_species(pdf, CMV)
    assert len(out) == 2
    assert all(v.startswith("Human herpesvirus 5") for v in out["antigen.species"])


def test_the_preset_slug_keys_the_cache_so_another_species_cannot_collide():
    """Derived filenames carry the slug; two species sharing one would serve each other's rows."""
    assert vdjdb.TRB_FILENAME.format(species=CMV.slug, slug="v_cdr3") != \
        vdjdb.TRB_FILENAME.format(species="ebv", slug="v_cdr3")
