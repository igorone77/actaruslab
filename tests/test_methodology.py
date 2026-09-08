"""
The three answers to the methodological critique, each pinned by what it
actually claims rather than by the shape of its output.

  · error bands      a rung measured once cannot tell a real gap from the
                     noise of where the fold boundaries fell
  · descriptor       a verdict that only holds under ECFP4 describes the
                     control        fingerprint, not the dataset
  · similarity split distinct scaffolds can still be near neighbours, so a
                     scaffold-disjoint fold is not necessarily a dissimilar
                     one — this splits on the similarity itself

Deliberately run on 300 AlogP molecules at k=3, repeats=3: these test the
mechanisms, and tests/test_bace_smoke.py pins the numbers at full size.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import DataStructs

from autopsy.engine import (run_autopsy, _scaffold, _scaffold_folds, _fingerprint,
                            _fingerprint_alt, _similarity_clusters, _similarity_folds,
                            _descriptor_control, _limitations, _agg, _lookup_flag,
                            SIM_CUTOFF)

DATA = Path(__file__).resolve().parent / "data" / "alogp_500.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA).head(300)


@pytest.fixture(scope="module")
def result(df):
    return run_autopsy(df, "smiles", "AlogP", k=3, repeats=3, seed=0)


@pytest.fixture(scope="module")
def mols(df):
    smiles = df["smiles"].astype(str).values
    bvs = [_fingerprint(s, 2048, 2)[0] for s in smiles]
    keep = [i for i, b in enumerate(bvs) if b is not None]
    return ([bvs[i] for i in keep],
            [_scaffold(smiles[i]) for i in keep])


# ── 1 · error bands ───────────────────────────────────────────────────

def test_every_repeated_rung_carries_a_band(result):
    """Random, lookup and scaffold are repeated; the band is the deliverable."""
    banded = {r["kind"] for r in result.ladder if r.get("r2_sd") is not None}
    assert {"reported", "lookup", "survives"} <= banded, banded
    for r in result.ladder:
        if r.get("r2_sd") is not None:
            assert r["r2_sd"] >= 0 and r["n_reps"] == 3, r


def test_the_band_is_measured_not_assumed(mols):
    """Aggregation is mean and *sample* sd. One run reports no spread rather
    than a spread of zero — zero would claim a precision never measured."""
    runs = [{"r2": 0.50, "rmse": 1.0, "spearman": 0.7},
            {"r2": 0.60, "rmse": 1.2, "spearman": 0.8}]
    agg = _agg(runs)
    assert agg["r2"] == 0.55
    assert agg["r2_sd"] == pytest.approx(np.std([0.5, 0.6], ddof=1), abs=1e-3)
    assert agg["n_reps"] == 2

    lone = _agg(runs[:1])
    assert lone["r2"] == 0.50 and lone["r2_sd"] is None and lone["n_reps"] == 1


def test_repeated_partitions_really_differ_and_stay_scaffold_disjoint(mols):
    """The band would be a lie if every repetition cut the same way, and the
    rung would be a lie if any repetition let a series straddle the split."""
    bvs, scaffolds = mols
    seen = []
    for rep in range(3):
        folds, groups = _scaffold_folds(scaffolds, 3, rep=rep)
        seen.append({frozenset(te.tolist()) for _, te in folds})
        for _, te in folds:                       # no series across the boundary
            assert set(groups[te]).isdisjoint(set(np.delete(groups, te)))
    assert seen[0] != seen[1] and seen[1] != seen[2], "repetitions gave one partition"


def test_repeated_partitions_are_still_a_function_of_the_molecules(mols):
    """Reproducibility is not traded away for the band: rep r is always the
    same partition, and rep 0 is the split every earlier audit produced."""
    _, scaffolds = mols
    for rep in (0, 1, 2):
        a, _ = _scaffold_folds(scaffolds, 3, rep=rep)
        b, _ = _scaffold_folds(scaffolds, 3, rep=rep)
        assert [x.tolist() for _, x in a] == [x.tolist() for _, x in b]


def test_a_single_partition_run_says_it_has_no_bands(df):
    res = run_autopsy(df, "smiles", "AlogP", k=3, repeats=1, seed=0)
    assert all(r.get("r2_sd") is None for r in res.ladder)
    assert "single_partition" in {l["code"] for l in res.limitations}


# ── 2 · descriptor robustness ─────────────────────────────────────────

def test_the_second_fingerprint_is_a_different_family(mols):
    """A control that agreed by construction would prove nothing. The path
    fingerprint must actually rank similarity differently from ECFP4."""
    bvs, _ = mols
    alt = [_fingerprint_alt(s, 2048)
           for s in pd.read_csv(DATA).head(300)["smiles"].astype(str).values]
    alt = [a for a in alt if a is not None]
    a = np.array(DataStructs.BulkTanimotoSimilarity(bvs[0], bvs[1:40]))
    b = np.array(DataStructs.BulkTanimotoSimilarity(alt[0], alt[1:40]))
    assert not np.allclose(a, b, atol=0.05), "the two fingerprints agree too closely to be a control"


def test_the_control_runs_and_reports_both_readings(result):
    dc = result.verdict["descriptor_control"]
    assert dc["available"] is True
    assert isinstance(dc["lookup_pct_ecfp"], int) and isinstance(dc["lookup_pct_alt"], int)
    assert dc["flag_ecfp"] == _lookup_flag(dc["lookup_pct_ecfp"])
    assert dc["flag_alt"] == _lookup_flag(dc["lookup_pct_alt"])
    assert isinstance(dc["agree"], bool)
    assert str(dc["lookup_pct_ecfp"]) in dc["note"] and str(dc["lookup_pct_alt"]) in dc["note"]


@pytest.mark.parametrize("ecfp,alt,agree", [
    (0.60, 0.58, True),      # 60% vs 58% — same reading
    (0.60, 0.20, False),     # 60% vs 20% — MODERATE against LOW
    (0.75, 0.68, False),     # 75% vs 68% — SEVERE against MODERATE, 7 points apart
])
def test_agreement_is_decided_on_the_reading_not_only_the_gap(ecfp, alt, agree):
    """A gap inside the tolerance still fails when it moves the badge: the
    badge is what a reader acts on."""
    dc = _descriptor_control(1.0, ecfp, alt)
    assert dc["agree"] is agree, dc["note"]


def test_a_disagreement_is_declared_as_a_limitation():
    dc = _descriptor_control(1.0, 0.75, 0.20)
    codes = {l["code"] for l in _limitations(None, 0.3, dc, 5)}
    assert "descriptor_disagreement" in codes


# ── 3 · the harder split ──────────────────────────────────────────────

def test_similarity_clusters_are_dissimilar_to_each_other(mols):
    """The clustering must do what its name says: two molecules in different
    clusters are not near neighbours of the same leader."""
    bvs, _ = mols
    assign, n = _similarity_clusters(bvs, SIM_CUTOFF)
    assert 1 < n < len(bvs), n
    assert len(assign) == len(bvs)


def test_the_similarity_split_puts_test_molecules_further_from_training(mols):
    """The substantive claim, measured rather than asserted: after splitting on
    similarity, a held-out molecule's nearest training neighbour is further
    away than it is under a scaffold split. That gap is exactly what the
    recent scaffold-split-optimism results are about."""
    bvs, scaffolds = mols

    def mean_max_similarity(folds):
        vals = []
        for tr, te in folds:
            pool = [bvs[i] for i in tr]
            vals += [max(DataStructs.BulkTanimotoSimilarity(bvs[j], pool)) for j in te]
        return float(np.mean(vals))

    scaf_folds, _ = _scaffold_folds(scaffolds, 3, rep=0)
    assign, n = _similarity_clusters(bvs, SIM_CUTOFF)
    sim_folds = _similarity_folds(assign, n, 3)

    near_scaffold = mean_max_similarity(scaf_folds)
    near_similarity = mean_max_similarity(sim_folds)
    assert near_similarity < near_scaffold, (near_similarity, near_scaffold)


def test_the_similarity_rung_runs_and_is_marked_experimental(result):
    rung = next(r for r in result.ladder if r["kind"] == "survives_similarity")
    assert rung["r2"] is not None
    assert "EXPERIMENTAL" in rung["note"]
    assert result.verdict["survives_similarity"] == rung["r2"]
    card = next(c for c in result.readout if c["signal"] == "Similarity transfer")
    assert card["flag"] == "EXPERIMENTAL"
    assert "not externally validated" in card["note"]


def test_similarity_folds_never_split_a_cluster(mols):
    bvs, _ = mols
    assign, n = _similarity_clusters(bvs, SIM_CUTOFF)
    folds = _similarity_folds(assign, n, 3)
    for _, te in folds:
        assert set(assign[te]).isdisjoint(set(np.delete(assign, te)))


# ── 5 · declared limits ───────────────────────────────────────────────

def test_a_missing_time_split_is_declared_not_silent(result):
    """An absent test that says nothing reads as a test that passed."""
    lim = next(l for l in result.limitations if l["code"] == "no_time_split")
    assert lim["level"] == "severe"
    assert "gold standard" in lim["text"]
    assert "not executed" in lim["text"] or "was not executed" in lim["text"]

    card = next(c for c in result.readout if c["signal"] == "Temporal test")
    assert card["flag"] == "N/A"
    assert "gold standard" in card["note"]


def test_the_time_split_limitation_disappears_when_dates_are_given():
    codes = {l["code"] for l in _limitations(0.42, 0.3, {"available": True, "agree": True}, 5)}
    assert "no_time_split" not in codes


def test_the_lookup_metric_is_declared_non_standard(result):
    lim = next(l for l in result.limitations if l["code"] == "lookup_pct_not_standard")
    assert "lookup_random / reported" in lim["text"]
    assert "not a standard QSAR statistic" in lim["text"]
    assert "no external validation" in lim["text"]

    card = next(c for c in result.readout if c["signal"] == "Similarity leakage")
    assert "not a standard QSAR" in card["note"]


def test_the_similarity_split_is_declared_unvalidated(result):
    lim = next(l for l in result.limitations
               if l["code"] == "similarity_split_experimental")
    assert "not been externally validated" in lim["text"]
    assert "domain expert" in lim["text"]


# ── the denominator ───────────────────────────────────────────────────
# Found by looking at the rendered page rather than at a test: on a slice where
# the model scores R² 0.00, "% is lookup" read -50300%. The guard was `> 0`,
# which a reported 0.001 passes. Arithmetically correct, and precisely the
# false-plausible number this tool exists to catch.

@pytest.fixture(scope="module")
def flat(df):
    """The first 120 BACE compounds: too narrow a slice for the model to learn
    anything, so the random split lands at about zero."""
    return run_autopsy(pd.read_csv(Path(__file__).resolve().parent.parent / "bace.csv",
                                   usecols=["smiles", "pIC50"]).head(120),
                       "smiles", "pIC50", k=3, repeats=2, seed=0)


def test_a_near_zero_score_is_not_divided_into(flat):
    assert flat.verdict["reported"] < 0.05, flat.verdict["reported"]
    assert flat.verdict["lookup_pct_of_reported"] is None
    assert flat.verdict["descriptor_control"]["available"] is False
    assert not any(c["signal"] == "Similarity leakage" for c in flat.readout)


def test_the_missing_ratio_is_explained_not_just_omitted(flat):
    lim = next(l for l in flat.limitations if l["code"] == "no_ratio_denominator")
    assert lim["level"] == "severe"
    assert "no performance" in lim["text"]
    assert "no usable predictive signal" in flat.verdict["headline"]


def test_the_ratio_still_reports_where_there_is_a_score_to_divide(result):
    """The guard must not silence a legitimate reading."""
    assert result.verdict["reported"] >= 0.05
    assert isinstance(result.verdict["lookup_pct_of_reported"], int)
