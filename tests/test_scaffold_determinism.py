"""
KNOWN DEFECT — the scaffold split is not reproducible.

`_scaffold_folds` delegates to `sklearn.model_selection.GroupKFold`, which
orders scaffold series by size with

    indices = np.argsort(n_samples_per_group)[::-1]

an *unstable* sort. On BACE, 200 of the 377 series contain exactly one
compound, so that ordering is one arbitrary choice among many — and which
one you get depends on the scikit-learn version and on the CPU (numpy 2.x
picks a SIMD sort path at runtime from the host's instruction set).

Every tie-breaking is a legitimate "largest series first" order, and each
produces a different partition of the same series into the same-sized
folds. The scaffold rungs move with it. Measured on BACE-1, k=5, seed=0:

    tie order            scaffold   scaffold-lookup   learned beyond lookup
    ascending group id     0.611          0.414               0.197
    descending group id    0.633          0.461               0.172
    this box               0.622          0.365               0.257  <- published
    GitHub Actions         —              0.421                 —

So `learned beyond lookup` — the number this tool exists to report — lands
anywhere in ~0.17–0.26 on identical code and identical data. The engine
docstring's promise ("Deterministic: fixed seeds, so an audit is
reproducible") does not hold for the scaffold rungs: `seed` never reaches
this decision.

These tests assert the defect, not specific scores, so they pass on any
machine. When the split is made deterministic in-engine, the first test
starts failing — that is the signal to re-run the BACE audit, refresh
bace_report.html, and update the numbers in the README.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autopsy.engine import _scaffold, _fingerprint, _nn_oof

BACE = Path(__file__).resolve().parent.parent / "bace.csv"
K = 5


@pytest.fixture(scope="module")
def bace():
    df = pd.read_csv(BACE, usecols=["smiles", "pIC50"])
    smiles = df["smiles"].astype(str).values
    groups = pd.factorize(pd.Series([_scaffold(s) for s in smiles]))[0]
    return smiles, df["pIC50"].astype(float).values, groups


def _greedy_folds(groups, order, k=K):
    """GroupKFold's own algorithm — largest series first into the lightest
    fold — but with the group order handed in explicitly."""
    counts = np.bincount(groups)
    weight = np.zeros(k)
    group_to_fold = np.zeros(len(counts), dtype=int)
    for g in order:
        lightest = int(np.argmin(weight))
        weight[lightest] += counts[g]
        group_to_fold[g] = lightest
    per_sample = group_to_fold[groups]
    return [(np.where(per_sample != f)[0], np.where(per_sample == f)[0]) for f in range(k)]


def _by_size_desc(groups, tie):
    """A valid 'largest series first' order; `tie` picks the tie-breaking."""
    counts = np.bincount(groups)
    ids = np.arange(len(counts))
    return np.lexsort((ids if tie == "asc" else -ids, -counts))


def test_ties_are_pervasive(bace):
    """The precondition: without ties the sort order would be forced."""
    _, _, groups = bace
    counts = np.bincount(groups)
    assert len(counts) == 377
    assert (counts == 1).sum() == 200, "200 singleton series all tie at weight 1"


def test_tie_order_changes_the_partition(bace):
    """Two equally valid orderings, two different splits of the same series."""
    _, _, groups = bace
    asc = _greedy_folds(groups, _by_size_desc(groups, "asc"))
    desc = _greedy_folds(groups, _by_size_desc(groups, "desc"))

    assert [len(te) for _, te in asc] == [len(te) for _, te in desc], \
        "fold sizes match — only the assignment differs, which is what hides this"

    moved = sum(len(set(a_te) ^ set(d_te)) for (_, a_te), (_, d_te) in zip(asc, desc))
    assert moved > 0, "if this fails the split is now deterministic — see module docstring"


def test_tie_order_moves_the_reported_number(bace):
    """And the split it picks moves the headline the audit reports."""
    smiles, y, groups = bace
    bvs = [_fingerprint(s, 2048, 2)[0] for s in smiles]

    asc = _nn_oof(bvs, y, _greedy_folds(groups, _by_size_desc(groups, "asc")))["r2"]
    desc = _nn_oof(bvs, y, _greedy_folds(groups, _by_size_desc(groups, "desc")))["r2"]

    assert abs(asc - desc) > 0.02, (
        f"scaffold-lookup: {asc} vs {desc} — a tie-breaking choice, not chemistry")
