"""
The scaffold split must be a function of the molecules — nothing else.

This file used to assert the opposite. `_scaffold_folds` delegated to
scikit-learn's `GroupKFold`, which orders series by size with

    indices = np.argsort(n_samples_per_group)[::-1]

an *unstable* sort, and 200 of BACE's 377 series hold a single compound.
Every tie-breaking was a valid "largest series first" order and each gave a
different partition into folds of identical size — so the same code on the
same data read `scaffold-lookup` 0.365 on one machine and 0.421 on another,
with identical library versions. Group ids came from `pd.factorize`, i.e.
order of first appearance, so shuffling the input rows renamed them too.
`seed` reached neither decision.

Both are now fixed by construction: ids from `sorted(set(scaffolds))`,
groups filled largest-first into the lightest fold with ties broken by group
id and by fold index. These tests assert that guarantee holds — identical
folds under any row ordering — rather than the defect it replaced.

Scope: this covers the *split*. The scores on top of it are not yet
order-invariant — `_nn_oof` breaks nearest-neighbour ties by position and
XGBoost subsamples against row positions. See the determinism note in the
README; `test_scores_still_read_row_order` pins what is left.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autopsy.engine import _scaffold, _scaffold_folds, _fingerprint, _nn_oof

BACE = Path(__file__).resolve().parent.parent / "bace.csv"
K = 5


@pytest.fixture(scope="module")
def bace():
    """SMILES, activity and scaffold, in file order. Scaffolds are a pure
    function of SMILES, so permuting these three together is exactly what a
    reordered input CSV looks like to `_scaffold_folds`."""
    df = pd.read_csv(BACE, usecols=["smiles", "pIC50"])
    smiles = df["smiles"].astype(str).values
    return smiles, df["pIC50"].astype(float).values, [_scaffold(s) for s in smiles]


def _membership(smiles, scaffolds, k=K):
    """Fold contents as sets of molecules — identity, not position."""
    folds, _ = _scaffold_folds(scaffolds, k)
    return {frozenset(smiles[test]) for _, test in folds}


def test_ties_are_pervasive(bace):
    """The precondition. Without ties, any sort would agree and none of this
    would matter; with 200 of them, the tie-break *is* the split."""
    _, _, scaffolds = bace
    counts = pd.Series(scaffolds).value_counts()
    assert len(counts) == 377
    assert int((counts == 1).sum()) == 200


def test_group_ids_come_from_scaffold_content(bace):
    """Ids must track sorted scaffold content, not order of appearance."""
    _, _, scaffolds = bace
    _, groups = _scaffold_folds(scaffolds, K)
    expected = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
    assert list(groups) == [expected[s] for s in scaffolds]


@pytest.mark.parametrize("seed", [1, 7, 1234])
def test_folds_survive_row_permutation(bace, seed):
    """The guarantee: reorder the CSV, get the same folds."""
    smiles, _, scaffolds = bace
    perm = np.random.default_rng(seed).permutation(len(smiles))

    base = _membership(smiles, scaffolds)
    shuffled = _membership(smiles[perm], [scaffolds[i] for i in perm])

    assert shuffled == base, "row order changed the scaffold partition"


def test_folds_are_balanced_and_disjoint(bace):
    """A group never straddles folds, and the greedy fill keeps them even."""
    smiles, _, scaffolds = bace
    folds, groups = _scaffold_folds(scaffolds, K)

    assert len(folds) == K
    test_idx = np.concatenate([te for _, te in folds])
    assert sorted(test_idx) == list(range(len(smiles))), "folds must partition the rows"

    for _, te in folds:                      # no series split across the boundary
        assert set(groups[te]).isdisjoint(set(np.delete(groups, te)))

    sizes = sorted(len(te) for _, te in folds)
    assert sizes[-1] - sizes[0] <= 0.02 * len(smiles), sizes


def test_scores_still_read_row_order(bace):
    """KNOWN GAP, deliberately pinned: the folds are order-invariant but the
    1-NN rung on top of them is not. `_nn_oof` resolves tied neighbours with
    `np.argmax`, which picks by position — 99 of 1513 test molecules have a
    tied nearest neighbour here and 83 of those tie across different
    activities. Break ties on a content key and this test should start
    failing; that is the signal to re-run the audit and drop this test."""
    smiles, y, scaffolds = bace
    perm = np.random.default_rng(1).permutation(len(smiles))

    def lookup(sm, sc, act):
        folds, _ = _scaffold_folds(sc, K)
        return _nn_oof([_fingerprint(s, 2048, 2)[0] for s in sm], act, folds)["r2"]

    base = lookup(smiles, scaffolds, y)
    shuffled = lookup(smiles[perm], [scaffolds[i] for i in perm], y[perm])

    assert base != shuffled, "if this fails, 1-NN tie-breaking is order-free now"
