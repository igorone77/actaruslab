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

The scores on top of the split used to read row order too — `_nn_oof` broke
nearest-neighbour ties with `np.argmax`, and XGBoost's subsample and
colsample_bytree draw against row positions. Both are closed: the engine puts
rows in canonical SMILES order before anything reads a position, and the
lookup baseline averages its ties. `test_verdict_survives_row_permutation`
asserts the whole verdict now, on two datasets.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autopsy.engine import run_autopsy, _scaffold, _scaffold_folds, _fingerprint, _nn_oof

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


@pytest.mark.parametrize("csv,smiles_col,y_col", [
    (BACE, "smiles", "pIC50"),
    (Path(__file__).resolve().parent / "data" / "alogp_500.csv", "smiles", "AlogP"),
], ids=["bace-1513", "alogp-500"])
def test_verdict_survives_row_permutation(csv, smiles_col, y_col):
    """The guarantee this engine sells: shuffle the input file and every
    headline number comes back identical — not just the folds.

    Two datasets, because one proves the property held once. BACE is the
    reference audit; alogp_500 is a different target on 500 of its molecules,
    where the lookup baseline collapses on new scaffolds and the model really
    does learn beyond it. Opposite verdicts, same invariance.
    """
    # repeats=2 rather than the default 5: rep 0 is the size-ordered
    # deterministic partition and rep 1 is a seeded permutation, so both code
    # paths that build a partition are exercised. Reps 2-4 are rep 1 with a
    # different seed and would add three full audits per dataset to CI for no
    # additional guarantee.
    df = pd.read_csv(csv)
    base = run_autopsy(df, smiles_col, y_col, k=5, seed=0, repeats=2).verdict
    shuffled = run_autopsy(df.sample(frac=1.0, random_state=1).reset_index(drop=True),
                           smiles_col, y_col, k=5, seed=0, repeats=2).verdict

    for key in ("reported", "lookup_random", "survives_scaffold", "lookup_scaffold",
                "learned_beyond_lookup", "permutation_floor", "lookup_pct_of_reported",
                "reported_sd", "survives_scaffold_sd", "survives_similarity"):
        assert shuffled[key] == base[key], f"{key}: {base[key]} -> {shuffled[key]}"


@pytest.mark.filterwarnings("ignore:An input array is constant")
def test_lookup_averages_its_tied_neighbours():
    """The baseline takes the mean of every neighbour at maximum similarity,
    not whichever one came first.

    Two training molecules are the same structure carrying different
    activities, so anything is equidistant from both. Averaging predicts 2.0
    for all three test molecules — residuals 0, 3, 6, so rmse sqrt(15) =
    3.873. `np.argmax` would have kept the first neighbour's 1.0, for
    residuals 1, 4, 7 and rmse sqrt(22) = 4.690.
    """
    fp = lambda smi: _fingerprint(smi, 2048, 2)[0]
    bvs = [fp("CCO"), fp("CCO"), fp("CCCO"), fp("CCCCO"), fp("CCCCCO")]
    y = np.array([1.0, 3.0, 2.0, 5.0, 8.0])
    folds = [(np.array([0, 1]), np.array([2, 3, 4]))]

    assert _nn_oof(bvs, y, folds)["rmse"] == pytest.approx(3.873, abs=1e-3)
