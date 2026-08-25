"""
Regression guard. Runs the full BACE-1 audit and asserts the headline
numbers land where they should, so a future change to featurisation,
splitting, or the model can't silently move them.

    python -m tests.test_bace_smoke      (or: pytest tests/)
"""
from pathlib import Path

import pandas as pd
from autopsy.engine import run_autopsy

BACE = Path(__file__).resolve().parent.parent / "bace.csv"


def test_bace_headline_numbers():
    df = pd.read_csv(BACE)
    res = run_autopsy(df, "smiles", "pIC50", k=5, seed=0)
    v, s = res.verdict, res.specimen

    # specimen composition is fixed for this dataset
    assert s["n_compounds"] == 1513
    assert s["n_scaffold_series"] == 377
    assert s["n_singleton_series"] == 200

    # the ladder must descend in the expected way (tolerances, not exact floats)
    assert 0.68 <= v["reported"] <= 0.75, v["reported"]
    assert 0.54 <= v["lookup_random"] <= 0.61, v["lookup_random"]
    assert 0.58 <= v["survives_scaffold"] <= 0.66, v["survives_scaffold"]
    # the scaffold-lookup floor pins the scaffold *partition*, not just the score:
    # it is the rung that moves if GroupKFold changes how it assigns series to folds
    # (see the scikit-learn pin in requirements.txt)
    assert 0.32 <= v["lookup_scaffold"] <= 0.41, v["lookup_scaffold"]
    assert v["permutation_floor"] < 0.05, v["permutation_floor"]      # floor collapses

    # the two headline claims
    assert 75 <= v["lookup_pct_of_reported"] <= 88, v["lookup_pct_of_reported"]
    assert 0.15 <= v["learned_beyond_lookup"] <= 0.35, v["learned_beyond_lookup"]

    print("✓ BACE regression: "
          f"reported={v['reported']} lookup={v['lookup_random']} "
          f"survives={v['survives_scaffold']} nn_scaffold={v['lookup_scaffold']} "
          f"floor={v['permutation_floor']} "
          f"lookup%={v['lookup_pct_of_reported']} learned={v['learned_beyond_lookup']}")


if __name__ == "__main__":
    test_bace_headline_numbers()
