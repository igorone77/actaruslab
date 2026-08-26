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

    # The split is deterministic now, so these are tight bands around measured
    # values rather than the wide ones the nondeterministic split needed.
    # The 1-NN rungs are exact for a given CSV; the XGBoost rungs carry ~±0.002
    # across xgboost builds, hence the slightly wider window on those.
    assert 0.699 <= v["reported"] <= 0.719, v["reported"]                    # 0.709
    assert 0.567 <= v["lookup_random"] <= 0.577, v["lookup_random"]          # 0.572
    assert 0.587 <= v["survives_scaffold"] <= 0.607, v["survives_scaffold"]  # 0.597
    assert 0.446 <= v["lookup_scaffold"] <= 0.456, v["lookup_scaffold"]      # 0.451
    assert -0.24 <= v["permutation_floor"] <= -0.20, v["permutation_floor"]  # -0.222

    # the two headline claims
    assert 80 <= v["lookup_pct_of_reported"] <= 82, v["lookup_pct_of_reported"]
    assert 0.136 <= v["learned_beyond_lookup"] <= 0.156, v["learned_beyond_lookup"]

    print("✓ BACE regression: "
          f"reported={v['reported']} lookup={v['lookup_random']} "
          f"survives={v['survives_scaffold']} nn_scaffold={v['lookup_scaffold']} "
          f"floor={v['permutation_floor']} "
          f"lookup%={v['lookup_pct_of_reported']} learned={v['learned_beyond_lookup']}")


if __name__ == "__main__":
    test_bace_headline_numbers()
