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

    # Each repeated rung is now a mean over five scaffold-disjoint / random
    # partitions, so these moved when the bands arrived — and by less than the
    # bands themselves, which is the point. Previous single-partition values in
    # brackets. The 1-NN rungs are exact for a given partition set; the XGBoost
    # rungs carry ~±0.002 across xgboost builds.
    assert 0.704 <= v["reported"] <= 0.724, v["reported"]                    # 0.714 (was 0.709)
    assert 0.560 <= v["lookup_random"] <= 0.570, v["lookup_random"]          # 0.565 (was 0.572)
    assert 0.598 <= v["survives_scaffold"] <= 0.618, v["survives_scaffold"]  # 0.608 (was 0.597)
    assert 0.421 <= v["lookup_scaffold"] <= 0.431, v["lookup_scaffold"]      # 0.426 (was 0.451)
    assert -0.22 <= v["permutation_floor"] <= -0.18, v["permutation_floor"]  # -0.202

    # the two headline claims
    assert 78 <= v["lookup_pct_of_reported"] <= 80, v["lookup_pct_of_reported"]
    assert 0.162 <= v["learned_beyond_lookup"] <= 0.202, v["learned_beyond_lookup"]

    # the bands themselves — the audit must never claim a precision it did not
    # measure, so every repeated rung carries one and it is not zero
    for key in ("reported", "lookup_random", "survives_scaffold", "lookup_scaffold"):
        sd = v[key + "_sd"]
        assert sd is not None and sd > 0, f"{key} has no error band: {sd}"

    # Why the bands matter, on this dataset: the model's own contribution over
    # the lookup is 0.18 with a spread of ±0.04. The single-partition audit
    # read 0.146 and quoted it to three decimals. Both are the same finding;
    # only one of them says so.
    assert v["learned_beyond_lookup_sd"] > 0.02, v["learned_beyond_lookup_sd"]

    # the harder split lands below the scaffold split, as the recent
    # literature on scaffold-split optimism predicts
    assert v["survives_similarity"] < v["survives_scaffold"], (
        v["survives_similarity"], v["survives_scaffold"])

    print("✓ BACE regression: "
          f"reported={v['reported']}±{v['reported_sd']} "
          f"lookup={v['lookup_random']}±{v['lookup_random_sd']} "
          f"survives={v['survives_scaffold']}±{v['survives_scaffold_sd']} "
          f"nn_scaffold={v['lookup_scaffold']}±{v['lookup_scaffold_sd']} "
          f"similarity={v['survives_similarity']} floor={v['permutation_floor']} "
          f"lookup%={v['lookup_pct_of_reported']} "
          f"learned={v['learned_beyond_lookup']}±{v['learned_beyond_lookup_sd']}")


if __name__ == "__main__":
    test_bace_headline_numbers()
