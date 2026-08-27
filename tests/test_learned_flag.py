"""
The LEARNED STRUCTURE badge has to tell the truth on inputs whose answer we
already know.

`learned = survives - lookup_scaffold` is only a measure of the model while
the lookup baseline is itself sane. When the baseline scores below zero on
new scaffolds — worse than predicting the mean — subtracting it inflates
`learned`, and a badge reading only the number calls that a triumph. Four
real datasets produced four different stories behind similar numbers, so the
flag reads two dimensions: how much was added, and whether the baseline it
was measured against still worked.
"""
from pathlib import Path

import pandas as pd
import pytest

from autopsy.engine import (run_autopsy, _readout, _learned_flag,
                            LEARNED_NET, LEARNED_THIN)

ALOGP = Path(__file__).resolve().parent / "data" / "alogp_500.csv"


def _card(**verdict):
    """The learned-structure card as _readout builds it."""
    d = {"reported": 0.7, "lookup_rand": 0.5, "lookup_pct": 70, "survives": 0.5,
         "nn_scaf": 0.3, "learned": 0.2, "temporal": None, "floor": -0.2}
    d.update(verdict)
    cards = _readout(d["reported"], d["lookup_rand"], d["lookup_pct"], d["survives"],
                     d["nn_scaf"], d["learned"], d["temporal"], d["floor"])
    return next(c for c in cards if c["signal"] == "Learned structure")


# The four measured cases, as the datasets actually reported them.
@pytest.mark.parametrize("name,learned,nn_scaf,expected", [
    ("lipophilicity", 0.55,  0.30, "NET"),        # sound baseline, large gain
    ("esol",          0.64, -0.40, "ARTIFACT"),   # gain is the baseline collapsing
    ("chembl233",     0.24,  0.25, "MARGINAL"),   # sound baseline, modest gain
    ("bace",          0.146, 0.451, "THIN"),      # sound baseline, little added
])
def test_flag_matches_the_known_answer(name, learned, nn_scaf, expected):
    assert _learned_flag(learned, nn_scaf) == expected, name
    assert _card(learned=learned, nn_scaf=nn_scaf)["flag"] == expected, name


def test_artifact_wins_however_large_the_gain():
    """A negative baseline can manufacture any `learned` at all; none of them
    is evidence about the model."""
    for learned in (0.3, 0.64, 1.2, 5.0):
        assert _learned_flag(learned, -0.4) == "ARTIFACT"


def test_artifact_note_names_the_cause_and_redirects():
    card = _card(learned=0.64, nn_scaf=-0.40, survives=0.24)
    note = card["note"]
    assert "-0.40" in note, note                       # the baseline's actual score
    assert "worse than predicting the mean" in note
    assert "not learning a lot" in note                # says it plainly
    assert "0.24" in note                              # points at what to read instead


def test_cuts_are_the_documented_ones():
    """The boundaries are inclusive where the comment in engine.py says so."""
    assert _learned_flag(LEARNED_NET, 0.3) == "NET"
    assert _learned_flag(LEARNED_NET - 0.001, 0.3) == "MARGINAL"
    assert _learned_flag(LEARNED_THIN + 0.001, 0.3) == "MARGINAL"
    assert _learned_flag(LEARNED_THIN, 0.3) == "THIN"


def test_a_lookup_that_beats_the_model_reads_as_such():
    card = _card(learned=-0.15, nn_scaf=0.4)
    assert card["flag"] == "THIN"
    assert "the lookup baseline beats the model" in card["note"]


def test_real_dataset_with_a_collapsed_baseline_reads_artifact():
    """alogp_500 is the ESOL pattern in the repo: the lookup goes negative on
    new scaffolds, so `learned` reads high for the wrong reason."""
    res = run_autopsy(pd.read_csv(ALOGP), "smiles", "AlogP", k=5, seed=0)
    v = res.verdict

    assert v["lookup_scaffold"] < 0, v["lookup_scaffold"]
    assert v["learned_beyond_lookup"] > LEARNED_NET, v["learned_beyond_lookup"]

    card = next(c for c in res.readout if c["signal"] == "Learned structure")
    assert card["flag"] == "ARTIFACT", "a large gain over a collapsed baseline is not NET"
    assert "inflated" in v["headline"], v["headline"]


def test_real_dataset_with_a_sound_baseline_does_not_read_artifact():
    """BACE's baseline holds up, so its thin gain is reported as thin."""
    res = run_autopsy(pd.read_csv(Path(__file__).resolve().parent.parent / "bace.csv"),
                      "smiles", "pIC50", k=5, seed=0)
    assert res.verdict["lookup_scaffold"] > 0
    card = next(c for c in res.readout if c["signal"] == "Learned structure")
    assert card["flag"] == "THIN"
