"""
Seven malformed CSVs, from a robustness pass on the app.

The science underneath held — but the handling around it had three faults,
and each of these cases pins one of them shut:

  * nothing may reach the caller as an unhandled 500. A constant activity
    column did not even raise: it produced a perfect-looking audit, R² 1.00
    on every rung including the permutation control, and only broke when the
    NaN correlation hit a JSON serialiser that refuses NaN. The CLI printed
    that audit in full. It is refused up front now.
  * rows the engine discards must be visible. Silently auditing 54 of 60
    rows lets the reader believe all 60 were covered.
  * a message must blame the right thing. Text in the activity column used
    to be reported as "0 valid rows", sending the reader to inspect SMILES
    that were fine.

The bar for every message is the missing-column one, which was already
right: say what is wrong and what to do about it, in words.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from autopsy import api
from autopsy.engine import run_autopsy, AutopsyError

BACE = Path(__file__).resolve().parent.parent / "bace.csv"

# Raw jargon that must never reach whoever uploaded the file.
JARGON = ["Traceback", "ValueError", "KeyError", "TypeError", "IndexError",
          "NoneType", "dtype", "nan", "HTTP 500", "Exception", "None]"]


@pytest.fixture(scope="module")
def rows():
    return pd.read_csv(BACE, usecols=["smiles", "pIC50"]).head(120).reset_index(drop=True)


def _no_jargon(message: str):
    for bad in JARGON:
        assert bad not in message, f"{bad!r} leaked into: {message}"
    assert message[0].islower() or message[0].isupper()      # a sentence, not a code
    assert len(message.split()) >= 6, f"too terse to act on: {message}"


# ── cases that must be refused, with a message that names the real cause ──

def test_2_text_activity_blames_the_column_not_the_rows(rows):
    df = rows.assign(pIC50=["alto", "basso", "medio"] * 40)
    with pytest.raises(AutopsyError) as e:
        run_autopsy(df, "smiles", "pIC50")
    msg = str(e.value)
    assert "activity column 'pIC50'" in msg
    assert "no numeric values" in msg
    assert "alto" in msg                       # shows what it found instead
    assert "valid rows" not in msg             # the old, misleading phrasing
    _no_jargon(msg)


def test_3_missing_column_says_what_is_available(rows):
    with pytest.raises(AutopsyError) as e:
        run_autopsy(rows.rename(columns={"pIC50": "altro"}), "smiles", "pIC50")
    msg = str(e.value)
    assert "'pIC50' not found" in msg and "altro" in msg
    _no_jargon(msg)


def test_4_too_few_rows_counts_them(rows):
    with pytest.raises(AutopsyError) as e:
        run_autopsy(rows.head(20), "smiles", "pIC50")
    msg = str(e.value)
    assert "20 of 20" in msg and "at least 40" in msg
    _no_jargon(msg)


def test_5_constant_activity_is_refused_not_scored_at_1(rows):
    """The dangerous one: it used to return R² 1.00 on every rung."""
    with pytest.raises(AutopsyError) as e:
        run_autopsy(rows.assign(pIC50=7.0), "smiles", "pIC50")
    msg = str(e.value)
    assert "one repeated value" in msg and "zero variance" in msg
    _no_jargon(msg)


def test_5b_near_constant_activity_is_refused_too(rows):
    y = np.full(len(rows), 7.0)
    y[0] = 7.0 + 1e-9
    with pytest.raises(AutopsyError) as e:
        run_autopsy(rows.assign(pIC50=y), "smiles", "pIC50")
    assert "barely varies" in str(e.value)


def test_7_empty_file_says_so(rows):
    with pytest.raises(AutopsyError) as e:
        run_autopsy(rows.iloc[0:0], "smiles", "pIC50")
    _no_jargon(str(e.value))
    assert "no rows" in str(e.value)


def test_1b_broken_smiles_below_the_minimum_names_that_cause(rows):
    """60 rows with 30 unreadable leaves 30 usable — under the 40 an audit
    needs. The old code checked the minimum *before* parsing SMILES, so it
    audited 30 compounds while claiming to require 40."""
    df = rows.head(60).copy()
    df.loc[::2, "smiles"] = "XYZ123"
    with pytest.raises(AutopsyError) as e:
        run_autopsy(df, "smiles", "pIC50")
    msg = str(e.value)
    assert "30 of 60" in msg and "unreadable SMILES" in msg
    _no_jargon(msg)


# ── cases that must run, and say what they threw away ──────────────────

def test_1_broken_smiles_runs_and_warns_loudly(rows):
    """120 rows, 40 unreadable: 80 remain, enough to audit — and a third of
    the file is gone, which the reader has to be told."""
    df = rows.copy()
    df.loc[::3, "smiles"] = "XYZ123"
    res = run_autopsy(df, "smiles", "pIC50", k=5, seed=0)

    assert res.specimen["n_compounds"] == 80
    assert res.specimen["n_unparseable_dropped"] == 40
    assert res.specimen["n_rows_in"] == 120
    assert res.warnings, "40 rows vanished without a word"

    w = res.warnings[0]
    assert w["level"] == "severe", "a third of the file is not a footnote"
    assert "40 of 120 rows dropped" in w["text"]
    assert "unreadable SMILES (40)" in w["text"]
    assert "may not represent your dataset" in w["text"]


def test_6_missing_activity_runs_and_warns(rows):
    df = rows.copy()
    df.loc[::10, "pIC50"] = np.nan
    res = run_autopsy(df, "smiles", "pIC50", k=5, seed=0)

    assert res.specimen["n_activity_dropped"] == 12
    assert res.specimen["n_compounds"] == 108
    w = res.warnings[0]
    assert w["level"] == "note", "10% dropped is a note, not an alarm"
    assert "missing or non-numeric activity (12)" in w["text"]


def test_empty_smiles_do_not_enter_as_phantom_molecules(rows):
    """An empty SMILES parses into a molecule with no atoms, whose all-zero
    fingerprint would otherwise join the audit as a compound."""
    df = rows.copy()
    df.loc[::4, "smiles"] = ""
    res = run_autopsy(df, "smiles", "pIC50", k=5, seed=0)
    assert res.specimen["n_unparseable_dropped"] == 30
    assert res.specimen["n_compounds"] == 90


def test_a_clean_file_carries_no_warning(rows):
    res = run_autopsy(rows, "smiles", "pIC50", k=5, seed=0)
    assert res.warnings == []
    assert res.specimen["n_dropped_total"] == 0


# ── the endpoints: refusals are 422, never 500 ─────────────────────────

@pytest.mark.parametrize("name,mutate", [
    ("text-activity",     lambda d: d.assign(pIC50=["alto", "basso", "medio"] * 40)),
    ("missing-column",    lambda d: d.rename(columns={"pIC50": "altro"})),
    ("too-few-rows",      lambda d: d.head(20)),
    ("constant-activity", lambda d: d.assign(pIC50=7.0)),
    ("empty-file",        lambda d: d.iloc[0:0]),
])
def test_endpoint_refuses_with_422_and_a_readable_message(rows, name, mutate):
    with pytest.raises(HTTPException) as e:
        api._run(mutate(rows), "smiles", "pIC50", None, 5)
    assert e.value.status_code == 422, f"{name} came back as {e.value.status_code}"
    _no_jargon(e.value.detail)


def test_results_that_run_are_json_serialisable(rows):
    """Starlette serialises with allow_nan=False, so a NaN metric used to
    surface as an unexplained 500. No rung may emit one."""
    import json
    df = rows.copy()
    df.loc[::3, "smiles"] = "XYZ123"
    out = api._run(df, "smiles", "pIC50", None, 5)
    json.dumps(out, allow_nan=False)
    assert out["warnings"], "the API result must carry the drop notice too"


def test_refusals_land_at_submit_not_as_a_failed_job(rows):
    """A constant activity column is visible for free, so the caller should be
    told in milliseconds rather than after queueing an audit."""
    with pytest.raises(HTTPException) as e:
        api._precheck(rows.assign(pIC50=7.0), "smiles", "pIC50", None)
    assert e.value.status_code == 422
    assert "zero variance" in e.value.detail
    _no_jargon(e.value.detail)
