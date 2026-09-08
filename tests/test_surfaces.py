"""
Surface smoke tests: the two things wrapped around the engine — the HTML
report and the HTTP API — must keep consuming an AutopsyResult without
drifting from its shape.

Deliberately cheap: a 300-compound slice of BACE, k=3. These guard the
serialisers, not the numbers; tests/test_bace_smoke.py guards the numbers.
"""
import json
import re
import time
import uuid
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

from autopsy.engine import run_autopsy, AutopsyError
from autopsy.report import render_html
from autopsy import api
from autopsy.api import health, autopsy_records, RecordsRequest, Record
from autopsy.cli import main

BACE = Path(__file__).resolve().parent.parent / "bace.csv"


@pytest.fixture()
def subscriber(tmp_path, monkeypatch):
    """A real paid subscriber on a throwaway database.

    Holding a live subscription is what buys the reserved tier — the audit as
    the engine built it. These tests are about that full shape, not about the
    paywall (tests/test_billing.py) or the split (tests/test_tiers.py)."""
    import importlib, time
    monkeypatch.setenv("AUTOPSY_DB", str(tmp_path / "subs.db"))
    import autopsy.billing as b
    importlib.reload(b)
    b.init_db()
    key = b.create_subscriber("t@t.t", "cus_t", "sub_t", "active", time.time() + 86400)
    return b.by_key(key)


@pytest.fixture(scope="module")
def slice_df():
    return pd.read_csv(BACE, usecols=["smiles", "pIC50"]).head(300)


@pytest.fixture(scope="module")
def result(slice_df):
    return run_autopsy(slice_df, "smiles", "pIC50", k=3, seed=0)


def test_report_renders_standalone_html(result):
    html = render_html(result, source="slice.csv")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "</html>" in html
    assert "<svg" in html                      # the ladder plot
    assert "slice.csv" in html                 # provenance line
    # self-contained: the web fonts are the only permitted external reference
    assert {u for u in re.findall(r'https?://[^"\' <>]+', html)
            if "fonts.googleapis.com" not in u} == set()
    for rung in result.ladder:
        assert rung["model"] in html


def test_api_health():
    assert health()["status"] == "ok"


def test_api_records_returns_full_result(slice_df, subscriber):
    req = RecordsRequest(
        records=[Record(smiles=s, y=float(v))
                 for s, v in slice_df[["smiles", "pIC50"]].itertuples(index=False)],
        k=3,
    )
    out = autopsy_records(req, subscriber)
    assert set(out) == {"tier", "specimen", "ladder", "verdict", "readout",
                        "warnings", "meta", "limitations"}
    assert out["tier"] == "full"
    assert out["specimen"]["n_compounds"] == 300
    assert out["verdict"]["reported"] is not None


def test_engine_rejects_bad_column(slice_df):
    with pytest.raises(AutopsyError, match="not found"):
        run_autopsy(slice_df, "smiles", "no_such_column")


def test_engine_rejects_tiny_dataset(slice_df):
    """The message counts the rows and names the floor — see
    tests/test_malformed_input.py for the full set of refusal messages."""
    with pytest.raises(AutopsyError, match="at least 40"):
        run_autopsy(slice_df.head(10), "smiles", "pIC50")


def test_cli_writes_report_and_json(tmp_path, slice_df):
    csv = tmp_path / "slice.csv"
    slice_df.to_csv(csv, index=False)
    out_html, out_json = tmp_path / "r.html", tmp_path / "r.json"

    rc = main([str(csv), "--smiles", "smiles", "--y", "pIC50", "--k", "3",
               "--quiet", "--out", str(out_html), "--json", str(out_json)])

    assert rc == 0
    assert out_html.read_text().lstrip().startswith("<!DOCTYPE html>")
    data = json.loads(out_json.read_text())
    assert set(data) == {"specimen", "ladder", "verdict", "readout", "warnings",
                         "meta", "limitations"}
    assert data["verdict"]["reported"] is not None


def test_cli_reports_bad_column_as_exit_1(tmp_path, slice_df):
    csv = tmp_path / "slice.csv"
    slice_df.to_csv(csv, index=False)
    assert main([str(csv), "--smiles", "smiles", "--y", "nope", "--quiet"]) == 1


def test_cors_is_closed_by_default():
    """A public deployment must not answer cross-site calls unless asked to.
    The UI this app serves is same-origin, so the middleware should be absent
    entirely when AUTOPSY_ALLOWED_ORIGINS is unset."""
    names = [m.cls.__name__ for m in api.app.user_middleware]
    assert "CORSMiddleware" not in names, names


def test_api_rejects_oversized_dataset(monkeypatch, slice_df, subscriber):
    """The lookup rung is O(n²), so the service caps what it will accept
    rather than holding a worker open indefinitely."""
    monkeypatch.setattr(api, "MAX_ROWS", 100)
    req = RecordsRequest(
        records=[Record(smiles=s, y=float(v))
                 for s, v in slice_df[["smiles", "pIC50"]].itertuples(index=False)],
        k=3,
    )
    with pytest.raises(HTTPException) as exc:
        autopsy_records(req, subscriber)
    assert exc.value.status_code == 413
    assert "300 rows" in exc.value.detail


# ── jobs ──────────────────────────────────────────────────────────────
# The audit is too slow to answer inside a request, so it runs off it. These
# drive the worker directly — no threads, no live server — since what needs
# pinning is the state machine, not the executor.

def _queue(rows: int):
    """Returns (job_id, job_token). The token is the claim on the result —
    api.submit_job mints one for every job, so a test that fabricates a job
    without one would be testing a state the service never produces."""
    jid = uuid.uuid4().hex
    token = api.issue_job_token()
    api._jobs[jid] = {"status": "queued", "progress": "queued", "result": None,
                      "error": None, "created": time.time(), "started": None,
                      "finished": None, "rows": rows, "key_hash": None,
                      "owner_hash": api._token_hash(token), "session_hash": None}
    return jid, token


def test_job_runs_to_a_readable_result(slice_df):
    """The state machine runs to completion and the owner reads the result.
    Which tier that result is, and what each contains, is tests/test_tiers.py
    and tests/test_job_lifecycle.py."""
    jid, token = _queue(len(slice_df))
    api._work(jid, slice_df, "smiles", "pIC50", None, 3)

    out = api.job_status(jid, sub=None, x_job_token=token, autopsy_session="")
    assert out["status"] == "done"
    assert out["progress"] == "complete"
    assert out["result"]["tier"] == "full"
    assert out["result"]["verdict"]["reported"] is not None
    assert out["elapsed"] >= 0


def test_job_records_engine_failure_instead_of_raising(slice_df):
    """A bad audit must land as a failed job the poller can read, not as an
    exception that kills the worker silently."""
    jid, token = _queue(len(slice_df))
    api._work(jid, slice_df, "smiles", "no_such_column", None, 3)

    out = api.job_status(jid, sub=None, x_job_token=token, autopsy_session="")
    assert out["status"] == "failed"
    assert "not found" in out["error"]
    assert "result" not in out


def test_unknown_job_is_404():
    with pytest.raises(HTTPException) as exc:
        api.job_status("nope", sub=None, x_job_token="", autopsy_session="")
    assert exc.value.status_code == 404


def test_columns_are_checked_before_queueing(slice_df):
    """A typo in a column name should come back at submit, not a minute later
    as a failed job. Same for anything else the engine can see for free —
    see tests/test_malformed_input.py."""
    with pytest.raises(HTTPException) as exc:
        api._precheck(slice_df, "smiles", "nope", None)
    assert exc.value.status_code == 422
    assert "nope" in exc.value.detail


def test_finished_jobs_are_reaped(monkeypatch):
    monkeypatch.setattr(api, "JOB_TTL", 0)
    jid, _ = _queue(100)
    api._jobs[jid].update(status="done", finished=time.time() - 1)
    api._reap(time.time())
    assert jid not in api._jobs
