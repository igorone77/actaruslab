"""
Surface smoke tests: the two things wrapped around the engine — the HTML
report and the HTTP API — must keep consuming an AutopsyResult without
drifting from its shape.

Deliberately cheap: a 300-compound slice of BACE, k=3. These guard the
serialisers, not the numbers; tests/test_bace_smoke.py guards the numbers.
"""
import json
import re
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


def test_api_records_returns_full_result(slice_df):
    req = RecordsRequest(
        records=[Record(smiles=s, y=float(v))
                 for s, v in slice_df[["smiles", "pIC50"]].itertuples(index=False)],
        k=3,
    )
    out = autopsy_records(req)
    assert set(out) == {"specimen", "ladder", "verdict", "readout", "meta"}
    assert out["specimen"]["n_compounds"] == 300
    assert out["verdict"]["reported"] is not None


def test_engine_rejects_bad_column(slice_df):
    with pytest.raises(AutopsyError, match="not found"):
        run_autopsy(slice_df, "smiles", "no_such_column")


def test_engine_rejects_tiny_dataset(slice_df):
    with pytest.raises(AutopsyError, match=">= 40"):
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
    assert set(data) == {"specimen", "ladder", "verdict", "readout", "meta"}
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


def test_api_rejects_oversized_dataset(monkeypatch, slice_df):
    """The lookup rung is O(n²), so the service caps what it will accept
    rather than holding a worker open indefinitely."""
    monkeypatch.setattr(api, "MAX_ROWS", 100)
    req = RecordsRequest(
        records=[Record(smiles=s, y=float(v))
                 for s, v in slice_df[["smiles", "pIC50"]].itertuples(index=False)],
        k=3,
    )
    with pytest.raises(HTTPException) as exc:
        autopsy_records(req)
    assert exc.value.status_code == 413
    assert "300 rows" in exc.value.detail
