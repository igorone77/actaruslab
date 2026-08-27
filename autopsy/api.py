"""
MODEL AUTOPSY — HTTP service.

    uvicorn autopsy.api:app --reload

Exposes the same engine the CLI uses, so the NEUTRA UI can POST a dataset
and render the real verdict instead of the precalculated BACE numbers.

Endpoints
---------
GET  /health                 -> liveness
POST /autopsy/jobs           -> multipart upload -> 202 {job_id}; the audit runs
                                off the request. This is the path a hosted
                                deployment must use.
GET  /autopsy/jobs/{id}      -> {status, progress, result?, error?}
POST /autopsy/csv            -> multipart file upload + column names -> full result JSON
POST /autopsy/records        -> JSON body {records:[{smiles,y,date?}], ...} -> full result JSON
GET  /                       -> the NEUTRA UI, if web/static has been built

The two synchronous endpoints run the whole ladder before answering — 19 s
for 1500 compounds, quadratic in the lookup rung — so they outlive the
request timeout of most hosts on any realistic dataset. They are kept for
local use and scripting; anything public should submit a job and poll.

Serving the UI from this app is what makes it usable in a browser with one
command: the page and the API share an origin, so nothing needs CORS.

Deployment knobs, all environment variables, all safe by default:

    AUTOPSY_ALLOWED_ORIGINS   comma-separated origins allowed to call the API
                              cross-site. Unset means same-origin only — no
                              CORS headers at all, which is correct when this
                              app serves its own UI. Set it only for a front
                              end hosted elsewhere (a Vite dev server, a
                              separate static host).
    AUTOPSY_MAX_UPLOAD_MB     reject larger uploads with 413. Default 25.
    AUTOPSY_MAX_ROWS          reject wider datasets with 413. Default 5000,
                              the ceiling the O(n²) lookup rung is comfortable
                              at.
    AUTOPSY_WORKERS           audits running at once. Default 1 — each one
                              saturates the CPU, so more threads mostly means
                              everyone waits longer.
    AUTOPSY_QUEUE_DEPTH       jobs allowed to be waiting. Beyond it, submits
                              get 429. Default 8.
    AUTOPSY_JOB_TTL           seconds a finished job stays readable. Default
                              3600.

Jobs live in this process's memory: they do not survive a restart and are
not shared between replicas. That is the right trade for a single container
and the wrong one for a horizontally scaled deployment, which would need
Redis or a database behind the same two endpoints.
"""
from __future__ import annotations
import io
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import run_autopsy, precheck, AutopsyError
from . import __version__

app = FastAPI(title="ActarusLab · Model Autopsy", version=__version__)

MAX_UPLOAD_BYTES = int(float(os.getenv("AUTOPSY_MAX_UPLOAD_MB", "25")) * 1024 * 1024)
MAX_ROWS = int(os.getenv("AUTOPSY_MAX_ROWS", "5000"))
WORKERS = int(os.getenv("AUTOPSY_WORKERS", "1"))
QUEUE_DEPTH = int(os.getenv("AUTOPSY_QUEUE_DEPTH", "8"))
JOB_TTL = int(os.getenv("AUTOPSY_JOB_TTL", "3600"))

# Same-origin only unless told otherwise. The UI this app serves needs no
# CORS at all, so an open policy would only ever widen the attack surface of
# a public deployment for nobody's benefit.
_ORIGINS = [o.strip() for o in os.getenv("AUTOPSY_ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "model-autopsy", "version": __version__}


# ── validation, shared by every entry point ───────────────────────────
def _parse_upload(raw: bytes) -> pd.DataFrame:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV is {len(raw) / 1048576:.1f} MB; the limit is "
                   f"{MAX_UPLOAD_BYTES / 1048576:.0f} MB.")
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not parse CSV: {e}")


def _check_size(df: pd.DataFrame) -> None:
    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"{len(df)} rows; this service audits up to {MAX_ROWS}. The "
                   f"nearest-neighbour rung is O(n²), so larger sets belong in "
                   f"the CLI, or behind an approximate-NN index.")
    if len(df) < 40:
        raise HTTPException(
            status_code=422,
            detail=f"need >= 40 rows to run an audit; got {len(df)}.")


def _precheck(df: pd.DataFrame, smiles: str, y: str, date: Optional[str]) -> None:
    """The engine's own cheap validations, run before queueing, so a typo in a
    column name or a constant activity column comes back in milliseconds
    instead of as a failed job a minute later."""
    try:
        precheck(df, smiles, y, date)
    except AutopsyError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── CSV upload path (what a browser file-picker sends) ────────────────
@app.post("/autopsy/csv")
async def autopsy_csv(
    file: UploadFile = File(...),
    smiles: str = Form(...),
    y: str = Form(...),
    date: Optional[str] = Form(None),
    k: int = Form(5),
):
    return _run(_parse_upload(await file.read()), smiles, y, date, k)


# ── JSON records path (what the UI can send after in-browser parsing) ─
class Record(BaseModel):
    smiles: str
    y: float
    date: Optional[float] = None


class RecordsRequest(BaseModel):
    records: List[Record] = Field(..., min_length=40, max_length=MAX_ROWS)
    k: int = 5
    has_date: bool = False


@app.post("/autopsy/records")
def autopsy_records(req: RecordsRequest):
    rows = [{"smiles": r.smiles, "y": r.y, **({"date": r.date} if r.date is not None else {})}
            for r in req.records]
    df = pd.DataFrame(rows)
    date_col = "date" if (req.has_date and "date" in df.columns) else None
    return _run(df, "smiles", "y", date_col, req.k)


# ── shared runner ─────────────────────────────────────────────────────
def _run(df: pd.DataFrame, smiles: str, y: str, date: Optional[str], k: int):
    _check_size(df)
    try:
        res = run_autopsy(df, smiles, y, date, k=k)
    except AutopsyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        # A class name is not a message. Anything the engine did not anticipate
        # is our bug, and says so in words the uploader can act on.
        raise HTTPException(
            status_code=500,
            detail="the audit could not be completed on this file. Nothing about it looked wrong "
                   "up front, so this is a fault on our side rather than a problem with "
                   "your data.")
    return res.to_dict()


# ── jobs ──────────────────────────────────────────────────────────────
# The audit is far too slow to answer inside a request, so a submit hands
# it to a worker thread and returns an id. `progress` carries the engine's
# own log line, which is what makes a 20-second wait legible rather than a
# spinner.

_executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="autopsy")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _reap(now: float) -> None:
    """Drop finished jobs past their TTL. Caller holds the lock."""
    for jid, job in list(_jobs.items()):
        if job["finished"] and now - job["finished"] > JOB_TTL:
            del _jobs[jid]


def _pending() -> int:
    """Caller holds the lock."""
    return sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))


def _work(job_id: str, df: pd.DataFrame, smiles: str, y: str,
          date: Optional[str], k: int) -> None:
    def note(msg: str) -> None:
        with _lock:
            if job_id in _jobs:
                _jobs[job_id]["progress"] = msg

    with _lock:
        if job_id not in _jobs:          # reaped or cancelled before we started
            return
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started"] = time.time()

    try:
        res = run_autopsy(df, smiles, y, date, k=k, log=note)
        outcome = {"status": "done", "result": res.to_dict(), "progress": "complete"}
    except AutopsyError as e:
        outcome = {"status": "failed", "error": str(e), "progress": None}
    except Exception:                    # our bug, not their data — say it in words
        outcome = {"status": "failed",
                   "error": "the audit could not be completed on this file. Nothing about it looked wrong "
                   "up front, so this is a fault on our side rather than a problem with "
                   "your data.",
                   "progress": None}

    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(outcome, finished=time.time())


@app.post("/autopsy/jobs", status_code=202)
async def submit_job(
    file: UploadFile = File(...),
    smiles: str = Form(...),
    y: str = Form(...),
    date: Optional[str] = Form(None),
    k: int = Form(5),
):
    """Validate now, audit later. Bad columns and oversized data still fail
    immediately with 4xx; only the slow part is deferred."""
    df = _parse_upload(await file.read())
    _check_size(df)
    _precheck(df, smiles, y, date)

    now = time.time()
    with _lock:
        _reap(now)
        if _pending() >= QUEUE_DEPTH:
            raise HTTPException(
                status_code=429,
                detail=f"{QUEUE_DEPTH} audits already queued. Each one saturates a "
                       f"CPU for tens of seconds; try again shortly.")
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"status": "queued", "progress": "queued", "result": None,
                         "error": None, "created": now, "started": None,
                         "finished": None, "rows": int(len(df))}

    _executor.submit(_work, job_id, df, smiles, y, date, k)
    return {"job_id": job_id, "status": "queued", "rows": int(len(df))}


@app.get("/autopsy/jobs/{job_id}")
def job_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail="unknown job — it expired, or the service restarted")
        out = {"job_id": job_id, "status": job["status"], "progress": job["progress"],
               "rows": job["rows"],
               "elapsed": round((job["finished"] or time.time()) - job["created"], 1)}
        if job["status"] == "done":
            out["result"] = job["result"]
        elif job["status"] == "failed":
            out["error"] = job["error"]
        return out


# ── the UI ────────────────────────────────────────────────────────────
# Mounted last so it cannot shadow the routes above. Absent until the
# bundle is built (`npm install && npm run build`), in which case the API
# still serves normally — only the browser front end is missing.
_STATIC = Path(__file__).resolve().parent.parent / "web" / "static"

if (_STATIC / "app.js").exists():
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="ui")
else:
    @app.get("/")
    def ui_not_built():
        return {
            "detail": "UI bundle not built",
            "fix": "npm install && npm run build, then restart",
            "api": ["/health", "/autopsy/csv", "/autopsy/records", "/docs"],
        }
