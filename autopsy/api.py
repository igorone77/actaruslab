"""
MODEL AUTOPSY — HTTP service.

    uvicorn autopsy.api:app --reload

Exposes the same engine the CLI uses, so the NEUTRA UI can POST a dataset
and render the real verdict instead of the precalculated BACE numbers.

Endpoints
---------
GET  /health                 -> liveness
POST /autopsy/jobs           -> multipart upload -> 202 {job_id, job_token}; the
                                audit runs off the request. This is the path a
                                hosted deployment must use.
GET  /autopsy/jobs/{id}      -> {status, progress, result?, error?}, to the
                                holder of that job's token
POST /autopsy/csv            -> multipart file upload + column names -> result JSON
POST /autopsy/records        -> JSON body {records:[{smiles,y,date?}], ...} -> result JSON
GET  /autopsy/demo           -> the BACE-1 audit, precomputed, in full
GET  /                       -> the NEUTRA UI, if web/static has been built

Who gets what
-------------
The engine is free: an audit runs for anyone, with no key and no
subscription. What comes back is decided in autopsy/tiers.py — the free tier
is the synthetic verdict, one percentage, and the reserved tier is the
diagnosis behind it. `AUTOPSY_PAYWALL_ENABLED` (default false, see
billing.paywall_enabled) is the only thing that puts the engine itself behind
a subscription again.

Two mechanisms that look alike and are not:

    require_subscription   may this caller *run* an audit — the payment gate,
                           switched off by default
    _authorize_job         may this caller *read this result* — ownership,
                           on in every configuration, because a free audit is
                           still the uploader's data

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
    AUTOPSY_PAYWALL_ENABLED   put the engine behind a subscription again.
                              Default false. Read by autopsy/billing.py.

Jobs live in this process's memory: they do not survive a restart and are
not shared between replicas. That is the right trade for a single container
and the wrong one for a horizontally scaled deployment, which would need
Redis or a database behind the same two endpoints.
"""
from __future__ import annotations
import hashlib
import hmac
import io
import json
import os
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List

import pandas as pd
from fastapi import Depends, FastAPI, Header, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import billing, tiers
from .billing import QuotaExhausted, require_subscription
from .engine import run_autopsy, precheck, AutopsyError
from . import __version__

app = FastAPI(title="ActarusLab · Model Autopsy", version=__version__)

billing.init_db()
app.include_router(billing.router)

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
    sub=Depends(require_subscription),
):
    df = _parse_upload(await file.read())
    _precheck(df, smiles, y, date)
    return tiers.view_for(_metered(sub, lambda: _run(df, smiles, y, date, k)), sub)


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
def autopsy_records(req: RecordsRequest, sub=Depends(require_subscription)):
    rows = [{"smiles": r.smiles, "y": r.y, **({"date": r.date} if r.date is not None else {})}
            for r in req.records]
    df = pd.DataFrame(rows)
    date_col = "date" if (req.has_date and "date" in df.columns) else None
    _precheck(df, "smiles", "y", date_col)
    return tiers.view_for(
        _metered(sub, lambda: _run(df, "smiles", "y", date_col, req.k)), sub)


# ── metering ──────────────────────────────────────────────────────────
def _metered(sub, work):
    """Spend one audit from the cycle, and give it back if the failure was
    ours. A subscriber must never lose one of their 20 to our bug."""
    if sub is None:                      # paywall off — the audit is free
        return work()
    try:
        billing.reserve_audit(sub["key_hash"])
    except QuotaExhausted as e:
        raise HTTPException(
            status_code=402,
            detail=f"you have used all {billing.QUOTA} audits in this billing "
                   f"cycle. The count resets when the subscription renews.",
            headers={"X-Checkout-URL": billing.checkout_url()})
    try:
        return work()
    except HTTPException as e:
        if e.status_code >= 500:            # our fault, not their data
            billing.refund_audit(sub["key_hash"])
        raise
    except Exception:
        billing.refund_audit(sub["key_hash"])
        raise


# ── the free demo ─────────────────────────────────────────────────────
_DEMO = Path(__file__).resolve().parent / "demo_result.json"


@app.get("/autopsy/demo")
def autopsy_demo():
    """The BACE-1 audit, precomputed, and the one full result served to
    anyone. It is the showcase: our dataset, our diagnosis, published in
    full so that a visitor can see exactly what the reserved tier contains
    before asking for it on their own data."""
    if not _DEMO.exists():
        raise HTTPException(
            status_code=503,
            detail="the demo result has not been generated on this deployment.")
    return json.loads(_DEMO.read_text())


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

# One answer for "not yours" and for "never existed". A 403 would confirm
# that this id belongs to someone, which is already the leak.
_NO_SUCH_JOB = "unknown job — it expired, or the service restarted"


def issue_job_token() -> str:
    """The claim on a job's result, minted at submit and returned once.

    An audit run without a subscription still has an owner: whoever uploaded
    the file. This is what makes that true when there is no account to hang
    ownership on — 256 bits from the OS, held by the browser that submitted,
    stored here only as a hash. Losing it means the result is unreadable by
    anybody, which is the correct failure for someone else's data.
    """
    return secrets.token_urlsafe(32)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _authorize_job(job: dict, sub, token: str) -> None:
    """May this caller read this result? Nothing to do with payment.

    The free tier makes an audit cheap, not public. Two ways to own a job,
    checked in every configuration including the free one:

      · the one-time token from the submit response — an anonymous audit;
      · the subscription key that submitted it, when the paywall is on.

    A job with neither recorded is owned by nobody and readable by nobody.
    That is the safe direction to fail: an entry point that forgets to set an
    owner locks its own results instead of publishing everyone else's.
    """
    if (sub is not None and job.get("key_hash")
            and hmac.compare_digest(job["key_hash"], sub["key_hash"])):
        return
    if (token and job.get("owner_hash")
            and hmac.compare_digest(job["owner_hash"], _token_hash(token))):
        return
    raise HTTPException(status_code=404, detail=_NO_SUCH_JOB)


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
        with _lock:
            kh = (_jobs.get(job_id) or {}).get("key_hash")
        if kh:
            billing.refund_audit(kh)
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
    sub=Depends(require_subscription),
):
    """Validate now, audit later. Bad columns and oversized data still fail
    immediately with 4xx; only the slow part is deferred.

    The response carries `job_token` once. It is the only claim on the result
    and it is not recoverable — this process keeps a hash of it, exactly as
    billing keeps a hash of an API key."""
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
        token = issue_job_token()
        _jobs[job_id] = {"status": "queued", "progress": "queued", "result": None,
                         "error": None, "created": now, "started": None,
                         "finished": None, "rows": int(len(df)),
                         "key_hash": sub["key_hash"] if sub else None,
                         "owner_hash": _token_hash(token)}

    # Spent here rather than on completion: a queued audit already holds a
    # worker, and refunded below if the failure turns out to be ours.
    if sub is None:                      # paywall off — the audit is free
        _executor.submit(_work, job_id, df, smiles, y, date, k)
        return {"job_id": job_id, "job_token": token, "status": "queued",
                "rows": int(len(df))}
    try:
        used = billing.reserve_audit(sub["key_hash"])
    except QuotaExhausted:
        with _lock:
            _jobs.pop(job_id, None)
        raise HTTPException(
            status_code=402,
            detail=f"you have used all {billing.QUOTA} audits in this billing "
                   f"cycle. The count resets when the subscription renews.",
            headers={"X-Checkout-URL": billing.checkout_url()})

    _executor.submit(_work, job_id, df, smiles, y, date, k)
    return {"job_id": job_id, "job_token": token, "status": "queued",
            "rows": int(len(df)),
            "audits_used": used,
            "audits_per_cycle": billing.QUOTA}


@app.get("/autopsy/jobs/{job_id}")
def job_status(job_id: str, sub=Depends(require_subscription),
               x_job_token: str = Header(default="")):
    """An audit result is the uploader's data, so only whoever submitted it
    can read it — free audit or paid one, the check is the same and it is
    always on. Job ids are unguessable, but that is not a reason to serve one
    to whoever asks.

    The token travels in a header rather than the path so that it stays out
    of access logs, browser history and referrers, which the job id does not.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=_NO_SUCH_JOB)
        _authorize_job(job, sub, x_job_token)
        out = {"job_id": job_id, "status": job["status"], "progress": job["progress"],
               "rows": job["rows"],
               "elapsed": round((job["finished"] or time.time()) - job["created"], 1)}
        if job["status"] == "done":
            # The full audit stays in this dict; what leaves is the caller's
            # tier of it, built key by key in tiers.public_view.
            out["result"] = tiers.view_for(job["result"], sub)
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
