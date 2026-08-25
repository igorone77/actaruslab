"""
MODEL AUTOPSY — HTTP service.

    uvicorn autopsy.api:app --reload

Exposes the same engine the CLI uses, so the NEUTRA UI can POST a dataset
and render the real verdict instead of the precalculated BACE numbers.

Endpoints
---------
GET  /health                 -> liveness
POST /autopsy/csv            -> multipart file upload + column names -> full result JSON
POST /autopsy/records        -> JSON body {records:[{smiles,y,date?}], ...} -> full result JSON
GET  /                       -> the NEUTRA UI, if web/static has been built

Serving the UI from this app is what makes it usable in a browser with one
command: the page and the API share an origin, so nothing needs CORS. The
CORS middleware below only matters for a UI hosted somewhere else, and is
dev-open — lock it to the UI origin before deploying.
"""
from __future__ import annotations
import io
from pathlib import Path
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import run_autopsy, AutopsyError
from . import __version__

app = FastAPI(title="ActarusLab · Model Autopsy", version=__version__)

# Dev-open CORS. In production replace "*" with the UI origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "model-autopsy", "version": __version__}


# ── CSV upload path (what a browser file-picker sends) ────────────────
@app.post("/autopsy/csv")
async def autopsy_csv(
    file: UploadFile = File(...),
    smiles: str = Form(...),
    y: str = Form(...),
    date: Optional[str] = Form(None),
    k: int = Form(5),
):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not parse CSV: {e}")
    return _run(df, smiles, y, date, k)


# ── JSON records path (what the UI can send after in-browser parsing) ─
class Record(BaseModel):
    smiles: str
    y: float
    date: Optional[float] = None


class RecordsRequest(BaseModel):
    records: List[Record] = Field(..., min_length=40)
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
    try:
        res = run_autopsy(df, smiles, y, date, k=k)
    except AutopsyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # unexpected — surface as 500 but don't leak internals
        raise HTTPException(status_code=500, detail=f"autopsy failed: {type(e).__name__}")
    return res.to_dict()


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
