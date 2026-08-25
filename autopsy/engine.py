"""
ActarusLab · MODEL AUTOPSY — engine
===================================
One function, `run_autopsy(df, ...)`, that performs the forensic
Leakage Ladder on a molecular dataset and returns a structured verdict.

The claim is never "the model is bad". The claim is:
    how much of a model's apparent performance is supported by
    evidence that generalises beyond memorised chemical neighbourhood.

Ladder (each rung harder to fool than the last):
    XGBoost · random split      -> the optimistic number people report
    1-NN Tanimoto · random      -> how much of that is pure similarity lookup
    XGBoost · scaffold split    -> what survives a genuinely new chemical series
    1-NN Tanimoto · scaffold    -> lookup floor on new series
    XGBoost · temporal split    -> generalisation forward in time (if dates given)
    permutation (shuffled y)    -> honesty floor; must collapse toward 0

Nothing here is model magic: RDKit + scikit-learn + XGBoost do the numbers.
The engine only decides how to *validate*, and reports what it finds.

Design notes
------------
* Pure function of a DataFrame in, dict out — trivially wrapped by CLI or API.
* No global state, no file I/O here (the CLI/serialisers handle that).
* Deterministic: fixed seeds, so an audit is reproducible.
* Fails loud and specific on bad input (the API surfaces these as 4xx).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

import xgboost as xgb

RDLogger.DisableLog("rdApp.*")

__all__ = ["run_autopsy", "AutopsyError", "AutopsyResult"]


class AutopsyError(ValueError):
    """Raised for malformed input the caller must fix (bad columns, empty data)."""


# ─────────────────────────────────────────────────────────────────────
# featurisation
# ─────────────────────────────────────────────────────────────────────
def _fingerprint(smiles: str, n_bits: int, radius: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return bv, arr


def _scaffold(smiles: str) -> str:
    """Generic Bemis–Murcko scaffold (atoms genericised, so analogue series
    collapse to one core). Acyclic / uncomputable -> a shared bucket label."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "__invalid__"
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
    except Exception:
        return "__nocore__"


# ─────────────────────────────────────────────────────────────────────
# model + evaluation
# ─────────────────────────────────────────────────────────────────────
def _model(seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed,
    )


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "spearman": round(float(spearmanr(y_true, y_pred).statistic), 3),
    }


def _xgb_oof(X, y, folds, seed, target=None) -> dict:
    yy = y if target is None else target
    pred = np.full(len(yy), np.nan)
    for tr, te in folds:
        m = _model(seed)
        m.fit(X[tr], yy[tr])
        pred[te] = m.predict(X[te])
    mask = ~np.isnan(pred)
    return _metrics(yy[mask], pred[mask])


def _nn_oof(bvs, y, folds) -> dict:
    """1-nearest-neighbour by Tanimoto: predict each molecule's activity as
    that of its single closest training molecule. This is the pure-similarity
    baseline — how far you get with a lookup table and no learning at all."""
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        pool = [bvs[i] for i in tr]
        for j in te:
            sims = DataStructs.BulkTanimotoSimilarity(bvs[j], pool)
            pred[j] = y[tr[int(np.argmax(sims))]]
    mask = ~np.isnan(pred)
    return _metrics(y[mask], pred[mask])


# ─────────────────────────────────────────────────────────────────────
# splitters
# ─────────────────────────────────────────────────────────────────────
def _random_folds(n, k, seed):
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)))


def _scaffold_folds(scaffolds, k):
    groups = pd.factorize(pd.Series(scaffolds))[0]
    k_eff = min(k, len(set(groups)))
    if k_eff < 2:
        return None, groups
    return list(GroupKFold(n_splits=k_eff).split(np.arange(len(groups)), groups=groups)), groups


def _temporal_folds(dates, k):
    """Expanding window: sort by date, test each future block on the past
    before it. This is the split a random fold hides completely."""
    order = np.argsort(np.asarray(dates, dtype=float))
    blocks = np.array_split(order, k + 1)
    return [(np.concatenate(blocks[:i]), blocks[i]) for i in range(1, k + 1)]


# ─────────────────────────────────────────────────────────────────────
# result container
# ─────────────────────────────────────────────────────────────────────
@dataclass
class AutopsyResult:
    specimen: dict
    ladder: list                    # ordered rungs, each a dict
    verdict: dict                   # headline numbers + interpretation
    readout: list                   # per-signal cards for the UI
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────
# the engine
# ─────────────────────────────────────────────────────────────────────
def run_autopsy(
    df: pd.DataFrame,
    smiles_col: str,
    y_col: str,
    date_col: Optional[str] = None,
    *,
    k: int = 5,
    n_bits: int = 2048,
    radius: int = 2,
    seed: int = 0,
    log=lambda msg: None,          # optional progress callback (CLI prints, API streams)
) -> AutopsyResult:
    # ---- validate ----------------------------------------------------
    for col in (smiles_col, y_col):
        if col not in df.columns:
            raise AutopsyError(f"column '{col}' not found. Available: {list(df.columns)}")
    if date_col and date_col not in df.columns:
        raise AutopsyError(f"date column '{date_col}' not found. Available: {list(df.columns)}")

    work = df[[smiles_col, y_col] + ([date_col] if date_col else [])].copy()
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[smiles_col, y_col])
    if len(work) < 40:
        raise AutopsyError(f"need >= 40 valid rows to run an audit; got {len(work)}.")

    # ---- featurise ---------------------------------------------------
    log("featurising molecules (ECFP)…")
    bvs, arrs, keep = [], [], []
    for i, s in enumerate(work[smiles_col].astype(str).values):
        bv, arr = _fingerprint(s, n_bits, radius)
        if bv is not None:
            bvs.append(bv); arrs.append(arr); keep.append(i)
    n_bad = len(work) - len(keep)
    d = work.iloc[keep].reset_index(drop=True)
    X = np.vstack(arrs)
    y = d[y_col].astype(float).values
    log(f"  {len(y)} parsed, {n_bad} unparseable SMILES dropped")

    # ---- scaffolds ---------------------------------------------------
    log("computing Bemis–Murcko scaffold series…")
    scaffolds = [_scaffold(s) for s in d[smiles_col].astype(str).values]
    scaf_counts = pd.Series(scaffolds).value_counts()
    n_scaffolds = int(len(scaf_counts))
    n_singletons = int((scaf_counts == 1).sum())
    largest = int(scaf_counts.iloc[0])

    # ---- ladder ------------------------------------------------------
    ladder = []
    rf = _random_folds(len(y), k, seed)

    log("rung · XGBoost random split…")
    m = _xgb_oof(X, y, rf, seed)
    ladder.append(_rung("XGBoost", "random split", m, "reported"))
    reported = m["r2"]

    log("rung · 1-NN Tanimoto random split…")
    m = _nn_oof(bvs, y, rf)
    ladder.append(_rung("1-NN lookup", "random split", m, "lookup"))
    lookup_rand = m["r2"]

    log("rung · XGBoost scaffold split…")
    sf, groups = _scaffold_folds(scaffolds, k)
    if sf is not None:
        m = _xgb_oof(X, y, sf, seed)
        ladder.append(_rung("XGBoost", f"scaffold split · {n_scaffolds} series", m, "survives"))
        survives = m["r2"]
        log("rung · 1-NN Tanimoto scaffold split…")
        m = _nn_oof(bvs, y, sf)
        ladder.append(_rung("1-NN lookup", "scaffold split", m, "lookup"))
        nn_scaf = m["r2"]
    else:
        survives = nn_scaf = None
        ladder.append(_rung("XGBoost", "scaffold split", None, "survives",
                            note="too few scaffold series to split"))

    # temporal (optional)
    if date_col and d[date_col].notna().any():
        log("rung · XGBoost temporal split…")
        dates = pd.to_numeric(d[date_col], errors="coerce")
        if dates.notna().sum() >= 40 and dates.nunique() >= 3:
            valid = dates.notna().values
            m = _xgb_oof(X[valid], y[valid], _temporal_folds(dates[valid].values, k), seed)
            ladder.append(_rung("XGBoost", "temporal split · past→future", m, "temporal"))
            temporal = m["r2"]
        else:
            temporal = None
            ladder.append(_rung("XGBoost", "temporal split", None, "temporal",
                                note="too few dated points to split"))
    else:
        temporal = None
        ladder.append(_rung("XGBoost", "temporal split", None, "temporal",
                            note="no assay dates supplied"))

    # permutation control
    log("rung · permutation floor…")
    rng = np.random.default_rng(seed)
    yp = y.copy(); rng.shuffle(yp)
    m = _xgb_oof(X, yp, rf, seed, target=yp)
    ladder.append(_rung("Permutation", "shuffled target", m, "floor"))
    floor = m["r2"]

    # ---- verdict -----------------------------------------------------
    learned = round(survives - nn_scaf, 3) if (survives is not None and nn_scaf is not None) else None
    lookup_pct = int(round(100 * lookup_rand / reported)) if reported > 0 else None

    verdict = {
        "reported": reported,
        "lookup_random": lookup_rand,
        "survives_scaffold": survives,
        "lookup_scaffold": nn_scaf,
        "learned_beyond_lookup": learned,
        "temporal": temporal,
        "permutation_floor": floor,
        "lookup_pct_of_reported": lookup_pct,
        "headline": _headline(reported, lookup_pct, survives, learned, temporal, floor),
    }

    readout = _readout(reported, lookup_rand, lookup_pct, survives, nn_scaf,
                       learned, temporal, floor)

    specimen = {
        "n_compounds": int(len(y)),
        "n_unparseable_dropped": n_bad,
        "n_scaffold_series": n_scaffolds,
        "n_singleton_series": n_singletons,
        "largest_series": largest,
        "largest_series_pct": round(100 * largest / len(y), 1),
        "target_mean": round(float(np.mean(y)), 3),
        "target_sd": round(float(np.std(y)), 3),
        "exact_duplicate_smiles": int(d[smiles_col].duplicated().sum()),
        "has_dates": bool(date_col and d[date_col].notna().any()),
    }

    meta = {
        "featurisation": f"ECFP{2*radius} · {n_bits} bit",
        "model": "XGBoost (400 trees, depth 6)",
        "k_folds": k,
        "seed": seed,
        "columns": {"smiles": smiles_col, "target": y_col, "date": date_col},
    }

    return AutopsyResult(specimen=specimen, ladder=ladder,
                         verdict=verdict, readout=readout, meta=meta)


# ─────────────────────────────────────────────────────────────────────
# small builders (kept pure so tests can assert on them)
# ─────────────────────────────────────────────────────────────────────
def _rung(model, cond, metrics, kind, note=None):
    row = {"model": model, "condition": cond, "kind": kind}
    if metrics is None:
        row.update({"r2": None, "rmse": None, "spearman": None, "note": note})
    else:
        row.update(metrics)
        if note:
            row["note"] = note
    return row


def _headline(reported, lookup_pct, survives, learned, temporal, floor):
    if reported <= 0:
        return "Model shows no predictive signal even on a random split."
    parts = []
    if lookup_pct is not None:
        parts.append(f"{lookup_pct}% of the reported R² {reported:.2f} is reproducible by a "
                     f"pure nearest-neighbour lookup — recognition of known analogues, not learned SAR.")
    if survives is not None:
        parts.append(f"On disjoint chemical series performance holds at {survives:.2f}.")
    if learned is not None:
        parts.append(f"Beyond the lookup baseline, the model's own contribution is {learned:.2f}.")
    if temporal is not None:
        parts.append(f"Forward in time it holds {temporal:.2f}.")
    if floor is not None and floor > 0.05:
        parts.append(f"WARNING: permutation floor is {floor:.2f} (> 0) — the pipeline itself may be leaking; "
                     f"fix featurisation/splitting before trusting any number above.")
    return " ".join(parts)


def _readout(reported, lookup_rand, lookup_pct, survives, nn_scaf, learned, temporal, floor):
    cards = []
    # similarity leakage
    if lookup_pct is not None:
        sev = "SEVERE" if lookup_pct >= 70 else "MODERATE" if lookup_pct >= 40 else "LOW"
        cards.append({"signal": "Similarity leakage", "flag": sev, "value_pct": lookup_pct,
                      "note": "Share of the reported score a bare nearest-neighbour lookup reproduces. "
                              "High means the model is rewarded for recognising known analogues."})
    # scaffold transfer
    if survives is not None:
        cards.append({"signal": "Scaffold transfer", "flag": "PARTIAL", "value": survives,
                      "note": "Performance on genuinely new chemical series — what generalises beyond the training scaffolds."})
    # learned structure
    if learned is not None:
        flag = "NET" if learned > 0.1 else "THIN"
        cards.append({"signal": "Learned structure", "flag": flag, "value": learned,
                      "note": "Scaffold performance minus the scaffold-split lookup: the only structure the model added over copying its nearest analogue."})
    # temporal
    cards.append({"signal": "Temporal test",
                  "flag": "N/A" if temporal is None else "TESTED",
                  "value": temporal,
                  "note": "Generalisation forward in time. Activates when assay dates are supplied — the split a random fold hides entirely."})
    # permutation floor
    if floor is not None:
        flag = "CLEAN" if floor <= 0.05 else "LEAK"
        cards.append({"signal": "Permutation floor", "flag": flag, "value": floor,
                      "note": "Shuffled-target control. Should collapse toward zero; anything well above signals a pipeline leak."})
    return cards
