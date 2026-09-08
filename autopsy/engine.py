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
                                   (ties averaged, so it cannot read row order)
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
* Deterministic: rows are put in canonical order and seeds are fixed, so
  the same molecules give the same audit — on any machine, in any file order.
* Fails loud and specific on bad input (the API surfaces these as 4xx).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

import xgboost as xgb

RDLogger.DisableLog("rdApp.*")

__all__ = ["run_autopsy", "precheck", "AutopsyError", "AutopsyResult"]


# Below this spread the target carries no information a model could be scored
# against, and R² becomes the ratio of two rounding errors.
MIN_TARGET_SD = 1e-6

# Past this share of the file discarded, the result describes a subset.
DROP_SEVERE_PCT = 20.0

# Below this reported R² there is no score for the lower rungs to be a
# fraction of, and every ratio taken against it is noise over noise. Guarding
# on `> 0` was not enough: a reported 0.001 with a lookup of -0.50 produced
# "-50300% of score is lookup" — arithmetically correct, and exactly the kind
# of false-plausible number this tool exists to catch. 0.05 is where a model
# stops explaining a usable share of the variance.
MIN_REPORTED_R2 = 0.05

# How many partitions each repeated rung is measured over. One split is one
# sample: it cannot tell a real difference between two rungs from the noise of
# where the fold boundaries happened to fall. Five is the smallest number that
# gives a usable spread while keeping a 1500-compound audit inside a couple of
# minutes — the cost is linear in this, and the repeated rungs are the
# expensive ones.
REPEATS = 5

# Tanimoto radius for the experimental similarity split. 0.40 on ECFP4 is the
# usual working line between "analogue" and "unrelated" in the medicinal
# chemistry literature; it is a convention, not a measurement, which is part
# of why that rung is marked experimental.
SIM_CUTOFF = 0.40


class AutopsyError(ValueError):
    """Raised for malformed input the caller must fix. Its message is shown to
    the person who uploaded the file, so it names what went wrong and what to
    do — never an exception class or a status code."""


# ─────────────────────────────────────────────────────────────────────
# featurisation
# ─────────────────────────────────────────────────────────────────────
def _fingerprint(smiles: str, n_bits: int, radius: int):
    """Returns (bitvector, dense array, canonical SMILES). The canonical form
    is the audit's sort key: it makes the row order a function of the
    molecules rather than of how the file happened to be written."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        # an empty SMILES parses into a valid molecule with no atoms, whose
        # all-zero fingerprint would join the audit as a phantom compound
        return None, None, None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return bv, arr, Chem.MolToSmiles(mol)


def _fingerprint_alt(smiles: str, n_bits: int):
    """A deliberately different fingerprint, for the descriptor control.

    RDKit's topological fingerprint enumerates linear paths through the graph;
    Morgan/ECFP enumerates circular environments around each atom. They
    disagree about what makes two molecules similar, which is the point: if
    the verdict only holds under ECFP4 it is a property of that descriptor
    rather than of the dataset, and the audit should not present it as the
    latter.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.RDKFingerprint(mol, fpSize=n_bits)


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
    """subsample and colsample_bytree draw against row positions, so this is
    reproducible only because run_autopsy canonicalises the row order first.
    n_jobs=-1 is safe here: measured across 1, 2, 4 and all cores, the pooled
    R² was identical to three decimals, so the thread count does not change
    the reduction order enough to move a reported number."""
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed,
    )


def _finite(x):
    """None rather than NaN/inf. Starlette serialises with allow_nan=False, so a
    NaN metric used to surface as an unexplained 500 instead of a missing value;
    a rung that could not be scored should read as unscored, everywhere."""
    v = float(x)
    return round(v, 3) if np.isfinite(v) else None


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": _finite(r2_score(y_true, y_pred)),
        "rmse": _finite(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": _finite(spearmanr(y_true, y_pred).statistic),
    }


def _agg(runs: list) -> dict:
    """Mean and sample standard deviation across repeated partitions.

    ddof=1 because these are a sample of the partitions that could have been
    drawn, not the population of them. With one run there is no spread to
    report and the band is None rather than zero — zero would claim a
    precision that was never measured.
    """
    out = {"n_reps": len(runs)}
    for key in ("r2", "rmse", "spearman"):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if not vals:
            out[key] = out[key + "_sd"] = None
        else:
            out[key] = _finite(np.mean(vals))
            out[key + "_sd"] = _finite(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return out


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
    """Nearest-neighbour by Tanimoto, ties averaged: predict each molecule's
    activity as the mean activity of *every* training molecule sitting at the
    maximum similarity. This is the pure-similarity baseline — how far you get
    with a lookup table and no learning at all.

    Averaging matters twice over. `np.argmax` would keep whichever tied
    neighbour came first, so the score moved with the order of the input file
    — on BACE, 99 of 1513 test molecules have a tied nearest neighbour and 83
    of those tie across neighbours with different activities. And when several
    analogues are equidistant, their mean is the better estimate of what a
    lookup can tell you; picking one at random throws information away."""
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        pool = [bvs[i] for i in tr]
        y_pool = y[tr]
        for j in te:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(bvs[j], pool))
            pred[j] = y_pool[sims == sims.max()].mean()
    mask = ~np.isnan(pred)
    return _metrics(y[mask], pred[mask])


# ─────────────────────────────────────────────────────────────────────
# splitters
# ─────────────────────────────────────────────────────────────────────
def _random_folds(n, k, seed):
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)))


def _fill_folds(groups, n_groups, k, order):
    """Greedy group-to-fold assignment: walk `order`, drop each group into the
    lightest fold. Shared by the scaffold split and the similarity split, which
    differ only in what a group *is*.

    `np.argmin` returns the lowest index on ties, so no step here consults an
    unstable sort and the partition is a function of `order` alone.
    """
    counts = np.bincount(groups, minlength=n_groups)
    weight = np.zeros(k, dtype=np.int64)
    group_to_fold = np.empty(n_groups, dtype=int)
    for g in order:
        f = int(np.argmin(weight))
        weight[f] += counts[int(g)]
        group_to_fold[int(g)] = f

    per_sample = group_to_fold[groups]
    return [(np.where(per_sample != f)[0], np.where(per_sample == f)[0])
            for f in range(k)]


def _scaffold_folds(scaffolds, k, rep: int = 0):
    """Partition scaffold series across folds — deterministically.

    Two decisions here used to be delegated, and both leaked nondeterminism
    into the audit:

      * group ids came from `pd.factorize`, i.e. order of first appearance,
        so shuffling the input rows renamed the groups;
      * fold assignment came from `GroupKFold`, which orders series by size
        with `np.argsort(...)` — an *unstable* sort. On BACE 200 of the 377
        series hold a single compound, so that order is one arbitrary choice
        among many, and which one you get depends on the scikit-learn version
        and on the CPU (numpy 2.x selects a SIMD sort path from the host's
        instruction set). Same code, same data, different partition: the
        scaffold-lookup rung read 0.365 on one machine and 0.421 on another.

    Both are now fixed by construction. Group ids come from
    `sorted(set(scaffolds))`, so they are a function of scaffold *content*
    rather than row order; groups are filled largest-first with ties broken
    by group id, into the lightest fold with ties broken by fold index. No
    step consults an unstable sort, so the partition is a function of the
    molecules alone — identical on any machine, and under any row ordering.

    `rep` selects *which* scaffold-disjoint partition. rep 0 is the ordering
    above and is what every earlier version of this engine produced; rep > 0
    walks the series in a seeded permutation instead, so a different set of
    series keeps company in each fold. Every partition is still scaffold-
    disjoint — no fold is ever tested on a series it trained on — and the
    spread across them is what the error band on this rung measures: how much
    of a rung's score is the dataset and how much is one arbitrary cut through
    it. The seed is `rep` and nothing else, so the whole set of partitions
    stays a function of the molecules.
    """
    index = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
    groups = np.fromiter((index[s] for s in scaffolds), dtype=int, count=len(scaffolds))
    n_groups = len(index)

    k_eff = min(k, n_groups)
    if k_eff < 2:
        return None, groups

    counts = np.bincount(groups, minlength=n_groups)
    if rep == 0:
        order = sorted(range(n_groups), key=lambda g: (-int(counts[g]), g))  # size desc, id asc
    else:
        order = np.random.default_rng(rep).permutation(n_groups)

    return _fill_folds(groups, n_groups, k_eff, order), groups


# ── the experimental rung ─────────────────────────────────────────────
def _similarity_clusters(bvs, cutoff: float):
    """Sphere exclusion on Tanimoto: group molecules by direct similarity
    rather than by shared scaffold.

    Walk the molecules in canonical order. Each one either falls within
    `cutoff` of an existing leader — and joins that leader's cluster — or
    becomes a leader itself. Splitting on these clusters keeps train and test
    apart in fingerprint space directly, which a scaffold split does not: two
    Bemis-Murcko cores can be formally distinct and still sit next to each
    other, so a scaffold-disjoint fold can still be full of near neighbours.

    Deterministic: canonical row order in, `np.argmax` taking the first leader
    on ties. O(n . leaders) and no distance matrix, so it stays inside the
    memory a 5000-row audit is allowed.

    A heuristic, not a validated protocol — see the limitation the result
    carries alongside the rung it feeds.
    """
    leaders: list = []
    assign = np.empty(len(bvs), dtype=int)
    for i, bv in enumerate(bvs):
        if leaders:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(bv, leaders))
            best = int(np.argmax(sims))
            if sims[best] >= cutoff:
                assign[i] = best
                continue
        leaders.append(bv)
        assign[i] = len(leaders) - 1
    return assign, len(leaders)


def _similarity_folds(assign, n_clusters, k):
    """Same greedy fill as the scaffold split, over similarity clusters."""
    k_eff = min(k, n_clusters)
    if k_eff < 2:
        return None
    counts = np.bincount(assign, minlength=n_clusters)
    order = sorted(range(n_clusters), key=lambda g: (-int(counts[g]), g))
    return _fill_folds(assign, n_clusters, k_eff, order)


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
    warnings: list = field(default_factory=list)   # {level, text} — see _drop_warning
    meta: dict = field(default_factory=dict)
    limitations: list = field(default_factory=list)  # declared, not hidden — see _limitations

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
    repeats: int = REPEATS,        # partitions per repeated rung; 1 disables the bands
    sim_cutoff: float = SIM_CUTOFF,
    log=lambda msg: None,          # optional progress callback (CLI prints, API streams)
) -> AutopsyResult:
    # ---- validate ----------------------------------------------------
    precheck(df, smiles_col, y_col, date_col)

    n_rows_in = int(len(df))
    work = df[[smiles_col, y_col] + ([date_col] if date_col else [])].copy()
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    n_bad_y = int(work[y_col].isna().sum())

    work = work.dropna(subset=[smiles_col, y_col])

    # ---- featurise ---------------------------------------------------
    log("featurising molecules (ECFP)…")
    bvs, arrs, canon, keep = [], [], [], []
    for i, s in enumerate(work[smiles_col].astype(str).values):
        bv, arr, cs = _fingerprint(s, n_bits, radius)
        if bv is not None:
            bvs.append(bv); arrs.append(arr); canon.append(cs); keep.append(i)
    n_bad = len(work) - len(keep)

    if len(keep) < 40:
        why = []
        if n_bad:
            why.append(f"{n_bad} had unreadable SMILES")
        if n_bad_y:
            why.append(f"{n_bad_y} had missing or non-numeric activity")
        detail = f" ({'; '.join(why)})" if why else ""
        raise AutopsyError(
            f"only {len(keep)} of {n_rows_in} rows are usable{detail}, and an audit needs "
            f"at least 40. Below that the folds are too small for the numbers to mean "
            f"anything.")

    d = work.iloc[keep].reset_index(drop=True)
    X = np.vstack(arrs)
    y = d[y_col].astype(float).values
    log(f"  {len(y)} parsed, {n_bad} unparseable SMILES dropped")

    # ---- canonical row order ------------------------------------------
    # Everything downstream reads positions: KFold splits them, XGBoost's
    # subsample and colsample_bytree draw against them, and the 1-NN pool is
    # walked in them. Left as they arrived, the same molecules in a different
    # file order gave a different audit. Sorting by canonical SMILES (then by
    # activity, and rows tying on both are interchangeable) makes every rung a
    # function of the molecule set alone.
    order = sorted(range(len(y)), key=lambda i: (canon[i], y[i]))
    bvs = [bvs[i] for i in order]
    X, y = X[order], y[order]
    d = d.iloc[order].reset_index(drop=True)

    # ---- scaffolds ---------------------------------------------------
    log("computing Bemis–Murcko scaffold series…")
    scaffolds = [_scaffold(s) for s in d[smiles_col].astype(str).values]
    scaf_counts = pd.Series(scaffolds).value_counts()
    n_scaffolds = int(len(scaf_counts))
    n_singletons = int((scaf_counts == 1).sum())
    largest = int(scaf_counts.iloc[0])

    # ---- ladder ------------------------------------------------------
    # The four rungs below are each measured over `repeats` different
    # partitions and reported as mean ± sd. One split is one sample: without
    # the spread there is no way to tell a real gap between two rungs from the
    # noise of where the fold boundaries fell. Model and lookup share the same
    # partitions within a repetition, so their difference is measured on the
    # same cut and not across two.
    ladder = []
    rep_reported, rep_lookup, rep_survives, rep_nn_scaf, rep_learned = [], [], [], [], []
    first_rand_folds = None
    scaffolds_splittable = _scaffold_folds(scaffolds, k, rep=0)[0] is not None

    for rep in range(repeats):
        tag = f" · partition {rep + 1}/{repeats}" if repeats > 1 else ""
        rf = _random_folds(len(y), k, seed + rep)
        if rep == 0:
            first_rand_folds = rf

        log(f"rung · XGBoost random split{tag}…")
        rep_reported.append(_xgb_oof(X, y, rf, seed))
        log(f"rung · 1-NN Tanimoto random split{tag}…")
        rep_lookup.append(_nn_oof(bvs, y, rf))

        if not scaffolds_splittable:
            continue
        sf, groups = _scaffold_folds(scaffolds, k, rep=rep)
        log(f"rung · XGBoost scaffold split{tag}…")
        rep_survives.append(_xgb_oof(X, y, sf, seed))
        log(f"rung · 1-NN Tanimoto scaffold split{tag}…")
        rep_nn_scaf.append(_nn_oof(bvs, y, sf))
        if rep_survives[-1]["r2"] is not None and rep_nn_scaf[-1]["r2"] is not None:
            rep_learned.append(rep_survives[-1]["r2"] - rep_nn_scaf[-1]["r2"])

    m = _agg(rep_reported)
    ladder.append(_rung("XGBoost", "random split", m, "reported"))
    reported, reported_sd = m["r2"], m["r2_sd"]

    m = _agg(rep_lookup)
    ladder.append(_rung("1-NN lookup", "random split", m, "lookup"))
    lookup_rand, lookup_rand_sd = m["r2"], m["r2_sd"]

    if scaffolds_splittable:
        _, groups = _scaffold_folds(scaffolds, k, rep=0)
        m = _agg(rep_survives)
        ladder.append(_rung("XGBoost", f"scaffold split · {n_scaffolds} series", m, "survives"))
        survives, survives_sd = m["r2"], m["r2_sd"]
        m = _agg(rep_nn_scaf)
        ladder.append(_rung("1-NN lookup", "scaffold split", m, "lookup"))
        nn_scaf, nn_scaf_sd = m["r2"], m["r2_sd"]
    else:
        survives = nn_scaf = survives_sd = nn_scaf_sd = None
        _, groups = _scaffold_folds(scaffolds, k, rep=0)
        ladder.append(_rung("XGBoost", "scaffold split", None, "survives",
                            note="too few scaffold series to split"))

    # ---- harder than scaffold: split on similarity itself (EXPERIMENTAL) ---
    # Distinct Bemis-Murcko cores can still sit next to each other in
    # fingerprint space, so a scaffold-disjoint fold is not necessarily a
    # dissimilar one. This rung cuts on the similarity directly. One partition
    # only: it is the newest and least established thing here, and spending
    # five of them on it would cost more than the number is currently worth.
    log("rung · XGBoost similarity split (experimental)…")
    sim_assign, n_clusters = _similarity_clusters(bvs, sim_cutoff)
    simf = _similarity_folds(sim_assign, n_clusters, k)
    if simf is not None:
        m = _xgb_oof(X, y, simf, seed)
        ladder.append(_rung(
            "XGBoost", f"similarity split · {n_clusters} clusters @ {sim_cutoff:g}",
            m, "survives_similarity",
            note="EXPERIMENTAL — not externally validated; read it as a second "
                 "severity level below the scaffold split, not as a floor"))
        survives_sim = m["r2"]
    else:
        survives_sim = None
        ladder.append(_rung("XGBoost", "similarity split", None, "survives_similarity",
                            note="too few similarity clusters to split"))

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

    # ---- descriptor control ------------------------------------------
    # The headline "% is lookup" is computed on ECFP4. If it only holds under
    # ECFP4 it describes the descriptor rather than the dataset, so the same
    # baseline is recomputed on a path-based fingerprint over the first random
    # partition and the two are reported side by side. One partition, because
    # this is a control on the metric and not a rung in its own right.
    log("control · 1-NN on a second fingerprint…")
    alt = [_fingerprint_alt(sm, n_bits) for sm in d[smiles_col].astype(str).values]
    if all(a is not None for a in alt) and first_rand_folds is not None:
        m_alt = _nn_oof(alt, y, first_rand_folds)
        lookup_alt = m_alt["r2"]
    else:
        lookup_alt = None

    # ---- verdict -----------------------------------------------------
    learned = round(survives - nn_scaf, 3) if (survives is not None and nn_scaf is not None) else None
    learned_sd = _finite(np.std(rep_learned, ddof=1)) if len(rep_learned) > 1 else None
    lookup_pct = (int(round(100 * lookup_rand / reported))
                  if reported is not None and reported >= MIN_REPORTED_R2 else None)

    # Rep 0 is the partition the descriptor control ran on, so the comparison
    # is like for like — a mean over five partitions against a single one
    # would confound the descriptor with the split.
    lookup_rep0 = rep_lookup[0]["r2"] if rep_lookup else None
    descriptor = _descriptor_control(reported, lookup_rep0, lookup_alt)

    verdict = {
        "reported": reported,
        "reported_sd": reported_sd,
        "lookup_random": lookup_rand,
        "lookup_random_sd": lookup_rand_sd,
        "survives_scaffold": survives,
        "survives_scaffold_sd": survives_sd,
        "lookup_scaffold": nn_scaf,
        "lookup_scaffold_sd": nn_scaf_sd,
        "learned_beyond_lookup": learned,
        "learned_beyond_lookup_sd": learned_sd,
        "survives_similarity": survives_sim,
        "temporal": temporal,
        "permutation_floor": floor,
        "lookup_pct_of_reported": lookup_pct,
        "n_repeats": repeats,
        "descriptor_control": descriptor,
        "headline": _headline(reported, lookup_pct, survives, learned, temporal, floor,
                              nn_scaf, reported_sd, survives_sd, survives_sim),
    }

    readout = _readout(reported, lookup_rand, lookup_pct, survives, nn_scaf,
                       learned, temporal, floor,
                       survives_sd=survives_sd, learned_sd=learned_sd,
                       survives_sim=survives_sim, n_clusters=int(n_clusters),
                       sim_cutoff=sim_cutoff)

    specimen = {
        "n_compounds": int(len(y)),
        "n_rows_in": n_rows_in,
        "n_unparseable_dropped": n_bad,
        "n_activity_dropped": n_bad_y,
        "n_dropped_total": n_bad + n_bad_y,
        "dropped_pct": round(100.0 * (n_bad + n_bad_y) / n_rows_in, 1),
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
        "repeats": repeats,
        "similarity_cutoff": sim_cutoff,
        "n_similarity_clusters": int(n_clusters),
        "control_featurisation": f"RDKit topological · {n_bits} bit",
        "columns": {"smiles": smiles_col, "target": y_col, "date": date_col},
    }

    limitations = _limitations(temporal, survives_sim, descriptor, repeats, lookup_pct)

    warn = _drop_warning(n_rows_in, len(y), n_bad, n_bad_y)
    warnings_out = [warn] if warn else []
    if warn:
        log(warn["text"])

    return AutopsyResult(specimen=specimen, ladder=ladder, verdict=verdict,
                         readout=readout, warnings=warnings_out, meta=meta,
                         limitations=limitations)


# ─────────────────────────────────────────────────────────────────────
# small builders (kept pure so tests can assert on them)
# ─────────────────────────────────────────────────────────────────────
def _pm(value, sd) -> str:
    """`0.59 ± 0.03`, or just `0.59` when there was no spread to measure."""
    if value is None:
        return "——"
    return f"{value:.2f}" if sd is None else f"{value:.2f} ± {sd:.2f}"


def _rung(model, cond, metrics, kind, note=None):
    row = {"model": model, "condition": cond, "kind": kind}
    if metrics is None:
        row.update({"r2": None, "r2_sd": None, "rmse": None, "rmse_sd": None,
                    "spearman": None, "spearman_sd": None, "n_reps": 0, "note": note})
    else:
        row.update(metrics)
        row.setdefault("n_reps", 1)      # a rung measured once says so
        row.setdefault("r2_sd", None)
        if note:
            row["note"] = note
    return row


# The band the two fingerprints must agree within for the headline metric to
# count as descriptor-independent. 10 points is the width at which the
# "% is lookup" reading — and the SEVERE/MODERATE/LOW cut that hangs off it —
# would start telling a different story.
DESCRIPTOR_AGREEMENT_PCT = 10


def _lookup_flag(pct) -> str:
    """The one place the similarity-leakage cut lives. _readout paints it and
    the descriptor control re-runs it; neither re-derives the numbers."""
    return "SEVERE" if pct >= 70 else "MODERATE" if pct >= 40 else "LOW"


def _descriptor_control(reported, lookup_ecfp, lookup_alt) -> dict:
    """Does the headline metric survive a change of fingerprint?

    Both baselines are 1-NN on the same first random partition, one on ECFP4
    and one on the RDKit path fingerprint, so the only thing that differs is
    how similarity is defined. Reported as two percentages, their gap, and
    whether the badge would change — a verdict that flips with the descriptor
    is a property of the descriptor.
    """
    if reported is None or reported < MIN_REPORTED_R2:
        return {"available": False,
                "note": f"the control has nothing to compare: the reported score is "
                        f"{'——' if reported is None else format(reported, '.2f')}, "
                        f"below the {MIN_REPORTED_R2:.2f} at which a share-of-score "
                        f"ratio means anything."}
    if lookup_ecfp is None or lookup_alt is None:
        return {"available": False,
                "note": "the control did not run on this dataset — the second "
                        "fingerprint could not be computed for every molecule."}

    pct_a = int(round(100 * lookup_ecfp / reported))
    pct_b = int(round(100 * lookup_alt / reported))
    flag_a, flag_b = _lookup_flag(pct_a), _lookup_flag(pct_b)
    agree = abs(pct_a - pct_b) <= DESCRIPTOR_AGREEMENT_PCT and flag_a == flag_b

    if agree:
        note = (f"Two unrelated fingerprints agree: the lookup reproduces "
                f"{pct_a}% of the reported score on ECFP4 and {pct_b}% on the "
                f"RDKit path fingerprint, both reading {flag_a}. The finding is "
                f"a property of the dataset, not of the descriptor.")
    else:
        note = (f"The two fingerprints disagree: {pct_a}% ({flag_a}) on ECFP4 "
                f"against {pct_b}% ({flag_b}) on the RDKit path fingerprint. "
                f"Treat the headline figure as descriptor-dependent and quote "
                f"both, or neither.")
    return {"available": True, "lookup_pct_ecfp": pct_a, "lookup_pct_alt": pct_b,
            "flag_ecfp": flag_a, "flag_alt": flag_b, "agree": agree,
            "partition": "first random split", "note": note}


def _limitations(temporal, survives_sim, descriptor, repeats, lookup_pct=0) -> list:
    """What this audit does not establish, said out loud.

    An absent test that says nothing reads as a test that passed. Each entry
    is rendered by the report and the UI next to the numbers it qualifies, so
    a reader meets the caveat at the same time as the figure.
    """
    out = []
    if temporal is None:
        out.append({
            "code": "no_time_split", "level": "severe",
            "title": "The most important test could not be run",
            "text": "No assay-date column was supplied, so the temporal "
                    "generalisation test — the gold standard for prospective use "
                    "in QSAR — was not executed on this dataset. Everything above "
                    "measures generalisation to new chemistry, not generalisation "
                    "forward in time; a model can pass every rung here and still "
                    "fail on next quarter's compounds. For the most severe audit "
                    "this tool can perform, supply the assay dates."})
    if lookup_pct is None:
        out.append({
            "code": "no_ratio_denominator", "level": "severe",
            "title": "There is no score for the lower rungs to be a fraction of",
            "text": f"The random-split model scores below R\u00b2 {MIN_REPORTED_R2:.2f}, "
                    f"so the \u201c% is lookup\u201d figure and the descriptor control "
                    f"are not reported: dividing by a score that is essentially zero "
                    f"produces a large number with no meaning. Read the rungs "
                    f"themselves instead. A model that does not clear a random split "
                    f"has no inflated performance to diagnose \u2014 it has no "
                    f"performance."})
    out.append({
        "code": "lookup_pct_not_standard", "level": "note",
        "title": "The \u201c% is lookup\u201d figure is ours, not a literature metric",
        "text": "It is lookup_random / reported: the fraction of the random-split "
                "score that a bare 1-NN Tanimoto lookup reproduces, expressed as a "
                "percentage. It is an ActarusLab interpretive indicator, not a "
                "standard QSAR statistic, and it has no external validation. Use it "
                "to compare rungs within one audit; do not quote it as a "
                "field-recognised measure."})
    if survives_sim is not None:
        out.append({
            "code": "similarity_split_experimental", "level": "note",
            "title": "The similarity split is experimental",
            "text": "Sphere-exclusion clustering on Tanimoto at a fixed cutoff, "
                    "split so that training and test clusters are dissimilar. It is "
                    "reported beside the scaffold split, not instead of it, because "
                    "recent work finds Bemis-Murcko scaffold splits still optimistic. "
                    "The cutoff is a convention and this protocol has not been "
                    "externally validated: read it as a second severity level, not "
                    "as the true floor, and have a domain expert confirm it before "
                    "relying on the number."})
    if descriptor.get("available") and not descriptor.get("agree"):
        out.append({
            "code": "descriptor_disagreement", "level": "severe",
            "title": "The headline figure moves with the fingerprint",
            "text": descriptor["note"]})
    if repeats < 2:
        out.append({
            "code": "single_partition", "level": "note",
            "title": "No error bands in this run",
            "text": "This audit ran one partition per rung, so no spread was "
                    "measured and the scores carry no band. Differences between "
                    "rungs smaller than a few hundredths cannot be distinguished "
                    "from the choice of split."})
    return out


def _headline(reported, lookup_pct, survives, learned, temporal, floor, nn_scaf=None,
              reported_sd=None, survives_sd=None, survives_sim=None):
    if reported is None or reported < MIN_REPORTED_R2:
        return (f"Model shows no usable predictive signal even on a random split "
                f"(R\u00b2 {_pm(reported, reported_sd)}). There is no reported "
                f"performance here to be inflated, so the share-of-score figures are "
                f"not reported: read the rungs themselves.")
    parts = []
    if lookup_pct is not None:
        parts.append(f"{lookup_pct}% of the reported R² {_pm(reported, reported_sd)} is "
                     f"reproducible by a pure nearest-neighbour lookup — recognition of "
                     f"known analogues, not learned SAR.")
    if survives is not None:
        parts.append(f"On disjoint chemical series performance holds at "
                     f"{_pm(survives, survives_sd)}.")
    if survives_sim is not None:
        parts.append(f"Split on similarity instead of scaffold — harsher, and "
                     f"experimental — it reads {survives_sim:.2f}.")
    if learned is not None:
        if nn_scaf is not None and nn_scaf < 0:
            # subtracting a baseline that scores below zero inflates the figure,
            # so quoting it without that caveat would oversell the model
            parts.append(f"Its apparent contribution over the lookup, {learned:.3f}, is inflated: "
                         f"the lookup itself scores {nn_scaf:+.2f} on new scaffolds, worse than "
                         f"predicting the mean, so the gap measures the baseline's failure rather "
                         f"than the model's skill.")
        else:
            parts.append(f"Beyond the lookup baseline, the model's own contribution is {learned:.3f}.")
    if temporal is not None:
        parts.append(f"Forward in time it holds {temporal:.2f}.")
    if floor is not None and floor > 0.05:
        parts.append(f"WARNING: permutation floor is {floor:.2f} (> 0) — the pipeline itself may be leaking; "
                     f"fix featurisation/splitting before trusting any number above.")
    return " ".join(parts)



def precheck(df: pd.DataFrame, smiles_col: str, y_col: str,
             date_col: Optional[str] = None) -> None:
    """Every validation that costs nothing, so a caller can refuse a file in
    milliseconds instead of queueing an audit it is going to reject anyway.

    run_autopsy calls this itself and never assumes a caller did. Each message
    is written for whoever uploaded the file: what is wrong, and what to do.
    The bar is the missing-column case, which names both what is absent and
    what is present.
    """
    for col in (smiles_col, y_col):
        if col not in df.columns:
            raise AutopsyError(f"column '{col}' not found. Available: {list(df.columns)}")
    if date_col and date_col not in df.columns:
        raise AutopsyError(f"date column '{date_col}' not found. Available: {list(df.columns)}")

    n_rows_in = int(len(df))
    if n_rows_in == 0:
        raise AutopsyError("the file has no rows — there is nothing to audit.")

    raw_y = df[y_col]
    y_num = pd.to_numeric(raw_y, errors="coerce")

    # A column of words fails every row for one reason, and blaming the rows
    # sends the reader to inspect their SMILES, which were fine.
    if int(y_num.isna().sum()) == n_rows_in:
        seen = [repr(v) for v in pd.Series(raw_y).dropna().astype(str).unique()[:3]]
        holds = f" It holds {', '.join(seen)}." if seen else " It is empty."
        raise AutopsyError(
            f"the activity column '{y_col}' contains no numeric values.{holds} "
            f"An audit needs a number per molecule to predict — an IC50, a pIC50, "
            f"a measured property. Point --y at the column that holds it.")

    # Constant targets are the dangerous case: nothing raises, every rung scores
    # a perfect 1.00 including the permutation control, and the audit reads as a
    # triumph. Refuse before computing anything.
    y = y_num.dropna().astype(float).values
    if len(y):
        n_distinct = int(len(np.unique(y)))
        sd = float(np.std(y))
        if n_distinct == 1:
            raise AutopsyError(
                f"the activity column '{y_col}' holds one repeated value ({y[0]:g}) — "
                f"zero variance, so there is nothing to predict and no model can be scored "
                f"against it. Check that --y points at the measurement rather than a label "
                f"or a constant.")
        if sd < MIN_TARGET_SD:
            raise AutopsyError(
                f"the activity column '{y_col}' barely varies: standard deviation {sd:.2g} "
                f"across {n_distinct} distinct values. There is not enough variation to score "
                f"a model against, and any R² would be floating-point noise.")


def _drop_warning(n_rows_in, n_kept, n_bad_smiles, n_bad_y):
    """Rows silently discarded are the difference between auditing a file and
    auditing whatever survived of it. Say so, and say it louder past a fifth."""
    n_dropped = n_bad_smiles + n_bad_y
    if not n_dropped:
        return None
    pct = 100.0 * n_dropped / n_rows_in
    bits = []
    if n_bad_smiles:
        bits.append(f"unreadable SMILES ({n_bad_smiles})")
    if n_bad_y:
        bits.append(f"missing or non-numeric activity ({n_bad_y})")
    text = (f"{n_dropped} of {n_rows_in} rows dropped — {', '.join(bits)}. "
            f"The audit below covers the {n_kept} that remain.")
    if pct > DROP_SEVERE_PCT:
        text += (f" That is {pct:.0f}% of the file, so these numbers describe a "
                 f"subset and may not represent your dataset.")
    return {"level": "severe" if pct > DROP_SEVERE_PCT else "note",
            "text": text, "n_dropped": n_dropped, "n_rows_in": n_rows_in,
            "pct": round(pct, 1)}


# ─────────────────────────────────────────────────────────────────────
# how "learned structure" is read
# ─────────────────────────────────────────────────────────────────────
# learned = survives - lookup_scaffold, and that subtraction only means
# something while the lookup baseline itself does. Once lookup_scaffold goes
# negative the baseline is doing worse than predicting the mean, and
# subtracting it *inflates* `learned`: the model looks good because the
# baseline collapsed, not because it found structure. Measured on four real
# sets, the same number tells four different stories:
#
#   Lipophilicity  learned 0.55   lookup +ve    beats a sound baseline
#   ESOL           learned 0.64   lookup -0.40  artefact of the baseline failing
#   CHEMBL233      learned 0.24   lookup +ve    real advantage, modest
#   BACE-1         learned 0.146  lookup +0.45  thin
#
# So the flag reads two dimensions. The 0.2 cut is kept from the earlier
# calibration on BACE, where 0.146 is honestly thin against a reported 0.71.
# 0.4 is where the model's own contribution stops being a minority share of a
# typical reported score and starts being the bulk of it.
#
# These four names are the only vocabulary; report.py and ui_connector.jsx
# colour them but never re-derive them.
LEARNED_NET = 0.4       # at or above, on a sound baseline: real structure added
LEARNED_THIN = 0.2      # at or below: the addition is thin


def _learned_flag(learned: float, lookup_scaffold) -> str:
    """ARTIFACT | NET | MARGINAL | THIN — see the cuts above."""
    if lookup_scaffold is not None and lookup_scaffold < 0:
        return "ARTIFACT"
    if learned >= LEARNED_NET:
        return "NET"
    if learned > LEARNED_THIN:
        return "MARGINAL"
    return "THIN"


def _learned_note(flag: str, learned: float, survives, lookup_scaffold) -> str:
    if flag == "ARTIFACT":
        base = (f"Read this as a warning, not an achievement. The lookup baseline scores "
                f"{lookup_scaffold:+.2f} on new scaffolds — worse than predicting the mean — so "
                f"subtracting it inflates this figure. The model is not learning a lot; the "
                f"similarity baseline is unreliable on new chemistry for this dataset.")
        if survives is not None:
            base += f" What the model actually holds on new series is the scaffold rung, {survives:.2f}."
        return base
    if flag == "NET":
        return ("Scaffold performance minus the scaffold-split lookup, against a baseline that "
                "still works. This is structure the model added over averaging its nearest analogues.")
    if flag == "MARGINAL":
        return ("Scaffold performance minus the scaffold-split lookup. A real advantage over "
                "copying the nearest analogues, but a modest one.")
    if learned < 0:
        return ("Negative: on new chemical series the lookup baseline beats the model. Averaging "
                "the nearest analogues would serve you better than this model does.")
    return ("Scaffold performance minus the scaffold-split lookup. The only structure the model "
            "added beyond averaging its nearest analogues, and there is little of it.")

def _readout(reported, lookup_rand, lookup_pct, survives, nn_scaf, learned, temporal, floor,
             survives_sd=None, learned_sd=None, survives_sim=None, n_clusters=None,
             sim_cutoff=SIM_CUTOFF):
    cards = []
    # similarity leakage
    if lookup_pct is not None:
        cards.append({"signal": "Similarity leakage", "flag": _lookup_flag(lookup_pct),
                      "value_pct": lookup_pct,
                      "note": "Share of the reported score a bare nearest-neighbour lookup "
                              "reproduces — computed as lookup_random / reported. High means "
                              "the model is rewarded for recognising known analogues. This is "
                              "an ActarusLab interpretive indicator, not a standard QSAR "
                              "metric, and it carries no external validation."})
    # scaffold transfer
    if survives is not None:
        cards.append({"signal": "Scaffold transfer", "flag": "PARTIAL", "value": survives,
                      "sd": survives_sd,
                      "note": "Performance on genuinely new chemical series — what generalises beyond the training scaffolds."})
    # similarity transfer — harder than scaffold, and not validated
    if survives_sim is not None:
        cards.append({"signal": "Similarity transfer", "flag": "EXPERIMENTAL",
                      "value": survives_sim,
                      "note": f"Same model, split so training and test clusters sit below "
                              f"Tanimoto {sim_cutoff:g} of each other "
                              f"({n_clusters} clusters). Harsher than the scaffold split, "
                              f"because distinct scaffolds can still be near neighbours. "
                              f"Experimental and not externally validated — a second severity "
                              f"level, not a floor."})
    # learned structure — two dimensions: how much, and whether the baseline
    # it is measured against was sound. See the cuts above.
    if learned is not None:
        flag = _learned_flag(learned, nn_scaf)
        cards.append({"signal": "Learned structure", "flag": flag, "value": learned,
                      "sd": learned_sd,
                      "note": _learned_note(flag, learned, survives, nn_scaf)})
    # temporal
    cards.append({"signal": "Temporal test",
                  "flag": "N/A" if temporal is None else "TESTED",
                  "value": temporal,
                  "note": "Generalisation forward in time. Activates when assay dates are supplied — the split a random fold hides entirely."
                          if temporal is not None else
                          "Not run: no assay dates in this file. This is the gold standard "
                          "for prospective use, and its absence is a gap in the audit rather "
                          "than a pass — see the declared limitations."})
    # permutation floor
    if floor is not None:
        flag = "CLEAN" if floor <= 0.05 else "LEAK"
        cards.append({"signal": "Permutation floor", "flag": flag, "value": floor,
                      "note": "Shuffled-target control. Should collapse toward zero; anything well above signals a pipeline leak."})
    return cards
