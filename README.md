# MODEL AUTOPSY — engine

Forensic validation for QSAR / activity-prediction models. Point it at a
dataset of molecules and it measures **how much of a model's apparent
performance survives an honest split** — separating learned chemistry from
memorised chemical neighbourhood.

It does not sell prediction. It reports surviving structure.

---

## What it does

Given `SMILES + activity` (and optionally an assay date), it climbs a
**Leakage Ladder** — each rung a harder-to-fool validation than the last —
and reports pooled out-of-fold R² at each:

| rung | what it measures |
|---|---|
| XGBoost · random split | the optimistic number people report |
| 1-NN Tanimoto · random | how much of that is a pure similarity lookup |
| XGBoost · scaffold split | what survives a genuinely new chemical series |
| 1-NN Tanimoto · scaffold | the lookup floor on new series |
| XGBoost · temporal split | generalisation forward in time *(if dates given)* |
| permutation (shuffled y) | the honesty floor — must collapse toward 0 |

The headline numbers it extracts:

- **% of reported score reproducible by lookup** — how much is memorisation
- **survives scaffold split** — what transfers to novel chemistry
- **learned beyond lookup** = scaffold − scaffold-lookup — the model's own contribution
- **temporal** — does it predict the *next* campaign, or ride historical trend
- **permutation floor** — is the pipeline itself leaking

Nothing is model magic: RDKit + scikit-learn + XGBoost do the numbers. The
engine only decides how to *validate*, and reports what it finds.

---

## Install

```bash
pip install -r requirements.txt          # rdkit, xgboost, sklearn, scipy, pandas (+ fastapi for the API)
```

## Run — CLI (the one-time audit)

```bash
python -m autopsy.cli data.csv --smiles smiles --y pIC50 \
       [--date document_year] [--out report.html] [--json result.json]
```

Prints the verdict + ASCII ladder to the terminal; with `--out` writes a
self-contained NEUTRA-styled HTML report you can hand to a client.

**Verified on BACE-1** (1513 compounds, 377 scaffold series, `k=5`, `seed=0`):
`random 0.72 → lookup 0.58 → scaffold 0.59 → scaffold-lookup 0.45 → floor −0.22`.
80% of the reported score is a lookup; **0.147** is learned beyond it
(`scaffold 0.595 − scaffold-lookup 0.448`). `bace_report.html` is that run;
`tests/test_bace_smoke.py` re-derives it in CI.

Note the ladder is no longer monotonically descending: the random-split lookup
(0.58) now sits *above* what survives a new chemical series (0.59) by a hair.
That shape is the finding, not a glitch — on novel chemistry the model barely
matches a random-split lookup table, and beats the scaffold-split lookup by
only 0.147.

> **Determinism.** The scaffold split is a function of the molecules alone.
> `_scaffold_folds` assigns group ids from `sorted(set(scaffolds))` — scaffold
> content, not order of appearance — and fills folds largest-series-first into
> the lightest fold, ties broken by group id and by fold index. Nothing
> consults an unstable sort, so the same compounds give the same folds on any
> machine and under any row ordering.
>
> This replaced a delegation to scikit-learn's `GroupKFold`, which ordered
> series by size with `np.argsort(...)` — an *unstable* sort — where 200 of
> BACE's 377 series tie at one compound. Every tie-breaking was a valid
> ordering and each produced a different partition into folds of identical
> size, so `scaffold-lookup` read 0.365 on one machine and 0.421 on another
> with the same library versions. `seed` never reached that decision. No
> dependency pin fixes it; only owning the assignment does.
>
> **Still outstanding — the scores are not yet order-invariant.** The folds
> are, but two rungs still read the row order *inside* a fold:
>
> * `_nn_oof` breaks ties with `np.argmax`, i.e. by position. On BACE's
>   scaffold folds 99 of 1513 test molecules have a tied nearest neighbour and
>   83 of those tie between neighbours with *different* activities, so the
>   prediction depends on which one comes first.
> * XGBoost's `subsample`/`colsample_bytree` draw against row positions.
>
> Shuffling the input rows moves `scaffold-lookup` 0.448 → 0.427 with the fold
> membership provably unchanged. Fixing it means breaking 1-NN ties on a
> content key (canonical SMILES) rather than position, and averaging or
> ordering the model draw. Until then the audit is reproducible for a *given*
> CSV, not for the same molecules in a different order.
>
> The XGBoost rungs also carry ~±0.002 across xgboost builds; the 1-NN rungs
> are exact. `tests/test_scaffold_determinism.py` guards the split itself.

## Run — API (what the NEUTRA UI calls)

```bash
uvicorn autopsy.api:app --reload            # http://127.0.0.1:8000
```

- `GET  /health`
- `POST /autopsy/csv`      — multipart file + `smiles`,`y`,`date?`,`k` form fields
- `POST /autopsy/records`  — JSON `{records:[{smiles,y,date?}], k, has_date}`

Both return the full result object (`specimen`, `ladder`, `verdict`,
`readout`, `meta`) — the same structure the CLI and HTML report consume.

---

## Architecture

```
autopsy/
  engine.py   ← run_autopsy(df, smiles, y, date?) -> AutopsyResult   (the one function; pure, deterministic)
  cli.py      ← terminal audit + report/JSON output
  report.py   ← AutopsyResult -> NEUTRA-styled standalone HTML
  api.py      ← FastAPI wrapper (same engine, HTTP surface)
```

One engine, two surfaces. The CLI is the one-time audit; the API is how the
prototype UI stops replaying BACE and starts computing on real uploads.

---

## Status

**Done — repo + CI.** `.github/workflows/ci.yml` runs the BACE regression audit
plus surface smoke tests for the CLI, the HTML report and the API, on every
push. Running that guard on a second machine is what surfaced the scaffold-split
defect — the repo's first finding was about the auditor, not the model.

**Done — deterministic scaffold split.** Fixed in `_scaffold_folds`, with
`bace_report.html` and the numbers above reissued from it.

**Open — order-invariant scores.** 1-NN tie-breaking and XGBoost subsampling
still read row order; see the determinism note above.

## Next steps — the last mile

To become the product Strikeon uses:

1. **Wire the UI to the API.** The prototype (`model_autopsy.jsx`) currently
   holds constants. Replace them with a `fetch('/autopsy/csv', …)` on file
   upload and render `result.ladder` / `result.verdict` / `result.readout`.
   A reference `useAutopsy()` hook is in `ui_connector.jsx`.
2. **Deploy the API** somewhere the UI can reach (a small container). Lock
   CORS to the UI origin — it's dev-open (`*`) right now.
3. **Harden for their data.** Real campaign CSVs are messier than BACE:
   mixed activity units (nM vs µM), censored values (`>10000`), salts in the
   SMILES, duplicate measurements. Add a normalisation pass before the engine
   (unit harmonisation, `>`/`<` handling, desalting, aggregating replicate
   activities) — each is a known QSAR-data pitfall and each is a place a naive
   pipeline leaks.
4. **Performance.** 1-NN Tanimoto is O(n²); fine to ~5k compounds, slow beyond.
   For larger sets, switch the lookup baseline to an approximate NN index
   (e.g. FAISS on folded fingerprints) — the *result* is what matters, the
   exact-NN guarantee isn't.

Items 1–2 make it a running service. Items 3–4 make it trustworthy on data
that isn't a clean public benchmark.
