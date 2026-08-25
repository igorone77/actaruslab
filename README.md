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
`random 0.72 → lookup 0.58 → scaffold 0.62 → scaffold-lookup 0.36 → floor −0.22`.
80% of the reported score is a lookup; 0.26 is learned beyond it.
`bace_report.html` is that run; `tests/test_bace_smoke.py` re-derives it in CI.

> **⚠ The scaffold rungs are not reproducible.** `seed` does not reach them.
> `_scaffold_folds` delegates to scikit-learn's `GroupKFold`, which orders
> series by size using `np.argsort(...)` — an *unstable* sort — and 200 of
> BACE's 377 series are tied at one compound. Every tie-breaking is a valid
> "largest series first" order, each gives a different partition into folds
> of the same size, and which one you get depends on the scikit-learn version
> and on the CPU (numpy 2.x selects a SIMD sort path from the host's
> instruction set). Same code, same data, k=5, seed=0:
>
> | tie order | scaffold | scaffold-lookup | learned beyond lookup |
> |---|---|---|---|
> | ascending group id | 0.611 | 0.414 | 0.197 |
> | descending group id | 0.633 | 0.461 | 0.172 |
> | the box that produced `bace_report.html` | 0.622 | 0.365 | **0.257** |
> | a GitHub Actions runner | — | 0.421 | — |
>
> So the headline lands anywhere in ~0.17–0.26 depending on the machine, and
> **0.26 is one draw, not the number**. The random-split, temporal and
> permutation rungs are unaffected — `KFold` takes an explicit `random_state`.
> No dependency pin fixes this; a client re-running the audit on their own
> hardware will not reproduce the report you sent them.
>
> **The fix** is to stop delegating the split — order the series in-engine
> with a deterministic tie-break, so `seed` genuinely covers it:
>
> ```python
> def _scaffold_folds(scaffolds, k):
>     groups = pd.factorize(pd.Series(scaffolds))[0]
>     k_eff = min(k, len(set(groups)))
>     if k_eff < 2:
>         return None, groups
>     counts = np.bincount(groups)
>     order = np.lexsort((np.arange(len(counts)), -counts))   # size desc, id asc
>     weight, group_to_fold = np.zeros(k_eff), np.zeros(len(counts), dtype=int)
>     for g in order:                       # largest series into the lightest fold
>         f = int(np.argmin(weight)); weight[f] += counts[g]; group_to_fold[g] = f
>     per_sample = group_to_fold[groups]
>     return ([(np.where(per_sample != f)[0], np.where(per_sample == f)[0])
>              for f in range(k_eff)], groups)
> ```
>
> That is a behaviour change — it fixes one partition for good, and the
> published numbers above move to the first row of the table (`scaffold 0.611
> → scaffold-lookup 0.414`, learned 0.197), on every machine. It is left
> undone deliberately: re-running the BACE audit and reissuing
> `bace_report.html` is a call for whoever owns the client-facing claims.
> `tests/test_scaffold_determinism.py` pins the defect meanwhile and will
> start failing the moment it is fixed.

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
(`reported ≈ 0.72`, `lookup_pct ≈ 80`, `floor < 0`) plus surface smoke tests for
the CLI, the HTML report and the API, on every push. Running that guard on a
second machine is what surfaced the scaffold-split defect above — the repo's
first finding was about the auditor, not the model.

**Open — item 0.** Decide on the scaffold-split fix before anything else here
ships to a client, since it changes every scaffold number the tool reports.

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
