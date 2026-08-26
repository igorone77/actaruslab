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
> Cross-machine check: this box and a GitHub Actions runner (different CPU,
> different Python patch) now agree digit-for-digit on all seven headline
> numbers — `reported=0.722 lookup=0.576 survives=0.595 nn_scaffold=0.448
> floor=-0.224 lookup%=80 learned=0.147`. Before the fix the same two machines
> disagreed on `nn_scaffold` by 0.056. The smoke-test bands stay slightly wide
> on the XGBoost rungs because a *different xgboost build* still shifts them by
> ~0.002; the 1-NN rungs are exact.
> `tests/test_scaffold_determinism.py` guards the split itself.

## Run — the app (browser)

```bash
pip install -r requirements.txt
uvicorn autopsy.api:app --reload            # open http://127.0.0.1:8000
```

Drop a CSV on **LOAD CSV**, name the SMILES and activity columns, hit **RUN
AUTOPSY**. The engine runs server-side and the ladder, dials, readout and
specimen tiles all fill from the result — roughly 20 s for 1500 compounds.
With no file loaded the page sits on the BACE-1 benchmark and **REPLAY
BENCHMARK** just re-animates it.

The same app serves the API, so the page and its backend share an origin and
nothing needs CORS. The CORS middleware only matters for a UI hosted
elsewhere, and is still dev-open (`*`) — lock it to the UI origin before
putting this on a public host.

### Endpoints

- `GET  /health`
- `POST /autopsy/jobs`     — multipart upload → `202 {job_id}`; the audit runs off
  the request. What the UI uses, and what a hosted deployment needs.
- `GET  /autopsy/jobs/{id}` — `{status, progress, elapsed, result?, error?}`
- `POST /autopsy/csv`      — multipart file + `smiles`,`y`,`date?`,`k` form fields,
  answered synchronously
- `POST /autopsy/records`  — JSON `{records:[{smiles,y,date?}], k, has_date}`
- `GET  /`                 — the UI · `GET /docs` — OpenAPI

Both autopsy endpoints return the full result object (`specimen`, `ladder`,
`verdict`, `readout`, `meta`) — the same structure the CLI and HTML report
consume.

### Rebuilding the front end

`web/static/app.js` is committed so the app runs with Python alone. After
editing `model_autopsy.jsx` or `ui_connector.jsx`:

```bash
npm install && npm run build     # or: npm run watch
```

The bundle is an IIFE, not an ES module, so `web/static/index.html` also
opens straight off the filesystem — browsers refuse module scripts from
`file://`, and the showcase has to survive being double-clicked.

### One page, two jobs

The same bundle is the public showcase and the working tool; which one it is
gets decided at load, not at build.

| | no engine reachable | engine reachable |
|---|---|---|
| where | Netlify, a `file://` double-click, any static host | `uvicorn autopsy.api:app` |
| badge | `BENCHMARK` | `ENGINE READY`, then `LIVE` |
| upload | withdrawn, with a line saying what to run | active; audits the CSV you pick |
| numbers | the BACE-1 audit, precomputed | whatever your data gives |

`engineReachable()` probes `./health` once on mount. `API_BASE` is relative
to the page — never an absolute URL, never a hardcoded port — so the same
file works on any port uvicorn is given, under a sub-path on a static host,
and from the filesystem. A host page can override it with
`window.__AUTOPSY_API__` to point at an API served elsewhere.

**Publishing the showcase.** `netlify.toml` sets `publish = "web/static"`, so
a repo-linked Netlify site needs no build step — the bundle is committed. Or
drag the `web/static` folder onto Netlify's deploy area.

### Container

```bash
docker build -t model-autopsy . && docker run -p 8000:8000 model-autopsy
```

No Node in the image — it copies the built bundle.

### Putting it on a public host

Six environment variables, all safe by default, so deployment is config
rather than a code edit:

| variable | default | what it does |
|---|---|---|
| `AUTOPSY_ALLOWED_ORIGINS` | unset | comma-separated origins allowed to call the API cross-site. Unset means same-origin only, and no CORS headers are sent at all — correct when this app serves its own UI. |
| `AUTOPSY_MAX_UPLOAD_MB` | `25` | larger uploads get 413 |
| `AUTOPSY_MAX_ROWS` | `5000` | wider datasets get 413 |
| `AUTOPSY_WORKERS` | `1` | audits running at once; each saturates a CPU |
| `AUTOPSY_QUEUE_DEPTH` | `8` | jobs allowed to wait; beyond it, submits get 429 |
| `AUTOPSY_JOB_TTL` | `3600` | seconds a finished job stays readable |

**The audit runs off the request.** It has to: measured here, 1513 compounds
take 19.2 s, and the lookup rung is O(n²), so 3000 compounds is roughly four
times that wait rather than twice. Platforms cut requests off well before
that — 30 s on some, 60 s on others, 100 s at a Cloudflare proxy. So the UI
submits a job and polls:

```
POST /autopsy/jobs        -> 202 {job_id, status, rows}      (0.4 s)
GET  /autopsy/jobs/{id}   -> {status, progress, elapsed, result?, error?}
```

Validation stays on the request — a wrong column name, an oversized file or
too many rows come back immediately as 4xx, so only the slow part is
deferred. `progress` carries the engine's own log line (`rung · 1-NN
Tanimoto scaffold split…`), which the UI shows while it waits.

`/autopsy/csv` and `/autopsy/records` still answer synchronously and are
kept for scripting and local use, where no proxy is going to hang up.

> Jobs live in the serving process's memory. They do not survive a restart
> and are not shared between replicas — the right trade for a single
> container, the wrong one for a horizontally scaled deployment, which would
> need Redis or a database behind the same two endpoints.

Once it is hosted, a subdomain is the natural shape — `autopsy.example.org`
pointed at the container — because the app is a Python server while a
marketing site usually is not. Serving it under a path on the main domain
means putting a reverse proxy in front of both.

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

**Done — the app runs in a browser.** `autopsy.api` serves the NEUTRA UI at
`/`, so one command gives you a page that takes a CSV upload and renders a
real audit. Same origin as the API, no CORS, no second dev server.

**Done — deployable.** CORS closed by default, uploads capped, and the audit
moved off the request onto a job queue, so it no longer dies at a platform's
request timeout.

**Open — order-invariant scores.** 1-NN tie-breaking and XGBoost subsampling
still read row order; see the determinism note above.

**Open — nothing authenticates a caller.** Anyone who can reach the URL can
queue audits. Fine behind a private host or a proxy that handles auth; decide
before it is public.

## Next steps — the last mile

To become the product Strikeon uses:

1. **Deploy it.** `Dockerfile` builds the whole thing; it needs a host and a
   DNS record. CORS, upload caps and the job queue are done — what is left is
   choosing where it runs and deciding whether it should be open to anyone
   with the URL, since nothing here authenticates a caller.
2. **Harden for their data.** Real campaign CSVs are messier than BACE:
   mixed activity units (nM vs µM), censored values (`>10000`), salts in the
   SMILES, duplicate measurements. Add a normalisation pass before the engine
   (unit harmonisation, `>`/`<` handling, desalting, aggregating replicate
   activities) — each is a known QSAR-data pitfall and each is a place a naive
   pipeline leaks.
3. **Performance.** 1-NN Tanimoto is O(n²); fine to ~5k compounds, slow beyond.
   For larger sets, switch the lookup baseline to an approximate NN index
   (e.g. FAISS on folded fingerprints) — the *result* is what matters, the
   exact-NN guarantee isn't.

Item 1 puts it on a URL. Items 2–3 make it trustworthy on data that isn't a
clean public benchmark.
