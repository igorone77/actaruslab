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

### What an audit gives back

**The engine is free.** Anyone can upload a dataset and run the full ladder —
no key, no account, no subscription. What comes back is split in two, and the
split is the product:

| | free — everyone | reserved — on request |
|---|---|---|
| the verdict | **inflation `+X%`**: how far the reported score sits above what survives an honest split | |
| the diagnosis | | where the leakage originates, which scaffold series carry it, the per-fold breakdown, the reproducible validation report, what to change |

The audit runs whole either way. The reserved half is simply never
serialised: `autopsy/tiers.py::public_view` builds the free response key by
key rather than filtering the full one, so there is nothing in the payload to
inspect, decode or reconstruct — and a field added to the engine tomorrow
cannot leak through it by default. `GET /autopsy/demo` is the exception on
purpose: the BACE-1 audit is published in full, so a visitor can see exactly
what the reserved tier contains before asking for it on their own data.

Free does not mean public. A result belongs to whoever submitted it, and the
submit hands back two claims on it: a `job_token` in the response body, for
scripts and the CLI, and an HttpOnly `autopsy_session` cookie, which a browser
presents on its own without the page having to remember anything. Either one
reads the result back; polling with neither answers **404** — the same answer
as a job that never existed, because a 403 would confirm the id belongs to
someone. The check runs in every configuration and is not connected to
billing.

The cookie is there because the token alone was too brittle to be the only
claim: it lived in one closure in one tab, so a reload lost a running audit, a
second tab could not see it, and a page that did not know to send the header
was told its live job had expired. It does not widen who can read a result —
HttpOnly, SameSite, and Secure wherever `AUTOPSY_PUBLIC_URL` is https — only
which of that browser's own audits it can read.

Ask for the full diagnosis at **actaruslab@proton.me**.

### Subscription — present, switched off

€199/month, 20 audits per billing cycle, blocked until renewal past that. All
of it is in the repository and none of it runs: Checkout, the
signature-verified webhook, key issuance, the quota counter, the customer
portal, the 402. One flag decides.

```bash
AUTOPSY_PAYWALL_ENABLED=false   # default — free showcase, no 402, no checkout link
AUTOPSY_PAYWALL_ENABLED=true    # the paid product, exactly as it was
```

With it on, everything under `/autopsy/` that computes on your data is behind
the paywall again and answers **402** with the checkout link; a live
subscription buys the reserved tier. `/health` and `GET /autopsy/demo` stay
free in both positions.

The flag, and not the presence of `STRIPE_SECRET_KEY`, is what decides. That
coupling made the paywall a side effect of configuration — set a key to test a
webhook and the engine silently locked. Turning a deployment paid is now one
explicit decision in one variable. Setting the flag *without* a Stripe key is
refused with a 503 rather than served free, so a deployment cannot believe it
is charging while it is not.

Stripe Checkout takes the payment, a signature-verified webhook grants access,
and the Stripe customer portal handles cancellation. A subscription issues one
API key, shown once and stored only as a SHA-256 hash, presented as
`Authorization: Bearer ma_…` or the `autopsy_key` cookie. With the paywall on,
that key is a third way to own a job result, alongside the `job_token` and
the session cookie.

Subscriber records live in SQLite (`AUTOPSY_DB`) rather than in memory: a
restart may forget a running audit, but it may never forget who paid. **Point
it at a persistent volume** — on an ephemeral filesystem a redeploy wipes
paying subscribers. A scaled deployment wants Postgres behind the same
functions.

An audit is spent when the job is accepted and refunded if the failure is
ours, so nobody loses one of their 20 to a bug on our side.

- `docs/BILLING_SETUP.md` — what to do in Stripe and in the deploy, in order
- `docs/TERMS.md` — draft terms, with the clauses needing a lawyer marked

### What gets refused, and what gets flagged

A refusal names what is wrong and what to do, never an exception class or a
status code. Everything cheap enough to see without computing is checked
before the audit is queued, so a bad column costs a second rather than a job:

| input | outcome |
|---|---|
| column not in the file | names it and lists what *is* there |
| activity column holds text | names the column and shows what it holds — not "0 valid rows" |
| activity is one repeated value | refused: zero variance, nothing to predict |
| activity barely varies | refused: any R² would be floating-point noise |
| fewer than 40 usable rows | counts them and says why they were lost |
| empty file | says so |

The constant-activity case is why this matters. It did not raise: it produced
a complete audit reading **R² 1.00 on every rung**, permutation control
included, and the CLI printed it. The API only escaped because the NaN
correlation hit a JSON serialiser that refuses NaN — an accident, surfacing as
an unexplained 500. Metrics that cannot be computed now come back as `null`
rather than NaN, so that whole class of failure is gone.

Rows the engine discards are reported rather than absorbed. `specimen` counts
them (`n_rows_in`, `n_unparseable_dropped`, `n_activity_dropped`,
`dropped_pct`) and `result.warnings` carries a sentence the CLI, the HTML
report and the UI all display above the numbers — louder past 20%, where the
audit describes a subset rather than the file. An empty SMILES counts as
unreadable: RDKit parses it into a valid molecule with no atoms, whose
all-zero fingerprint would otherwise join the audit as a phantom compound.

### Reading "learned beyond lookup"

That subtraction only measures the model while the baseline it is subtracting
still works. Once the scaffold-split lookup goes **negative** it is scoring
worse than predicting the mean, and subtracting it *inflates* the difference —
the model looks strong because the baseline collapsed. Four real datasets put
four different stories behind similar-looking numbers, so the badge reads two
dimensions rather than one:

| badge | when | reading |
|---|---|---|
| `ARTIFACT` | scaffold-lookup < 0, any gain | the gain is the baseline failing, not structure learned. Read the scaffold rung instead. |
| `NET` | sound baseline, learned ≥ 0.4 | real structure, beyond what averaging analogues gives |
| `MARGINAL` | sound baseline, 0.2 < learned < 0.4 | a real advantage, a modest one |
| `THIN` | sound baseline, learned ≤ 0.2 | little added; negative means the lookup beats the model |

Measured: Lipophilicity 0.55 on a sound baseline → `NET`. ESOL 0.64 against a
−0.40 baseline → `ARTIFACT`. CHEMBL233 0.24 → `MARGINAL`. BACE-1 0.146 →
`THIN`. The 0.2 cut is where 0.146 is honestly thin against a reported 0.71;
0.4 is where the model's own contribution stops being a minority share of a
typical reported score.

The cuts live in `LEARNED_NET` / `LEARNED_THIN` in `engine.py` and nowhere
else — `report.py` and `ui_connector.jsx` colour the names the engine chose,
they never re-derive them, so the three surfaces cannot drift apart.
`tests/test_learned_flag.py` asserts all four cases plus the two real datasets
in the repo.

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
`random 0.71 → lookup 0.57 → scaffold 0.60 → scaffold-lookup 0.45 → floor −0.22`.
81% of the reported score is a lookup; **0.146** is learned beyond it
(`scaffold 0.597 − scaffold-lookup 0.451`). `bace_report.html` is that run;
`tests/test_bace_smoke.py` re-derives it in CI.

Note the ladder is not monotonically descending: what survives a genuinely new
chemical series (0.597) sits a hair *above* the random-split lookup (0.572), so
the line ticks up at the third rung. That shape is the finding, not a glitch —
on novel chemistry the model barely matches a lookup table built on a random
split, and beats the scaffold-split lookup by only 0.146.

> **Determinism.** The audit is a function of the molecule set. Same
> compounds, same numbers — on any machine, in any file order, run after run.
> Three decisions used to leak something else in, and all three are closed.
>
> **The split.** `_scaffold_folds` assigns group ids from
> `sorted(set(scaffolds))` — scaffold content, not order of appearance — and
> fills folds largest-series-first into the lightest fold, ties broken by
> group id then fold index. This replaced a delegation to scikit-learn's
> `GroupKFold`, which ordered series by size with `np.argsort(...)`, an
> *unstable* sort, where 200 of BACE's 377 series tie at one compound. Every
> tie-breaking was a valid ordering and each produced a different partition
> into folds of identical size, so `scaffold-lookup` read 0.365 on one machine
> and 0.421 on another with the same library versions. No dependency pin fixes
> that; only owning the assignment does.
>
> **The row order.** Everything downstream reads positions — `KFold` splits
> them, XGBoost's `subsample` and `colsample_bytree` draw against them, the
> 1-NN pool is walked in them. So `run_autopsy` sorts rows by canonical SMILES
> before any of that happens. The seed was always propagated correctly and the
> thread count was never the problem (measured identical across 1, 2, 4 and
> all cores); the row order was.
>
> **The lookup ties.** `_nn_oof` predicts the mean activity of *every*
> training molecule at maximum Tanimoto, where it used to keep whichever
> `np.argmax` reached first. On BACE 99 of 1513 test molecules have a tied
> nearest neighbour and 83 tie across neighbours with different activities.
> Averaging removes the order dependence and is the better estimator — several
> equidistant analogues say more together than one of them picked arbitrarily.
> The baseline is now **"mean of the most similar neighbours"**, not "one
> arbitrary neighbour among the most similar", and its published numbers moved
> accordingly.
>
> Two checks stand behind this. `test_verdict_survives_row_permutation` audits
> a dataset, shuffles the rows, audits again, and asserts all seven headline
> numbers match exactly — on BACE and on a 500-molecule AlogP set where the
> verdict is the opposite one. And this box and a GitHub Actions runner agree
> digit-for-digit, on different CPUs and Python patch releases. The smoke-test
> bands stay slightly wide on the XGBoost rungs because a *different xgboost
> build* still shifts them by ~0.002; the 1-NN rungs are exact.

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
| `AUTOPSY_PAYWALL_ENABLED` | `false` | the master switch. `false` is the free showcase; `true` restores the €199/month paywall in full. Requires `STRIPE_SECRET_KEY`, or the service answers 503 rather than serving free audits from a deployment that believes it is charging. |

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

**Done — reproducible scores.** Canonical row ordering and a tie-averaged
lookup baseline close the last known gap: the whole verdict now survives a row
permutation, verified on two datasets.

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
