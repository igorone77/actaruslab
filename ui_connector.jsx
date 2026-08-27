// ─────────────────────────────────────────────────────────────────────
// ui_connector.jsx — the bridge between the UI and the engine API.
//
// model_autopsy.jsx uses `useAutopsy()` to turn a picked CSV into a real
// audit. There is no backend in an artifact sandbox, so a page served from
// anywhere but the API will sit on its benchmark constants.
//
// API_BASE is relative to the page, never an absolute URL and never a
// hardcoded port, so the same bundle works on whatever port uvicorn is given,
// under a sub-path on a static host, and from a file:// double-click.
// `window.__AUTOPSY_API__`, set by a host page before this script loads,
// overrides it to point at an API served from somewhere else.
// ─────────────────────────────────────────────────────────────────────

import { useState, useCallback } from "react";

const API_BASE = (typeof window !== "undefined" && window.__AUTOPSY_API__) || ".";

const POLL_MS = 1200;

// Is there an engine behind this page? Served by autopsy.api the answer is
// yes; opened as a bare file or on a static host it is no, and the UI drops
// its upload controls rather than offering a button that cannot work.
export async function engineReachable() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    return res.ok && (await res.json()).status === "ok";
  } catch {
    return false;
  }
}

// ── the hook the UI uses ─────────────────────────────────────────────
// Returns { result, status, error, progress, runFromFile, runFromRecords }.
// `result` matches the engine's AutopsyResult: {specimen, ladder, verdict, readout, meta}.
export function useAutopsy() {
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");   // idle | running | done | error
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(null); // the engine's current rung

  // Path A — user picks a CSV file (browser file input).
  //
  // Submits a job and polls. The audit takes tens of seconds and is
  // quadratic in the lookup rung, so it cannot answer inside a request on
  // any host worth deploying to. `progress` carries the engine's own log
  // line so the wait reads as work rather than a hang.
  const runFromFile = useCallback(async (file, { smiles, y, date, k = 5 }) => {
    setStatus("running"); setError(null); setProgress("uploading…");
    const form = new FormData();
    form.append("file", file);
    form.append("smiles", smiles);
    form.append("y", y);
    if (date) form.append("date", date);
    form.append("k", String(k));

    const fail = async (res) => {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    };

    try {
      const res = await fetch(`${API_BASE}/autopsy/jobs`, { method: "POST", body: form });
      if (!res.ok) await fail(res);
      const { job_id } = await res.json();

      for (;;) {
        await new Promise((r) => setTimeout(r, POLL_MS));
        const poll = await fetch(`${API_BASE}/autopsy/jobs/${job_id}`);
        if (!poll.ok) await fail(poll);
        const job = await poll.json();
        setProgress(job.progress || job.status);
        if (job.status === "done") {
          setResult(job.result); setStatus("done"); setProgress(null);
          return job.result;
        }
        if (job.status === "failed") throw new Error(job.error);
      }
    } catch (e) {
      setError(e.message); setStatus("error"); setProgress(null); throw e;
    }
  }, []);

  // Path B — UI already parsed rows in-browser and sends JSON
  const runFromRecords = useCallback(async (records, { k = 5, hasDate = false } = {}) => {
    setStatus("running"); setError(null);
    try {
      const res = await fetch(`${API_BASE}/autopsy/records`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ records, k, has_date: hasDate }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data); setStatus("done");
      return data;
    } catch (e) {
      setError(e.message); setStatus("error"); throw e;
    }
  }, []);

  return { result, status, error, progress, runFromFile, runFromRecords };
}

// ─────────────────────────────────────────────────────────────────────
// Mapping the engine result onto the existing NEUTRA components.
//
// The prototype's RUNGS / verdict constants map 1:1 onto the API result.
// Where the prototype had:
//
//     const RUNGS = [ { label, cond, r2, kind }, ... ]
//
// use instead, once you have `result` from the hook:
//
//     const RUNGS = result.ladder
//         .filter(r => r.r2 !== null)              // hide rungs that didn't run
//         .map(r => ({
//             label: r.model,                       // "XGBoost" | "1-NN lookup" | "Permutation"
//             cond:  r.condition.split(" · ")[0],   // "random split", "scaffold split", …
//             r2:    r.r2,
//             kind:  r.kind,                         // reported|lookup|survives|temporal|floor
//         }));
//
// And the headline strip / dials read straight from result.verdict:
//
//     const REPORTED    = result.verdict.reported;
//     const LOOKUP_RAND = result.verdict.lookup_random;
//     const SURVIVES    = result.verdict.survives_scaffold;
//     const LEARNED     = result.verdict.learned_beyond_lookup;
//     const LOOKUP_PCT  = result.verdict.lookup_pct_of_reported;
//
// The right-hand readout cards map from result.readout:
//
//     result.readout.map(c => (
//         <Readout key={c.signal}
//                  head={c.signal.toUpperCase()}
//                  pill={<Pill tone={flagColor(c.flag)}>{c.flag}</Pill>}
//                  value={c.value_pct != null ? `${c.value_pct}%`
//                         : c.value == null ? "——" : c.value.toFixed(2)}
//                  note={c.note} />
//     ))
//
// The Specimen X-ray tiles map from result.specimen (n_compounds,
// n_scaffold_series, n_singleton_series, largest_series, target_mean/sd,
// exact_duplicate_smiles).
//
// Nothing about the visual language changes — only the data source moves
// from a constant to a fetch. That is the entire demo→engine switch on the
// front end.
// ─────────────────────────────────────────────────────────────────────

// Flag names are decided by the engine (_learned_flag in engine.py); this
// only paints them, so the cuts can never drift between the two. ARTIFACT is
// red because it marks a number that is not what it looks like.
export function flagColor(flag) {
  const C = {
    SEVERE: "#D65A4E", LEAK: "#D65A4E", ARTIFACT: "#D65A4E",
    MODERATE: "#E0B24D", PARTIAL: "#E0B24D", THIN: "#E0B24D", MARGINAL: "#E0B24D",
    NET: "#7DE3E0", TESTED: "#7DE3E0",
    CLEAN: "#6FC49B", LOW: "#6FC49B",
    "N/A": "#54706F",
  };
  return C[flag] ?? "#7DE3E0";
}

// ── minimal example of the upload control (for reference only) ────────
export function AutopsyUploader() {
  const { result, status, error, runFromFile } = useAutopsy();
  const [cols, setCols] = useState({ smiles: "smiles", y: "pIC50", date: "" });

  return (
    <div>
      <input type="file" accept=".csv" onChange={(e) => {
        const f = e.target.files?.[0];
        if (f) runFromFile(f, { smiles: cols.smiles, y: cols.y, date: cols.date || undefined });
      }} />
      {status === "running" && <span>● running autopsy…</span>}
      {status === "error" && <span>error: {error}</span>}
      {status === "done" && result && (
        <span>reported {result.verdict.reported} · {result.verdict.lookup_pct_of_reported}% lookup</span>
      )}
    </div>
  );
}
