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

// A subscriber's key. Kept in localStorage so it survives a reload, and
// accepted from ?key=… once so a purchase confirmation can hand it over in a
// link. Never sent anywhere but this app's own API.
const KEY_STORE = "autopsy_key";

export function subscriberKey() {
  try {
    const fromUrl = new URLSearchParams(location.search).get("key");
    if (fromUrl) {
      localStorage.setItem(KEY_STORE, fromUrl);
      history.replaceState({}, "", location.pathname);   // keep it out of the bar
      return fromUrl;
    }
    return localStorage.getItem(KEY_STORE) || "";
  } catch {
    return "";                                            // private mode, no store
  }
}

export function setSubscriberKey(key) {
  try {
    key ? localStorage.setItem(KEY_STORE, key) : localStorage.removeItem(KEY_STORE);
  } catch { /* nothing to do if storage is blocked */ }
}

function auth() {
  const k = subscriberKey();
  return k ? { Authorization: `Bearer ${k}` } : {};
}

// Is there an engine behind this page, and what will it hand back?
//
// Served by autopsy.api the answer is yes; opened as a bare file or on a
// static host it is no, and the UI drops its upload controls rather than
// offering a button that cannot work.
//
// `mode` says what an audit will return before one is run — a server that
// withholds its findings looks exactly like a broken one until you know
// which it is. Absent on a server older than that field, which is the answer
// to a different question worth being able to ask.
export async function engineHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!res.ok) return { up: false, mode: null };
    const body = await res.json();
    return { up: body.status === "ok", mode: body.mode || null };
  } catch {
    return { up: false, mode: null };
  }
}

// Kept for callers that only need the boolean.
export async function engineReachable() {
  return (await engineHealth()).up;
}

// ── the hook the UI uses ─────────────────────────────────────────────
// Returns { result, status, error, progress, paywall, runFromFile, runFromRecords }.
//
// Two shapes come back, and the UI must tell them apart before anything else.
// Branch on whether `ladder` is there, not on the `tier` string: the shape is
// the fact, the name is only a promise about it.
//
//   "full"     { tier, specimen, ladder, verdict, readout, warnings, meta }
//              The whole audit. This is the default — an audit returns what
//              it found.
//   "verdict"  { tier, inflation_pct, inflation_basis, inflation_state,
//              warnings, contact:{email, message, withheld} }
//              The synthetic verdict alone, on a deployment that sets
//              AUTOPSY_VERDICT_ONLY. The audit still ran in full; the
//              diagnosis simply never left the server.
//
// See autopsy/tiers.py for which switch decides which.
export function useAutopsy() {
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");   // idle | running | done | error
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(null); // the engine's current rung
  const [paywall, setPaywall] = useState(null);   // checkout URL when unpaid

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
      const err = new Error(detail.detail || "the request could not be completed");
      if (res.status === 402) {
        err.paywall = true;
        err.checkoutUrl = res.headers.get("X-Checkout-URL") || `${API_BASE}/billing/checkout`;
      }
      throw err;
    };

    try {
      const res = await fetch(`${API_BASE}/autopsy/jobs`, {
        method: "POST", body: form, headers: auth(),
      });
      if (!res.ok) await fail(res);
      // `job_token` is the claim on this result and is returned once. It is
      // kept in this closure and nowhere else: not in localStorage, not in
      // the URL. A reload loses the audit, which is the correct trade for
      // data that is not ours to leave lying around.
      const { job_id, job_token } = await res.json();
      const claim = job_token ? { "X-Job-Token": job_token } : {};

      for (;;) {
        await new Promise((r) => setTimeout(r, POLL_MS));
        const poll = await fetch(`${API_BASE}/autopsy/jobs/${job_id}`,
                                 { headers: { ...auth(), ...claim } });
        if (!poll.ok) await fail(poll);
        const job = await poll.json();
        setProgress(job.progress || job.status);
        if (job.status === "done") {
          setResult(job.result); setPaywall(null);
          setStatus("done"); setProgress(null);
          return job.result;
        }
        if (job.status === "failed") throw new Error(job.error);
      }
    } catch (e) {
      setError(e.message); setPaywall(e.paywall ? e.checkoutUrl : null);
      setStatus("error"); setProgress(null); throw e;
    }
  }, []);

  // Path B — UI already parsed rows in-browser and sends JSON
  const runFromRecords = useCallback(async (records, { k = 5, hasDate = false } = {}) => {
    setStatus("running"); setError(null);
    try {
      const res = await fetch(`${API_BASE}/autopsy/records`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...auth() },
        body: JSON.stringify({ records, k, has_date: hasDate }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data); setPaywall(null); setStatus("done");
      return data;
    } catch (e) {
      setError(e.message); setStatus("error"); throw e;
    }
  }, []);

  return { result, status, error, progress, paywall, runFromFile, runFromRecords };
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
    // Not a severity. The similarity rung is unvalidated, so it is painted
    // apart from the scale rather than somewhere on it.
    EXPERIMENTAL: "#9FB6C4",
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
      {status === "done" && result && (Array.isArray(result.ladder) ? (
        <span>reported {result.verdict.reported} · {result.verdict.lookup_pct_of_reported}% lookup</span>
      ) : (
        <span>inflation +{result.inflation_pct}% · {result.contact.email}</span>
      ))}
    </div>
  );
}
