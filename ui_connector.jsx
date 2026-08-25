// ─────────────────────────────────────────────────────────────────────
// ui_connector.jsx — reference wiring for Claude Code
//
// This is the bridge that turns the MODEL AUTOPSY prototype (which holds
// hardcoded BACE constants) into a live client of the engine API.
//
// It is NOT meant to run in an artifact sandbox (no backend there). It is a
// reference for the Code step: drop the engine API behind a URL, then use
// this hook in place of the constants at the top of model_autopsy.jsx.
// ─────────────────────────────────────────────────────────────────────

import { useState, useCallback } from "react";

const API_BASE = import.meta?.env?.VITE_AUTOPSY_API ?? "http://127.0.0.1:8000";

// ── the hook the UI uses ─────────────────────────────────────────────
// Returns { result, status, error, runFromFile, runFromRecords }.
// `result` matches the engine's AutopsyResult: {specimen, ladder, verdict, readout, meta}.
export function useAutopsy() {
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");   // idle | running | done | error
  const [error, setError] = useState(null);

  // Path A — user picks a CSV file (browser file input)
  const runFromFile = useCallback(async (file, { smiles, y, date, k = 5 }) => {
    setStatus("running"); setError(null);
    const form = new FormData();
    form.append("file", file);
    form.append("smiles", smiles);
    form.append("y", y);
    if (date) form.append("date", date);
    form.append("k", String(k));
    try {
      const res = await fetch(`${API_BASE}/autopsy/csv`, { method: "POST", body: form });
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

  return { result, status, error, runFromFile, runFromRecords };
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

export function flagColor(flag) {
  const C = {
    SEVERE: "#D65A4E", LEAK: "#D65A4E",
    MODERATE: "#E0B24D", PARTIAL: "#E0B24D", THIN: "#E0B24D",
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
