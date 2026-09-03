import React, { useState, useEffect, useRef } from "react";
import { useAutopsy, flagColor, engineReachable, subscriberKey, setSubscriberKey } from "./ui_connector.jsx";

// ═══════════════════════════════════════════════════════════════════
// MODEL AUTOPSY — ActarusLab · NEUTRA-language build
// Forensic report for a QSAR model. Every number measured, not
// simulated. Opens on the BACE-1 audit (1513 compounds, 377 Bemis–
// Murcko series); load a CSV and it runs the same engine on your data.
// ═══════════════════════════════════════════════════════════════════

// The BACE-1 audit — the state the page opens in, and what it falls back to
// until a dataset is loaded. Numbers are the engine's own output for
// bace.csv (k=5, seed=0), so benchmark mode and bace_report.html agree to
// the last decimal; the prose is hand-written for this dataset.
const DEMO = {
  source: "BACE-1 · CLEAN BENCHMARK — IN-HOUSE DATA COLLAPSES FURTHER",
  rungs: [
    { key: "xgb_rand", label: "XGBoost", cond: "random split", r2: 0.709, rmse: 0.724, rho: 0.819, kind: "reported" },
    { key: "nn_rand", label: "1-NN lookup", cond: "random split", r2: 0.572, rmse: 0.878, rho: 0.749, kind: "lookup" },
    { key: "xgb_scaf", label: "XGBoost", cond: "scaffold split", r2: 0.597, rmse: 0.852, rho: 0.746, kind: "survives" },
    { key: "nn_scaf", label: "1-NN lookup", cond: "scaffold split", r2: 0.451, rmse: 0.995, rho: 0.695, kind: "lookup" },
    { key: "perm", label: "Permutation", cond: "shuffled target", r2: -0.222, rmse: 1.483, rho: 0.02, kind: "floor" },
  ],
  reported: 0.709, lookupRand: 0.572, survives: 0.597, nnScaf: 0.451,
  learned: 0.146,
  lookupPct: 81,
  headline: null,          // demo keeps the hand-written verdict below
  warnings: [],            // the benchmark drops nothing
  readout: [
    { signal: "Similarity leakage", flag: "SEVERE", value_pct: 81,
      note: "A bare nearest-neighbour lookup reproduces most of the headline. The score rewards recognising known analogues, not learned SAR." },
    { signal: "Scaffold transfer", flag: "PARTIAL", value: 0.597,
      note: "On disjoint chemical series the model holds 0.60 — real, but below the reported figure. This is what generalises to new chemistry." },
    { signal: "Learned structure", flag: "THIN", value: 0.146,
      note: "Scaffold performance minus the scaffold-split lookup. The only structure the model added beyond averaging its nearest analogues, and there is little of it." },
    { signal: "Temporal test", flag: "N/A", value: null,
      note: "Benchmark carries no assay dates. On a real ChEMBL target this rung activates from document year — the split a random fold hides entirely." },
    { signal: "Permutation floor", flag: "CLEAN", value: -0.222,
      note: "Shuffled-target control collapses below zero. The pipeline itself is honest — no featurisation or splitting leak." },
  ],
  specimen: { n_compounds: 1513, n_scaffold_series: 377, n_singleton_series: 200,
              largest_series: 63, largest_series_pct: 4.2, target_mean: 6.522,
              target_sd: 1.342, exact_duplicate_smiles: 0 },
  meta: "ECFP4 · 2048 bit · XGBoost (400 trees, depth 6) · 5-FOLD · POOLED OOF R² · DETERMINISTIC GROUPED FOLDS ON GENERIC SCAFFOLDS · TIE-AVERAGED TANIMOTO 1-NN · PERMUTATION CONTROL · SEED 0",
};

// engine AutopsyResult -> the shape this page renders
function fromResult(res, source) {
  const v = res.verdict, m = res.meta;
  return {
    source: source.toUpperCase(),
    rungs: res.ladder.filter((r) => r.r2 !== null).map((r, i) => ({
      key: `${r.kind}-${i}`, label: r.model, cond: r.condition.split(" · ")[0],
      r2: r.r2, rmse: r.rmse, rho: r.spearman, kind: r.kind,
    })),
    reported: v.reported, lookupRand: v.lookup_random,
    survives: v.survives_scaffold, nnScaf: v.lookup_scaffold,
    learned: v.learned_beyond_lookup, lookupPct: v.lookup_pct_of_reported,
    headline: v.headline,
    readout: res.readout,
    warnings: res.warnings || [],
    specimen: res.specimen,
    meta: `${m.featurisation} · ${m.k_folds}-FOLD · POOLED OOF R² · DETERMINISTIC GROUPED FOLDS ON GENERIC SCAFFOLDS · TIE-AVERAGED TANIMOTO 1-NN · PERMUTATION CONTROL · SEED ${m.seed}`,
  };
}

const C = {
  bg0: "#0A1418", bg1: "#0E1D22", panel: "#0C1A20", panelHi: "#112A31",
  edge: "#1B3A42", edgeSoft: "#152C33",
  cyan: "#7DE3E0", cyanDim: "#4FA9A6", ice: "#CFF6F4",
  text: "#AEC9CC", textDim: "#6E8A8E", mut: "#54706F",
  amber: "#E0B24D", red: "#D65A4E", redDim: "#7C3A36", green: "#6FC49B",
};
const mono = "'IBM Plex Mono', ui-monospace, monospace";
const sans = "'Barlow', 'Inter', system-ui, sans-serif";

const num = (v, dp = 2) => (v == null ? "——" : v.toFixed(dp));

const Pill = ({ children, tone = C.green }) => (
  <span style={{ fontFamily: mono, fontSize: 10, fontWeight: 600, letterSpacing: "0.04em",
    color: tone, border: `1px solid ${tone}44`, borderRadius: 20, padding: "1px 8px",
    background: `${tone}12`, whiteSpace: "nowrap" }}>{children}</span>
);
const Bar = ({ frac, tone = C.cyan, track = C.edgeSoft }) => (
  <div style={{ height: 4, background: track, borderRadius: 3, overflow: "hidden" }}>
    <div style={{ width: `${Math.max(0, Math.min(1, frac)) * 100}%`, height: "100%", background: tone,
      transition: "width 0.9s cubic-bezier(.2,.7,.2,1)" }} />
  </div>
);

export default function ModelAutopsyNeutra() {
  const [revealed, setRevealed] = useState(0);
  const [verdict, setVerdict] = useState(false);
  const [file, setFile] = useState(null);
  const [cols, setCols] = useState({ smiles: "smiles", y: "pIC50", date: "" });
  const [engineUp, setEngineUp] = useState(false);
  const [hasKey] = useState(() => !!subscriberKey());
  const timers = useRef([]);
  const { result, status, error, progress, paywall, runFromFile } = useAutopsy();

  useEffect(() => () => timers.current.forEach(clearTimeout), []);
  useEffect(() => {
    let alive = true;
    engineReachable().then((up) => { if (alive) setEngineUp(up); });
    return () => { alive = false; };
  }, []);

  // Two shapes come back from the engine and they are not interchangeable.
  // `reserved` is the whole audit and drives every panel. `free` is the
  // synthetic verdict — one percentage — and the ladder on screen stays the
  // BACE-1 benchmark, which the contact panel says in as many words. Painting
  // a benchmark ladder as if it were the visitor's own data would be the one
  // dishonest thing this app could do.
  const full = result && result.tier === "reserved" ? result : null;
  const free = result && result.tier === "free" ? result : null;
  const D = full ? fromResult(full, file ? file.name : "uploaded data") : DEMO;
  const warnings = free ? (free.warnings || []) : (D.warnings || []);
  const busy = status === "running";

  // With a free verdict on screen every other panel is still showing BACE-1.
  // Marking only the ladder would leave the dials, the readout cards and the
  // specimen tiles reading as the visitor's own diagnosis, which is precisely
  // the thing withheld from them. The prefix comes first in the title because
  // that is where the eye lands before the numbers.
  const bench = (title) => (free ? `BENCHMARK · ${title}` : title);

  const reveal = (rungs) => {
    timers.current.forEach(clearTimeout); timers.current = [];
    setVerdict(false); setRevealed(0);
    rungs.forEach((_, i) => timers.current.push(setTimeout(() => setRevealed(i + 1), 480 + i * 600)));
    timers.current.push(setTimeout(() => setVerdict(true), 480 + rungs.length * 600 + 260));
  };

  const run = async () => {
    if (!file) return reveal(DEMO.rungs);          // no dataset: replay the benchmark
    timers.current.forEach(clearTimeout); timers.current = [];
    setVerdict(false); setRevealed(0);
    try {
      const data = await runFromFile(file, { smiles: cols.smiles, y: cols.y, date: cols.date || undefined });
      reveal(data.tier === "reserved" ? data.ladder.filter((r) => r.r2 !== null) : DEMO.rungs);
    } catch { /* surfaced through `error` below */ }
  };

  const label = busy ? "● AUTOPSY RUNNING…" : file ? "▶ RUN AUTOPSY" : "▶ REPLAY BENCHMARK";
  // the engine's log line, trimmed of its trailing ellipsis and rung prefix
  const rung = progress ? progress.replace(/^rung · /, "").replace(/…$/, "") : null;

  return (
    <div style={{ minHeight: "100vh", boxSizing: "border-box",
      background: `radial-gradient(120% 90% at 50% -10%, ${C.bg1} 0%, ${C.bg0} 60%, #06090B 100%)`,
      color: C.text, fontFamily: sans, padding: "clamp(14px,2.5vw,30px)" }}>
      <style>{`
        .autopsy-grid {
          display: grid;
          grid-template-columns: minmax(230px,1fr) minmax(360px,1.9fr) minmax(250px,1.15fr);
          gap: 14px; align-items: start;
        }
        @media (max-width: 920px) { .autopsy-grid { grid-template-columns: 1fr; } }
        * { box-sizing: border-box; }
        button:focus-visible, label:focus-within, input:focus-visible { outline: 2px solid ${C.cyan}; outline-offset: 2px; }
        @media (prefers-reduced-motion: reduce) { * { transition-duration: .01ms !important; animation-duration: .01ms !important; } }
      `}</style>

      <div style={{ maxWidth: 1320, margin: "0 auto", border: `1px solid ${C.edge}`, borderRadius: 14,
        background: `linear-gradient(180deg, ${C.bg1}66, transparent 40%)`,
        boxShadow: `0 0 0 1px #000, 0 30px 80px -40px #000`, padding: "clamp(16px,2.2vw,26px)" }}>

        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16, marginBottom: 18 }}>
          <div>
            <div style={{ fontFamily: sans, fontSize: 11, fontWeight: 600, letterSpacing: "0.34em", color: C.cyan, marginBottom: 4 }}>ACTARUSLAB</div>
            <div style={{ fontFamily: sans, fontSize: "clamp(26px,4vw,40px)", fontWeight: 500, letterSpacing: "0.42em", color: C.ice, lineHeight: 1, marginBottom: 8 }}>AUTOPSY</div>
            <div style={{ fontFamily: sans, fontSize: 11, fontWeight: 600, letterSpacing: "0.28em", color: C.textDim, marginBottom: 6 }}>MODEL FORENSICS</div>
            <div style={{ fontFamily: sans, fontStyle: "italic", fontSize: 14, color: C.cyanDim }}>Measures what a model really learned. Warns you when the score is a lie.</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 22, flexWrap: "wrap" }}>
            <div style={{ display: "flex", border: `1px solid ${C.edge}`, borderRadius: 10, overflow: "hidden", background: C.panel }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "9px 16px", background: C.panelHi, borderRight: `1px solid ${C.edge}` }}>
                <Dot on /> <span style={{ fontFamily: sans, fontWeight: 600, fontSize: 14, color: C.ice }}>Leakage</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "9px 16px" }}>
                <Dot /> <span style={{ fontFamily: sans, fontWeight: 500, fontSize: 14, color: C.textDim }}>Survival</span>
              </div>
            </div>
            <Ghost label={full ? "LIVE" : free ? "VERDICT ONLY" : engineUp ? "ENGINE READY" : "BENCHMARK"}
                   on={!!result || engineUp} />
            {hasKey && <Ghost label="SUBSCRIBED" on />}
            <Ghost label="SPECIMEN" />
          </div>
        </header>

        <div style={{ fontFamily: sans, fontSize: 12.5, color: C.textDim, marginBottom: 16, letterSpacing: "0.02em" }}>
          Post-mortem: how much of a reported R² survives an honest split — measured on a live target, nothing simulated.
        </div>

        <div className="autopsy-grid">

          <Panel title={bench("CAUSE OF DEATH")}>
            <Dial value={D.lookupPct == null ? "——" : D.lookupPct} unit={D.lookupPct == null ? "" : "%"} caption="OF SCORE IS LOOKUP" sub="reproducible by nearest-neighbour" tone={C.red} big fill={(D.lookupPct ?? 0) / 100} />
            <div style={{ height: 14 }} />
            <Dial value={num(D.survives)} unit="" caption="SURVIVES NEW SERIES" sub="scaffold-disjoint R²" tone={C.amber} fill={(D.survives ?? 0) / 0.78} />
          </Panel>

          <Panel title={bench("POST-MORTEM · DISTANCE FROM THE HONEST SCORE")}>
            {warnings.map((w, i) => {
              const tone = w.level === "severe" ? C.red : C.amber;
              return (
                <div key={i} style={{ marginBottom: 14, padding: "12px 15px", borderRadius: 8,
                  background: C.panel, border: `1px solid ${tone}55`, borderLeft: `2px solid ${tone}` }}>
                  <div style={{ fontFamily: mono, fontSize: 11, fontWeight: 600, letterSpacing: "0.06em", color: tone }}>
                    ⚠ DROPPED ROWS
                  </div>
                  <div style={{ fontFamily: sans, fontSize: 13, lineHeight: 1.55, color: C.text, marginTop: 6 }}>
                    {w.text}
                  </div>
                </div>
              );
            })}
            <LadderChart rungs={D.rungs} revealed={revealed} reported={D.reported} />

            <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
              {engineUp && <FilePick file={file} onPick={(f) => { setFile(f); setRevealed(0); setVerdict(false); }} disabled={busy} />}
              <Action primary onClick={run} label={label} disabled={busy} />
              <Action onClick={() => { setRevealed(D.rungs.length); setVerdict(true); }} label="↧ REVEAL ALL" disabled={busy} />
            </div>

            {engineUp && file && <ColumnForm cols={cols} setCols={setCols} disabled={busy} />}
            {busy && rung && (
              <div style={{ display: "flex", alignItems: "center", gap: 9, marginTop: 10 }}>
                <span style={{ width: 6, height: 6, borderRadius: 6, background: C.cyan,
                  boxShadow: `0 0 8px ${C.cyan}`, flex: "none" }} />
                <span style={{ fontFamily: mono, fontSize: 11.5, color: C.cyanDim,
                  letterSpacing: "0.02em" }}>{rung}</span>
              </div>
            )}
            {!engineUp && (
              <div style={{ fontFamily: sans, fontSize: 11.5, color: C.mut, marginTop: 10, lineHeight: 1.5 }}>
                No engine behind this page — showing the BACE-1 benchmark. Run{" "}
                <code style={{ fontFamily: mono, color: C.cyanDim }}>uvicorn autopsy.api:app</code>{" "}
                and open it from there to audit your own CSV.
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 1, marginTop: 14, border: `1px solid ${C.edge}`, borderRadius: 8, overflow: "hidden", background: C.edge }}>
              {[["REPORTED", num(D.reported), C.green], ["LOOKUP", num(D.lookupRand), C.red],
                ["SURVIVES", num(D.survives), C.amber], ["LEARNED", num(D.learned, 3), C.cyan]].map(([k, v, t]) => (
                <div key={k} style={{ background: C.panel, padding: "11px 12px" }}>
                  <div style={{ fontFamily: sans, fontSize: 9.5, fontWeight: 600, letterSpacing: "0.14em", color: C.textDim, marginBottom: 5 }}>{k}</div>
                  <div style={{ fontFamily: mono, fontSize: 20, fontWeight: 500, color: t }}>{v}</div>
                </div>
              ))}
            </div>

            {error && (
              <div style={{ marginTop: 14, padding: "14px 16px", borderRadius: 8, background: C.panel,
                border: `1px solid ${paywall ? C.edge : C.redDim}`,
                borderLeft: `2px solid ${paywall ? C.cyan : C.red}` }}>
                <div style={{ fontFamily: sans, fontSize: 9.5, fontWeight: 700, letterSpacing: "0.2em",
                  color: paywall ? C.cyan : C.red, marginBottom: 8 }}>
                  {paywall ? "SUBSCRIPTION REQUIRED" : "AUTOPSY FAILED"}
                </div>
                <div style={{ fontFamily: sans, fontSize: 13.5, lineHeight: 1.55, color: C.text }}>{error}</div>
                {paywall && (
                  <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
                    <a href={paywall} style={{ fontFamily: mono, fontSize: 12, fontWeight: 600,
                      letterSpacing: "0.08em", padding: "10px 16px", borderRadius: 8, textDecoration: "none",
                      color: C.bg0, background: `linear-gradient(180deg, ${C.ice}, ${C.cyan})`,
                      border: `1px solid ${C.cyan}` }}>SUBSCRIBE</a>
                    <button onClick={() => {
                      const k = window.prompt("Paste your access key (ma_…)");
                      if (k) { setSubscriberKey(k.trim()); window.location.reload(); }
                    }} style={{ fontFamily: mono, fontSize: 12, fontWeight: 600, letterSpacing: "0.08em",
                      cursor: "pointer", padding: "10px 16px", borderRadius: 8, color: C.text,
                      background: C.panelHi, border: `1px solid ${C.edge}` }}>I HAVE A KEY</button>
                  </div>
                )}
              </div>
            )}

            {free && (
              <div style={{ marginTop: 14, padding: "16px 18px", borderRadius: 8, background: C.panel,
                border: `1px solid ${C.edge}`, borderLeft: `2px solid ${C.cyan}` }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
                  <div style={{ fontFamily: sans, fontSize: 9.5, fontWeight: 700, letterSpacing: "0.2em",
                    color: C.cyan }}>VERDETTO SINTETICO</div>
                  <div style={{ fontFamily: mono, fontSize: 34, fontWeight: 500, lineHeight: 1,
                    color: free.inflation_state === "clean" ? C.green : C.red }}>
                    {free.inflation_pct == null ? "——"
                      : `${free.inflation_pct > 0 ? "+" : ""}${free.inflation_pct}%`}
                  </div>
                  {free.inflation_basis && (
                    <div style={{ fontFamily: sans, fontSize: 10.5, fontWeight: 600, letterSpacing: "0.14em",
                      color: C.textDim }}>{`VS ${free.inflation_basis.toUpperCase()}`}</div>
                  )}
                </div>

                {/* The engine composes this text, percentage already in it, so the
                    page can never quote a number the audit did not produce. */}
                {free.contact.message.split("\n\n").map((para, i) => (
                  <div key={i} style={{ fontFamily: sans, fontSize: i === 0 ? 14.5 : 13.5, lineHeight: 1.6,
                    color: i === 0 ? C.text : C.textDim, marginTop: i === 0 ? 12 : 9,
                    maxWidth: 760 }}>{para}</div>
                ))}

                <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
                  <a href={`mailto:${free.contact.email}`} style={{ fontFamily: mono, fontSize: 12,
                    fontWeight: 600, letterSpacing: "0.08em", padding: "10px 16px", borderRadius: 8,
                    textDecoration: "none", color: C.bg0,
                    background: `linear-gradient(180deg, ${C.ice}, ${C.cyan})`,
                    border: `1px solid ${C.cyan}` }}>{free.contact.email.toUpperCase()}</a>
                </div>

                <div style={{ fontFamily: sans, fontSize: 11.5, color: C.mut, marginTop: 14,
                  lineHeight: 1.5, maxWidth: 760 }}>
                  This percentage is yours. Every other number on this page — the ladder,
                  the dials, the readout, the specimen tiles — is the BACE-1 benchmark,
                  published in full so you can see what a complete diagnosis contains.
                  Your file was audited exactly the same way.
                </div>
              </div>
            )}

            {verdict && !error && !free && (
              <div style={{ marginTop: 14, padding: "14px 16px", borderRadius: 8, background: C.panel, border: `1px solid ${C.redDim}`, borderLeft: `2px solid ${C.red}` }}>
                <div style={{ fontFamily: sans, fontSize: 9.5, fontWeight: 700, letterSpacing: "0.2em", color: C.red, marginBottom: 8 }}>VERDICT</div>
                <div style={{ fontFamily: sans, fontSize: 13.5, lineHeight: 1.55, color: C.text }}>
                  {D.headline ? D.headline : (
                    <>
                      The reported <b style={{ color: C.ice }}>{num(D.reported)}</b> is not a lie — it measures the wrong thing. It scores how well the model
                      recognises molecules it has effectively already seen. What predicts the next campaign is
                      <b style={{ color: C.amber, fontFamily: mono }}> {num(D.survives)}</b>; the model's own contribution over a lookup table is
                      <b style={{ color: C.cyan, fontFamily: mono }}> {num(D.learned, 3)}</b>. Put those two numbers in the due diligence, not the headline.
                    </>
                  )}
                </div>
              </div>
            )}
          </Panel>

          <Panel title={bench("FORENSIC READOUT")}>
            {D.readout.map((c) => {
              const tone = flagColor(c.flag);
              const value = c.value_pct != null ? `${c.value_pct}%` : c.value == null ? "——" : c.value.toFixed(c.signal === "Learned structure" ? 3 : 2);
              const frac = c.value_pct != null ? c.value_pct / 100 : c.value == null ? 0 : Math.max(0.04, c.value / 0.78);
              return <Readout key={c.signal} head={c.signal.toUpperCase()} pill={<Pill tone={tone}>{c.flag}</Pill>}
                              value={value} tone={tone} frac={frac} note={c.note} />;
            })}
          </Panel>
        </div>

        <div style={{ marginTop: 14 }}>
          <Panel title={bench("SPECIMEN X-RAY · WHY THE HONEST SPLIT MATTERS")}>
            <div style={{ fontFamily: sans, fontSize: 13, color: C.textDim, marginBottom: 14, maxWidth: 900, lineHeight: 1.55 }}>
              The tell is in the composition: <b style={{ color: C.amber }}>{D.specimen.n_singleton_series} of {D.specimen.n_scaffold_series} scaffold series appear only once</b>. A random split
              scatters near-identical analogues across train and test, so the model grades its own copies — that is where the phantom {num(D.reported)} comes from.
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 1, border: `1px solid ${C.edge}`, borderRadius: 8, overflow: "hidden", background: C.edge }}>
              {[["Compounds", D.specimen.n_compounds.toLocaleString("en-US"), "on the slab"],
                ["Scaffold series", String(D.specimen.n_scaffold_series), "Bemis–Murcko cores"],
                ["Singleton series", String(D.specimen.n_singleton_series), "seen once — only memorisable"],
                ["Largest series", String(D.specimen.largest_series), `one core = ${Math.round(D.specimen.largest_series_pct)}% of data`],
                ["Target mean", D.specimen.target_mean.toFixed(2), `± ${D.specimen.target_sd.toFixed(2)} sd`],
                ["Duplicates", String(D.specimen.exact_duplicate_smiles), D.specimen.exact_duplicate_smiles ? "repeat measurements" : "no trivial contamination"]].map(([k, v, s], i) => (
                <div key={k} style={{ background: C.panel, padding: "13px 14px" }}>
                  <div style={{ fontFamily: sans, fontSize: 10.5, fontWeight: 600, letterSpacing: "0.08em", color: C.textDim, marginBottom: 6 }}>{k.toUpperCase()}</div>
                  <div style={{ fontFamily: mono, fontSize: 22, fontWeight: 500, color: i === 2 ? C.amber : C.ice, lineHeight: 1 }}>{v}</div>
                  <div style={{ fontFamily: sans, fontSize: 11, color: C.mut, marginTop: 5 }}>{s}</div>
                </div>
              ))}
            </div>
          </Panel>
        </div>

        <div style={{ marginTop: 16, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 10, fontFamily: mono, fontSize: 10.5, color: C.mut, letterSpacing: "0.02em" }}>
          <span>{D.meta}</span>
          <span style={{ color: C.textDim }}>SPECIMEN · {D.source} · <span style={{ color: C.cyanDim }}>secure yes · local yes · sent no</span></span>
        </div>
      </div>
    </div>
  );
}

function FilePick({ file, onPick, disabled }) {
  return (
    <label style={{ fontFamily: mono, fontSize: 12, fontWeight: 600, letterSpacing: "0.08em",
      cursor: disabled ? "default" : "pointer", flex: "1 1 auto", minWidth: 130, padding: "12px 16px",
      borderRadius: 8, color: file ? C.ice : C.text, background: C.panelHi,
      border: `1px solid ${file ? C.cyanDim : C.edge}`, opacity: disabled ? 0.7 : 1,
      display: "flex", alignItems: "center", justifyContent: "center", gap: 8, textAlign: "center",
      overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" }}>
      ⇱ {file ? file.name : "LOAD CSV"}
      <input type="file" accept=".csv,text/csv" disabled={disabled} style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onPick(f); }} />
    </label>
  );
}

function ColumnForm({ cols, setCols, disabled }) {
  const field = (key, label, placeholder) => (
    <label style={{ display: "flex", flexDirection: "column", gap: 5, flex: "1 1 130px" }}>
      <span style={{ fontFamily: sans, fontSize: 9.5, fontWeight: 600, letterSpacing: "0.14em", color: C.textDim }}>{label}</span>
      <input value={cols[key]} disabled={disabled} placeholder={placeholder}
        onChange={(e) => setCols({ ...cols, [key]: e.target.value })}
        style={{ fontFamily: mono, fontSize: 12, color: C.ice, background: C.bg0,
          border: `1px solid ${C.edge}`, borderRadius: 6, padding: "8px 10px", minWidth: 0 }} />
    </label>
  );
  return (
    <div style={{ display: "flex", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
      {field("smiles", "SMILES COLUMN", "smiles")}
      {field("y", "ACTIVITY COLUMN", "pIC50")}
      {field("date", "DATE COLUMN · OPTIONAL", "document_year")}
    </div>
  );
}

function Dot({ on }) {
  return <span style={{ width: 8, height: 8, borderRadius: 8, background: on ? C.cyan : "transparent", border: `1px solid ${on ? C.cyan : C.textDim}`, boxShadow: on ? `0 0 8px ${C.cyan}` : "none", display: "inline-block" }} />;
}
function Ghost({ label, on }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
      <span style={{ width: 6, height: 6, borderRadius: 6, border: `1px solid ${on ? C.cyan : C.mut}`, background: on ? C.cyan : "transparent" }} />
      <span style={{ fontFamily: sans, fontSize: 11, fontWeight: 600, letterSpacing: "0.16em", color: on ? C.cyanDim : C.mut }}>{label}</span>
    </span>
  );
}
function Panel({ title, children }) {
  return (
    <section style={{ background: C.panel, border: `1px solid ${C.edge}`, borderRadius: 12, padding: "clamp(13px,1.4vw,18px)", boxShadow: `inset 0 1px 0 ${C.edgeSoft}` }}>
      <div style={{ fontFamily: sans, fontSize: 11, fontWeight: 600, letterSpacing: "0.18em", color: C.cyan, marginBottom: 14 }}>{title}</div>
      {children}
    </section>
  );
}
function Action({ label, onClick, primary, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{ fontFamily: mono, fontSize: 12, fontWeight: 600, letterSpacing: "0.08em",
      cursor: disabled ? "default" : "pointer", flex: "1 1 auto", minWidth: 130, padding: "12px 16px", borderRadius: 8,
      color: primary ? C.bg0 : C.text, background: primary ? `linear-gradient(180deg, ${C.ice}, ${C.cyan})` : C.panelHi,
      border: `1px solid ${primary ? C.cyan : C.edge}`, boxShadow: primary ? `0 0 22px -6px ${C.cyan}` : "none", opacity: disabled ? 0.7 : 1 }}>{label}</button>
  );
}
function Readout({ head, pill, value, note, tone, frac }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span style={{ fontFamily: sans, fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", color: C.text }}>{head}</span>
        {pill}
      </div>
      <div style={{ fontFamily: mono, fontSize: 24, fontWeight: 500, color: tone, lineHeight: 1, marginBottom: 8 }}>{value}</div>
      <Bar frac={frac} tone={tone} />
      <div style={{ fontFamily: sans, fontSize: 11.5, color: C.mut, marginTop: 8, lineHeight: 1.5 }}>{note}</div>
    </div>
  );
}
function Dial({ value, unit, caption, sub, tone, big, fill }) {
  const size = big ? 148 : 118;
  const r = size / 2 - 12; const cx = size / 2, cy = size / 2; const ticks = 44;
  const frac = fill == null ? 0.62 : Math.max(0, Math.min(1, fill));
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}>
      <div style={{ position: "relative", width: size, height: size }}>
        <svg viewBox={`0 0 ${size} ${size}`} style={{ width: "100%", height: "100%" }}>
          {Array.from({ length: ticks }).map((_, i) => {
            const a = (i / ticks) * Math.PI * 2 - Math.PI / 2;
            const on = i / ticks < frac;
            const r1 = r, r2o = r - (i % 4 === 0 ? 9 : 6);
            return <line key={i} x1={cx + r1 * Math.cos(a)} y1={cy + r1 * Math.sin(a)} x2={cx + r2o * Math.cos(a)} y2={cy + r2o * Math.sin(a)}
              stroke={on ? tone : C.edge} strokeWidth={i % 4 === 0 ? 1.6 : 1} opacity={on ? 0.95 : 0.5} />;
          })}
          <circle cx={cx} cy={cy} r={r - 16} fill="none" stroke={C.edge} strokeWidth="1" />
        </svg>
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <div style={{ fontFamily: mono, fontSize: big ? 40 : 32, fontWeight: 500, color: C.ice, lineHeight: 1 }}>
            {value}<span style={{ fontSize: big ? 18 : 15, color: tone }}>{unit}</span>
          </div>
        </div>
      </div>
      <div style={{ fontFamily: sans, fontSize: 11.5, fontWeight: 600, letterSpacing: "0.12em", color: C.text, marginTop: 10 }}>{caption}</div>
      <div style={{ fontFamily: sans, fontSize: 11, color: C.mut, marginTop: 3 }}>{sub}</div>
    </div>
  );
}
function LadderChart({ rungs, revealed, reported }) {
  const W = 640, H = 300, padL = 46, padR = 74, padT = 26, padB = 52;
  const bandH = H - padT - padB;
  const stepX = (W - padL - padR) / Math.max(1, rungs.length - 1);
  // The benchmark band, widened only if a loaded dataset falls outside it.
  const vals = rungs.map((r) => r.r2);
  const lo = Math.min(-0.3, Math.floor(Math.min(...vals) * 4) / 4);
  const hi = Math.max(0.78, Math.ceil(Math.max(...vals) * 4) / 4);
  const x = (i) => padL + i * stepX;
  const y = (r2) => padT + (1 - (r2 - lo) / (hi - lo)) * bandH;
  const gridlines = [0.75, 0.5, 0.25, 0, -0.25].filter((g) => g >= lo && g <= hi);
  const colFor = (k) => k === "reported" ? C.green : k === "survives" ? C.amber : k === "floor" ? C.mut : C.red;
  const pts = rungs.slice(0, revealed).map((r, i) => `${x(i)},${y(r.r2)}`).join(" ");
  return (
    <div style={{ background: C.bg0, border: `1px solid ${C.edge}`, borderRadius: 8, padding: "6px 4px" }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
        {gridlines.map((g) => (
          <g key={g}>
            <line x1={padL} x2={W - padR} y1={y(g)} y2={y(g)} stroke={C.edgeSoft} strokeWidth="1" strokeDasharray="2 5" />
            <text x={padL - 8} y={y(g) + 3.5} textAnchor="end" fontFamily={mono} fontSize="10" fill={C.mut}>{g.toFixed(2)}</text>
          </g>
        ))}
        {reported != null && <line x1={padL} x2={W - padR} y1={y(reported)} y2={y(reported)} stroke={C.green} strokeWidth="1" strokeDasharray="1 6" opacity="0.55" />}
        {revealed > 1 && <polyline points={pts} fill="none" stroke={C.red} strokeWidth="2" opacity="0.55" style={{ transition: "all .4s ease" }} />}
        {rungs.slice(0, revealed).map((r, i) => {
          const anchor = i === 0 ? "start" : i === rungs.length - 1 ? "end" : "middle";
          const lx = i === 0 ? x(i) - 22 : i === rungs.length - 1 ? x(i) + 22 : x(i);
          const vy = i === 0 ? y(r.r2) + 20 : y(r.r2) - 12;
          const vx = i === 0 ? x(i) + 22 : x(i);
          return (
            <g key={r.key} style={{ animation: "np 0.4s ease" }}>
              <circle cx={x(i)} cy={y(r.r2)} r="9" fill="none" stroke={colFor(r.kind)} strokeWidth="1" opacity="0.4" />
              <circle cx={x(i)} cy={y(r.r2)} r="4.5" fill={colFor(r.kind)} stroke={C.bg0} strokeWidth="1.5" />
              <text x={vx} y={vy} textAnchor="middle" fontFamily={mono} fontSize="14" fontWeight="600" fill={colFor(r.kind)}>{r.r2.toFixed(2)}</text>
              <text x={lx} y={H - 26} textAnchor={anchor} fontFamily={mono} fontSize="10.5" fill={C.text}>{r.label}</text>
              <text x={lx} y={H - 12} textAnchor={anchor} fontFamily={mono} fontSize="9.5" fill={C.mut}>{r.cond}</text>
            </g>
          );
        })}
      </svg>
      <style>{`@keyframes np{0%{opacity:0;transform:scale(.4)}100%{opacity:1;transform:scale(1)}}`}</style>
    </div>
  );
}
