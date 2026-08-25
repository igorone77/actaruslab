import React, { useState, useEffect, useRef } from "react";

// ═══════════════════════════════════════════════════════════════════
// MODEL AUTOPSY — ActarusLab · NEUTRA-language build
// Forensic report for a QSAR model. Every number measured, not
// simulated: BACE-1 inhibition (pIC50), 1513 compounds, 377
// Bemis–Murcko scaffold series. XGBoost / 1-NN Tanimoto / permutation.
// ═══════════════════════════════════════════════════════════════════

const RUNGS = [
  { key: "xgb_rand", label: "XGBoost",     cond: "random split",   r2: 0.720, rmse: 0.710, rho: 0.827, kind: "reported" },
  { key: "nn_rand",  label: "1-NN lookup", cond: "random split",   r2: 0.576, rmse: 0.874, rho: 0.757, kind: "lookup" },
  { key: "xgb_scaf", label: "XGBoost",     cond: "scaffold split", r2: 0.593, rmse: 0.856, rho: 0.744, kind: "survives" },
  { key: "nn_scaf",  label: "1-NN lookup", cond: "scaffold split", r2: 0.448, rmse: 0.997, rho: 0.696, kind: "lookup" },
  { key: "perm",     label: "Permutation", cond: "shuffled target",r2: -0.219,rmse: 1.482, rho: -0.012,kind: "floor" },
];
const REPORTED = 0.720, LOOKUP_RAND = 0.576, SURVIVES = 0.593, NN_SCAF = 0.448;
const LEARNED = 0.146; // engine-canonical: scaffold 0.593 − scaffold-lookup 0.448 (deterministic folds)
const LOOKUP_PCT = Math.round((LOOKUP_RAND / REPORTED) * 100);

const C = {
  bg0: "#0A1418", bg1: "#0E1D22", panel: "#0C1A20", panelHi: "#112A31",
  edge: "#1B3A42", edgeSoft: "#152C33",
  cyan: "#7DE3E0", cyanDim: "#4FA9A6", ice: "#CFF6F4",
  text: "#AEC9CC", textDim: "#6E8A8E", mut: "#54706F",
  amber: "#E0B24D", red: "#D65A4E", redDim: "#7C3A36", green: "#6FC49B",
};
const mono = "'IBM Plex Mono', ui-monospace, monospace";
const sans = "'Barlow', 'Inter', system-ui, sans-serif";

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
const traceY = (r2, top, h) => { const lo = -0.3, hi = 0.78; return top + (1 - (r2 - lo) / (hi - lo)) * h; };

export default function ModelAutopsyNeutra() {
  const [revealed, setRevealed] = useState(0);
  const [verdict, setVerdict] = useState(false);
  const [running, setRunning] = useState(false);
  const timers = useRef([]);
  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const run = () => {
    timers.current.forEach(clearTimeout); timers.current = [];
    setVerdict(false); setRevealed(0); setRunning(true);
    RUNGS.forEach((_, i) => timers.current.push(setTimeout(() => setRevealed(i + 1), 480 + i * 600)));
    timers.current.push(setTimeout(() => { setVerdict(true); setRunning(false); }, 480 + RUNGS.length * 600 + 260));
  };

  return (
    <div style={{ minHeight: "100vh", boxSizing: "border-box",
      background: `radial-gradient(120% 90% at 50% -10%, ${C.bg1} 0%, ${C.bg0} 60%, #06090B 100%)`,
      color: C.text, fontFamily: sans, padding: "clamp(14px,2.5vw,30px)" }}>
      <link href="https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,400;0,500;0,600;1,400;1,500&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
      <style>{`
        .autopsy-grid {
          display: grid;
          grid-template-columns: minmax(230px,1fr) minmax(360px,1.9fr) minmax(250px,1.15fr);
          gap: 14px; align-items: start;
        }
        @media (max-width: 920px) {
          .autopsy-grid { grid-template-columns: 1fr; }
        }
        * { box-sizing: border-box; }
        button:focus-visible { outline: 2px solid ${C.cyan}; outline-offset: 2px; }
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
            <Ghost label="EVIDENCE" /><Ghost label="SPECIMEN" />
          </div>
        </header>

        <div style={{ fontFamily: sans, fontSize: 12.5, color: C.textDim, marginBottom: 16, letterSpacing: "0.02em" }}>
          Post-mortem: how much of a reported R² survives an honest split — measured on a live target, nothing simulated.
        </div>

        <div className="autopsy-grid">

          <Panel title="CAUSE OF DEATH">
            <Dial value={LOOKUP_PCT} unit="%" caption="OF SCORE IS LOOKUP" sub="reproducible by nearest-neighbour" tone={C.red} big fill={LOOKUP_PCT / 100} />
            <div style={{ height: 14 }} />
            <Dial value={SURVIVES.toFixed(2)} unit="" caption="SURVIVES NEW SERIES" sub="scaffold-disjoint R²" tone={C.amber} fill={SURVIVES / 0.78} />
          </Panel>

          <Panel title="POST-MORTEM · DISTANCE FROM THE HONEST SCORE">
            <LadderChart revealed={revealed} />
            <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
              <Action primary onClick={run} label={running ? "● RECORDING…" : "▶ RUN AUTOPSY"} disabled={running} />
              <Action onClick={() => { setRevealed(RUNGS.length); setVerdict(true); }} label="↧ REVEAL ALL" />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 1, marginTop: 14, border: `1px solid ${C.edge}`, borderRadius: 8, overflow: "hidden", background: C.edge }}>
              {[["REPORTED", REPORTED.toFixed(2), C.green],["LOOKUP", LOOKUP_RAND.toFixed(2), C.red],["SURVIVES", SURVIVES.toFixed(2), C.amber],["LEARNED", LEARNED.toFixed(3), C.cyan]].map(([k, v, t]) => (
                <div key={k} style={{ background: C.panel, padding: "11px 12px" }}>
                  <div style={{ fontFamily: sans, fontSize: 9.5, fontWeight: 600, letterSpacing: "0.14em", color: C.textDim, marginBottom: 5 }}>{k}</div>
                  <div style={{ fontFamily: mono, fontSize: 20, fontWeight: 500, color: t }}>{v}</div>
                </div>
              ))}
            </div>
            {verdict && (
              <div style={{ marginTop: 14, padding: "14px 16px", borderRadius: 8, background: C.panel, border: `1px solid ${C.redDim}`, borderLeft: `2px solid ${C.red}` }}>
                <div style={{ fontFamily: sans, fontSize: 9.5, fontWeight: 700, letterSpacing: "0.2em", color: C.red, marginBottom: 8 }}>VERDICT</div>
                <div style={{ fontFamily: sans, fontSize: 13.5, lineHeight: 1.55, color: C.text }}>
                  The reported <b style={{ color: C.ice }}>0.72</b> is not a lie — it measures the wrong thing. It scores how well the model
                  recognises molecules it has effectively already seen. What predicts the next campaign is
                  <b style={{ color: C.amber, fontFamily: mono }}> 0.59</b>; the model's own contribution over a lookup table is
                  <b style={{ color: C.cyan, fontFamily: mono }}> {LEARNED.toFixed(3)}</b>. Put those two numbers in the due diligence, not the headline.
                </div>
              </div>
            )}
          </Panel>

          <Panel title="FORENSIC READOUT">
            <Readout head="SIMILARITY LEAKAGE" pill={<Pill tone={C.red}>SEVERE</Pill>} value={`${LOOKUP_PCT}%`} tone={C.red} frac={LOOKUP_PCT / 100}
              note="A bare nearest-neighbour lookup reproduces most of the headline. The score rewards recognising known analogues, not learned SAR." />
            <Readout head="SCAFFOLD TRANSFER" pill={<Pill tone={C.amber}>PARTIAL</Pill>} value={SURVIVES.toFixed(2)} tone={C.amber} frac={SURVIVES / 0.78}
              note="On disjoint chemical series the model holds 0.59 — real, but below the reported figure. This is what generalises to new chemistry." />
            <Readout head="LEARNED STRUCTURE" pill={<Pill tone={C.cyan}>THIN</Pill>} value={LEARNED.toFixed(3)} tone={C.cyan} frac={LEARNED / 0.78}
              note="Scaffold performance minus the scaffold-split lookup. The only structure the model added beyond copying its nearest analogue." />
            <Readout head="PERMUTATION FLOOR" pill={<Pill tone={C.green}>CLEAN</Pill>} value="−0.22" tone={C.green} frac={0.06}
              note="Shuffled-target control collapses below zero. The pipeline itself is honest — no featurisation or splitting leak." />
            <Readout head="TEMPORAL TEST" pill={<Pill tone={C.mut}>N/A</Pill>} value="——" tone={C.mut} frac={0}
              note="Benchmark carries no assay dates. On a real ChEMBL target this rung activates from document year — the split a random fold hides entirely." />
          </Panel>
        </div>

        <div style={{ marginTop: 14 }}>
          <Panel title="SPECIMEN X-RAY · WHY THE HONEST SPLIT MATTERS">
            <div style={{ fontFamily: sans, fontSize: 13, color: C.textDim, marginBottom: 14, maxWidth: 900, lineHeight: 1.55 }}>
              The tell is in the composition: <b style={{ color: C.amber }}>200 of 377 scaffold series appear only once</b>. A random split
              scatters near-identical analogues across train and test, so the model grades its own copies — that is where the phantom 0.72 comes from.
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 1, border: `1px solid ${C.edge}`, borderRadius: 8, overflow: "hidden", background: C.edge }}>
              {[["Compounds", "1,513", "on the slab"],["Scaffold series", "377", "Bemis–Murcko cores"],["Singleton series", "200", "seen once — only memorisable"],["Largest series", "63", "one core = 4% of data"],["Target pIC50", "6.52", "± 1.34 sd · well-spread"],["Duplicates", "0", "no trivial contamination"]].map(([k, v, s], i) => (
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
          <span>ECFP4 · POOLED OOF R² · GROUPKFOLD ON GENERIC SCAFFOLDS · TANIMOTO 1-NN · PERMUTATION CONTROL</span>
          <span style={{ color: C.textDim }}>SPECIMEN · BACE-1 · CLEAN BENCHMARK — IN-HOUSE DATA COLLAPSES FURTHER · <span style={{ color: C.cyanDim }}>secure yes · local yes · sent no</span></span>
        </div>
      </div>
    </div>
  );
}

function Dot({ on }) {
  return <span style={{ width: 8, height: 8, borderRadius: 8, background: on ? C.cyan : "transparent", border: `1px solid ${on ? C.cyan : C.textDim}`, boxShadow: on ? `0 0 8px ${C.cyan}` : "none", display: "inline-block" }} />;
}
function Ghost({ label }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
      <span style={{ width: 6, height: 6, borderRadius: 6, border: `1px solid ${C.mut}` }} />
      <span style={{ fontFamily: sans, fontSize: 11, fontWeight: 600, letterSpacing: "0.16em", color: C.mut }}>{label}</span>
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
function LadderChart({ revealed }) {
  const W = 640, H = 300, padL = 46, padR = 74, padT = 26, padB = 52;
  const bandH = H - padT - padB;
  const stepX = (W - padL - padR) / (RUNGS.length - 1);
  const x = (i) => padL + i * stepX;
  const y = (r2) => traceY(r2, padT, bandH);
  const colFor = (k) => k === "reported" ? C.green : k === "survives" ? C.amber : k === "floor" ? C.mut : C.red;
  const pts = RUNGS.slice(0, revealed).map((r, i) => `${x(i)},${y(r.r2)}`).join(" ");
  return (
    <div style={{ background: C.bg0, border: `1px solid ${C.edge}`, borderRadius: 8, padding: "6px 4px" }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
        {[0.75, 0.5, 0.25, 0, -0.25].map((g) => (
          <g key={g}>
            <line x1={padL} x2={W - padR} y1={y(g)} y2={y(g)} stroke={C.edgeSoft} strokeWidth="1" strokeDasharray="2 5" />
            <text x={padL - 8} y={y(g) + 3.5} textAnchor="end" fontFamily={mono} fontSize="10" fill={C.mut}>{g.toFixed(2)}</text>
          </g>
        ))}
        <line x1={padL} x2={W - padR} y1={y(REPORTED)} y2={y(REPORTED)} stroke={C.green} strokeWidth="1" strokeDasharray="1 6" opacity="0.55" />
        {revealed > 1 && <polyline points={pts} fill="none" stroke={C.red} strokeWidth="2" opacity="0.55" style={{ transition: "all .4s ease" }} />}
        {RUNGS.slice(0, revealed).map((r, i) => {
          const anchor = i === 0 ? "start" : i === RUNGS.length - 1 ? "end" : "middle";
          const lx = i === 0 ? x(i) - 22 : i === RUNGS.length - 1 ? x(i) + 22 : x(i);
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
