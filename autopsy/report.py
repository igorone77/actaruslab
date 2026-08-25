"""
Render an AutopsyResult as a self-contained NEUTRA-styled HTML report.
No JS framework, no external assets except the two web fonts — a single
file a client can open or you can attach to the scoping email.

The styling mirrors the NEUTRA UI: night-blue field, cyan vitals, amber
for surviving signal, red for collapse, mono numerals.
"""
from __future__ import annotations
import html
from .engine import AutopsyResult

C = dict(
    bg0="#0A1418", bg1="#0E1D22", panel="#0C1A20", edge="#1B3A42", edgeSoft="#152C33",
    cyan="#7DE3E0", cyanDim="#4FA9A6", ice="#CFF6F4",
    text="#AEC9CC", textDim="#6E8A8E", mut="#54706F",
    amber="#E0B24D", red="#D65A4E", redDim="#7C3A36", green="#6FC49B",
)

KIND_COLOR = {"reported": C["green"], "survives": C["amber"], "temporal": C["cyan"],
              "lookup": C["red"], "floor": C["mut"]}
FLAG_COLOR = {"SEVERE": C["red"], "LEAK": C["red"], "MODERATE": C["amber"],
              "PARTIAL": C["amber"], "THIN": C["amber"], "LOW": C["green"],
              "NET": C["cyan"], "TESTED": C["cyan"], "CLEAN": C["green"], "N/A": C["mut"]}


def _svg_ladder(ladder) -> str:
    pts = [r for r in ladder if r.get("r2") is not None]
    W, H, padL, padR, padT, padB = 720, 320, 50, 90, 30, 64
    band = H - padT - padB
    n = max(1, len(pts) - 1)
    step = (W - padL - padR) / n
    lo, hi = -0.3, 0.78

    def X(i): return padL + i * step
    def Y(r): return padT + (1 - (r - lo) / (hi - lo)) * band

    grid = ""
    for g in (0.75, 0.5, 0.25, 0.0, -0.25):
        y = Y(g)
        grid += (f'<line x1="{padL}" x2="{W-padR}" y1="{y:.1f}" y2="{y:.1f}" '
                 f'stroke="{C["edgeSoft"]}" stroke-width="1" stroke-dasharray="2 5"/>'
                 f'<text x="{padL-8}" y="{y+3.5:.1f}" text-anchor="end" '
                 f'font-family="monospace" font-size="10" fill="{C["mut"]}">{g:.2f}</text>')

    reported = next((r["r2"] for r in pts if r["kind"] == "reported"), None)
    refline = ""
    if reported is not None:
        y = Y(reported)
        refline = (f'<line x1="{padL}" x2="{W-padR}" y1="{y:.1f}" y2="{y:.1f}" '
                   f'stroke="{C["green"]}" stroke-width="1" stroke-dasharray="1 6" opacity="0.55"/>')

    poly = " ".join(f"{X(i):.1f},{Y(r['r2']):.1f}" for i, r in enumerate(pts))
    line = (f'<polyline points="{poly}" fill="none" stroke="{C["red"]}" '
            f'stroke-width="2" opacity="0.55"/>') if len(pts) > 1 else ""

    nodes = ""
    for i, r in enumerate(pts):
        col = KIND_COLOR.get(r["kind"], C["red"])
        x, y = X(i), Y(r["r2"])
        anchor = "start" if i == 0 else "end" if i == len(pts) - 1 else "middle"
        lx = x - 20 if i == 0 else x + 20 if i == len(pts) - 1 else x
        vx = x + 20 if i == 0 else x
        vy = y + 20 if i == 0 else y - 12
        model = html.escape(r["model"])
        cond = html.escape(r["condition"].split(" · ")[0])
        nodes += (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="none" stroke="{col}" stroke-width="1" opacity="0.4"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}" stroke="{C["bg0"]}" stroke-width="1.5"/>'
            f'<text x="{vx:.1f}" y="{vy:.1f}" text-anchor="middle" font-family="monospace" '
            f'font-size="14" font-weight="600" fill="{col}">{r["r2"]:.2f}</text>'
            f'<text x="{lx:.1f}" y="{H-26}" text-anchor="{anchor}" font-family="monospace" '
            f'font-size="10.5" fill="{C["text"]}">{model}</text>'
            f'<text x="{lx:.1f}" y="{H-12}" text-anchor="{anchor}" font-family="monospace" '
            f'font-size="9.5" fill="{C["mut"]}">{cond}</text>')

    return (f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block">'
            f'{grid}{refline}{line}{nodes}</svg>')


def _strip(verdict) -> str:
    cells = [("REPORTED", verdict["reported"], C["green"]),
             ("LOOKUP", verdict["lookup_random"], C["red"]),
             ("SURVIVES", verdict["survives_scaffold"], C["amber"]),
             ("LEARNED", verdict["learned_beyond_lookup"], C["cyan"])]
    out = ""
    for k, val, t in cells:
        disp = "——" if val is None else f"{val:.2f}"
        out += (f'<div style="background:{C["panel"]};padding:11px 12px">'
                f'<div style="font-size:9.5px;font-weight:600;letter-spacing:.14em;color:{C["textDim"]};margin-bottom:5px">{k}</div>'
                f'<div style="font-family:monospace;font-size:20px;color:{t}">{disp}</div></div>')
    return out


def _readout(cards) -> str:
    out = ""
    for c in cards:
        flag = c.get("flag", "")
        col = FLAG_COLOR.get(flag, C["cyan"])
        if "value_pct" in c:
            disp = f'{c["value_pct"]}%'
        elif c.get("value") is None:
            disp = "——"
        else:
            disp = f'{c["value"]:.2f}'
        out += (
            f'<div style="margin-bottom:16px">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
            f'<span style="font-size:11px;font-weight:600;letter-spacing:.1em;color:{C["text"]}">{html.escape(c["signal"])}</span>'
            f'<span style="font-family:monospace;font-size:10px;font-weight:600;color:{col};'
            f'border:1px solid {col}44;border-radius:20px;padding:1px 8px;background:{col}12">{html.escape(flag)}</span>'
            f'</div>'
            f'<div style="font-family:monospace;font-size:24px;color:{col};line-height:1;margin-bottom:8px">{disp}</div>'
            f'<div style="font-size:11.5px;color:{C["mut"]};line-height:1.5">{html.escape(c["note"])}</div>'
            f'</div>')
    return out


def _specimen(s) -> str:
    rows = [("Compounds", f'{s["n_compounds"]:,}', "on the slab", False),
            ("Scaffold series", str(s["n_scaffold_series"]), "Bemis–Murcko cores", False),
            ("Singleton series", str(s["n_singleton_series"]), "seen once — only memorisable", True),
            ("Largest series", str(s["largest_series"]), f'{s["largest_series_pct"]}% of data', False),
            ("Target", f'{s["target_mean"]}', f'± {s["target_sd"]} sd', False),
            ("Duplicates", str(s["exact_duplicate_smiles"]), "exact SMILES repeats", False)]
    out = ""
    for k, v, sub, hot in rows:
        col = C["amber"] if hot else C["ice"]
        out += (f'<div style="background:{C["panel"]};padding:13px 14px">'
                f'<div style="font-size:10.5px;font-weight:600;letter-spacing:.08em;color:{C["textDim"]};margin-bottom:6px">{k.upper()}</div>'
                f'<div style="font-family:monospace;font-size:22px;color:{col};line-height:1">{v}</div>'
                f'<div style="font-size:11px;color:{C["mut"]};margin-top:5px">{sub}</div></div>')
    return out


def render_html(res: AutopsyResult, source: str = "") -> str:
    v, s, m = res.verdict, res.specimen, res.meta
    src = html.escape(source)
    headline = html.escape(v["headline"])
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MODEL AUTOPSY — {src}</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,400;0,500;0,600;1,400;1,500&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{{color-scheme:dark}}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:'Barlow',system-ui,sans-serif;color:{C["text"]};
    background:radial-gradient(120% 90% at 50% -10%, {C["bg1"]} 0%, {C["bg0"]} 60%, #06090B 100%);
    padding:clamp(14px,3vw,34px)}}
  .wrap{{max-width:1220px;margin:0 auto;border:1px solid {C["edge"]};border-radius:14px;
    background:linear-gradient(180deg,{C["bg1"]}66,transparent 40%);
    box-shadow:0 0 0 1px #000,0 30px 80px -40px #000;padding:clamp(16px,2.4vw,28px)}}
  .grid{{display:grid;grid-template-columns:minmax(360px,1.9fr) minmax(260px,1.15fr);gap:14px;align-items:start}}
  @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
  .panel{{background:{C["panel"]};border:1px solid {C["edge"]};border-radius:12px;
    padding:clamp(13px,1.5vw,18px);box-shadow:inset 0 1px 0 {C["edgeSoft"]}}}
  .ptitle{{font-size:11px;font-weight:600;letter-spacing:.18em;color:{C["cyan"]};margin-bottom:14px}}
  .chart{{background:{C["bg0"]};border:1px solid {C["edge"]};border-radius:8px;padding:6px 4px}}
</style></head><body>
<div class="wrap">

  <header style="margin-bottom:18px">
    <div style="font-size:11px;font-weight:600;letter-spacing:.34em;color:{C["cyan"]};margin-bottom:4px">ACTARUSLAB</div>
    <div style="font-size:clamp(26px,4vw,40px);font-weight:500;letter-spacing:.42em;color:{C["ice"]};line-height:1;margin-bottom:8px">AUTOPSY</div>
    <div style="font-size:11px;font-weight:600;letter-spacing:.28em;color:{C["textDim"]};margin-bottom:6px">MODEL FORENSICS</div>
    <div style="font-style:italic;font-size:14px;color:{C["cyanDim"]}">Post-mortem of <b style="color:{C['text']};font-style:normal">{src}</b> — {s['n_compounds']} compounds, {s['n_scaffold_series']} scaffold series.</div>
  </header>

  <div class="grid">
    <section class="panel">
      <div class="ptitle">POST-MORTEM · DISTANCE FROM THE HONEST SCORE</div>
      <div class="chart">{_svg_ladder(res.ladder)}</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1px;margin-top:14px;border:1px solid {C['edge']};border-radius:8px;overflow:hidden;background:{C['edge']}">
        {_strip(v)}
      </div>
      <div style="margin-top:14px;padding:14px 16px;border-radius:8px;background:{C['panel']};border:1px solid {C['redDim']};border-left:2px solid {C['red']}">
        <div style="font-size:9.5px;font-weight:700;letter-spacing:.2em;color:{C['red']};margin-bottom:8px">VERDICT</div>
        <div style="font-size:13.5px;line-height:1.55;color:{C['text']}">{headline}</div>
      </div>
    </section>

    <section class="panel">
      <div class="ptitle">FORENSIC READOUT</div>
      {_readout(res.readout)}
    </section>
  </div>

  <div style="margin-top:14px">
    <section class="panel">
      <div class="ptitle">SPECIMEN X-RAY</div>
      <div style="font-size:13px;color:{C['textDim']};margin-bottom:14px;line-height:1.55;max-width:900px">
        {s['n_singleton_series']} of {s['n_scaffold_series']} scaffold series appear only once. A random split scatters near-identical analogues across train and test — that is where an inflated random-split score comes from.
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;border:1px solid {C['edge']};border-radius:8px;overflow:hidden;background:{C['edge']}">
        {_specimen(s)}
      </div>
    </section>
  </div>

  <div style="margin-top:16px;font-family:monospace;font-size:10.5px;color:{C['mut']};letter-spacing:.02em;line-height:1.7">
    {html.escape(m['featurisation'])} · {html.escape(m['model'])} · {m['k_folds']}-FOLD · POOLED OOF R² · GROUPKFOLD ON GENERIC SCAFFOLDS · TANIMOTO 1-NN · PERMUTATION CONTROL · SEED {m['seed']}
  </div>
</div>
</body></html>"""
