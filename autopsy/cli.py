"""
MODEL AUTOPSY — command-line audit.

    python -m autopsy.cli data.csv --smiles smiles --y pIC50 [--date document_year]
                          [--out report.html] [--json result.json]

CSV in, forensic verdict to the terminal, and a self-contained NEUTRA-styled
HTML report you can hand to a client. This is the shape of the one-time audit:
point it at a real target, read the collapse.
"""
from __future__ import annotations
import argparse, json, sys
import pandas as pd

from .engine import run_autopsy, AutopsyError
from .report import render_html


# ── terminal colours (no dependency) ─────────────────────────────────
class T:
    dim = "\033[2m"; bold = "\033[1m"; end = "\033[0m"
    cyan = "\033[96m"; amber = "\033[93m"; red = "\033[91m"; green = "\033[92m"; grey = "\033[90m"


def _bar(label, r2, ref):
    if r2 is None:
        return f"  {label:<34} {T.grey}——   (not run){T.end}"
    width = 26
    filled = max(0, min(width, int(round((r2 + 0.3) / 1.05 * width))))
    tone = T.green if r2 >= ref - 0.02 else T.amber if r2 >= 0.4 else T.red
    bar = tone + "█" * filled + T.grey + "·" * (width - filled) + T.end
    return f"  {label:<34} {bar} {tone}{r2:>6.2f}{T.end}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autopsy", description="Forensic validation for a QSAR model.")
    ap.add_argument("csv", help="input CSV of molecules")
    ap.add_argument("--smiles", required=True, help="column with SMILES")
    ap.add_argument("--y", required=True, help="column with activity (IC50/pIC50/…)")
    ap.add_argument("--date", default=None, help="optional column with assay date/year (activates temporal rung)")
    ap.add_argument("--k", type=int, default=5, help="CV folds (default 5)")
    ap.add_argument("--out", default=None, help="write HTML report here")
    ap.add_argument("--json", default=None, help="write raw result JSON here")
    ap.add_argument("--quiet", action="store_true", help="suppress progress lines")
    args = ap.parse_args(argv)

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"{T.red}Could not read CSV:{T.end} {e}", file=sys.stderr); return 2

    log = (lambda m: None) if args.quiet else (lambda m: print(f"{T.grey}· {m}{T.end}", file=sys.stderr))

    try:
        res = run_autopsy(df, args.smiles, args.y, args.date, k=args.k, log=log)
    except AutopsyError as e:
        print(f"{T.red}Input error:{T.end} {e}", file=sys.stderr); return 1

    v, s = res.verdict, res.specimen

    # ── terminal verdict ────────────────────────────────────────────
    print()
    print(f"{T.cyan}{T.bold}  MODEL AUTOPSY — {args.csv}{T.end}")
    print(f"{T.grey}  {s['n_compounds']} compounds · {s['n_scaffold_series']} scaffold series "
          f"· {s['n_singleton_series']} singletons · target {s['target_mean']}±{s['target_sd']}{T.end}")
    print()
    print(f"{T.dim}  LEAKAGE LADDER — pooled out-of-fold R²{T.end}")
    ref = v["reported"]
    for rung in res.ladder:
        label = f"{rung['model']} · {rung['condition']}"
        print(_bar(label, rung.get("r2"), ref))
    print()

    # ── headline ────────────────────────────────────────────────────
    if v["lookup_pct_of_reported"] is not None:
        print(f"  {T.red}{T.bold}{v['lookup_pct_of_reported']}% of the reported "
              f"{v['reported']:.2f} is a similarity lookup.{T.end}")
    if v["survives_scaffold"] is not None:
        print(f"  {T.amber}Survives a new chemical series: {v['survives_scaffold']:.2f}{T.end}")
    if v["learned_beyond_lookup"] is not None:
        print(f"  {T.cyan}Learned beyond lookup: {v['learned_beyond_lookup']:.3f}{T.end}")
    if v["temporal"] is not None:
        print(f"  {T.cyan}Holds forward in time: {v['temporal']:.2f}{T.end}")
    if v["permutation_floor"] is not None and v["permutation_floor"] > 0.05:
        print(f"  {T.red}⚠ permutation floor {v['permutation_floor']:.2f} > 0 — pipeline may be leaking{T.end}")
    print()
    print(f"{T.grey}  {v['headline']}{T.end}")
    print()

    # ── artefacts ───────────────────────────────────────────────────
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res.to_dict(), f, indent=2)
        print(f"{T.green}  ✓ wrote {args.json}{T.end}", file=sys.stderr)
    if args.out:
        with open(args.out, "w") as f:
            f.write(render_html(res, source=args.csv))
        print(f"{T.green}  ✓ wrote {args.out}{T.end}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
