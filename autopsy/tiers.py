"""
MODEL AUTOPSY — what an audit says in public, and what it keeps back.

The engine computes the whole ladder every time. This module decides how much
of it leaves the process, and it is the only place that decision is made.

Two tiers
---------
free       the synthetic verdict: how much the reported score is inflated over
           the honest one, as a single percentage, plus an invitation to ask
           for the rest. Nothing else.
reserved   the audit as the engine built it — the ladder, the per-fold
           metrics, the scaffold composition, the readout cards, the
           reproducible validation report. Where the leakage comes from, not
           just that it is there.

`public_view` builds the free tier from scratch. That is deliberate and it is
the whole security argument: a whitelist that copies four values out of the
result cannot leak a fifth, whereas a filter that pops keys off the full dict
leaks the day someone adds a field to the engine and forgets this file exists.
The withheld part is never serialised at all — it is not hidden inside the
response under another name, not truncated, not encoded. There is nothing in
the JSON to inspect, because it was never put there.

Which tier a caller gets is decided in api.py, by whether they hold a live
subscription. The CLI is not affected: it runs the engine locally on a file
the operator already has, and prints everything.
"""
from __future__ import annotations

from typing import Optional

CONTACT_EMAIL = "actaruslab@proton.me"

# What the reserved tier holds, named so the free response can say what it is
# withholding without saying any of it. Prose, not data.
WITHHELD = [
    "dove si origina il leakage",
    "quali scaffold sono coinvolti",
    "il dettaglio per fold",
    "il report di validazione riproducibile",
]


def inflation(verdict: dict) -> dict:
    """The one number the free tier exposes: reported / honest - 1, in percent.

    The honest score is the scaffold-split R² — the model graded on chemical
    series it has never seen. When the dataset has too few distinct series to
    split on, the temporal split stands in; when neither ran there is no
    honest score and no ratio to take.

    Returns {"pct", "basis", "state"} where state is one of:

        inflated      the reported score is above the honest one — the
                      ordinary finding, and `pct` is how far above
        clean         the honest split holds the reported score up; `pct` is
                      zero or negative and there is nothing to sell
        collapse      the honest split scores at or below the mean predictor.
                      The ratio is unbounded, so `pct` is None: "infinitely
                      inflated over nothing" is not a number to put in front
                      of a customer, and the finding is worse than any
                      percentage anyway
        unmeasurable  no honest rung ran, or the model reported nothing to
                      inflate in the first place

    A whole number, because the second decimal of an inflation figure is
    precision this comparison does not have.
    """
    reported = verdict.get("reported")
    honest, basis = verdict.get("survives_scaffold"), "scaffold split"
    if honest is None:
        honest, basis = verdict.get("temporal"), "temporal split"
    if honest is None:
        return {"pct": None, "basis": None, "state": "unmeasurable"}
    if reported is None or reported <= 0:
        return {"pct": None, "basis": basis, "state": "unmeasurable"}
    if honest <= 0:
        return {"pct": None, "basis": basis, "state": "collapse"}

    pct = int(round(100.0 * (reported / honest - 1.0)))
    return {"pct": pct, "basis": basis,
            "state": "inflated" if pct > 0 else "clean"}


def contact_message(inf: dict) -> str:
    """The invitation that stands where the subscribe button used to.

    The percentage is interpolated from the audit that just ran — the same
    value the response carries, so the message can never quote a number the
    result does not support. The three other openings are for the audits that
    do not produce one; the body and the address are the same in all four.
    """
    state = inf["state"]
    if state == "inflated":
        opening = f"Il tuo pipeline mostra un'inflazione del +{inf['pct']}%."
    elif state == "clean":
        drop = abs(inf["pct"] or 0)
        opening = ("Il tuo pipeline non mostra inflazione: il punteggio riportato "
                   "regge lo split onesto"
                   + (f" (−{drop}%)." if drop else "."))
    elif state == "collapse":
        opening = ("Il tuo pipeline non sopravvive a uno split onesto: sulle serie "
                   "chimiche che non ha mai visto non batte la media. "
                   "L'inflazione non è una percentuale, è l'intero punteggio.")
    else:
        opening = ("Il tuo pipeline non produce un confronto onesto misurabile su "
                   "questo dataset.")

    return (
        f"{opening}\n\n"
        "Questo è il verdetto sintetico. La diagnosi completa — dove si origina "
        "il leakage, quali scaffold sono coinvolti, il dettaglio per fold e il "
        "report di validazione riproducibile — è disponibile su richiesta.\n\n"
        f"Scrivici: {CONTACT_EMAIL}"
    )


def public_view(result: dict) -> dict:
    """The free tier, built by hand from the full audit.

    Every key in the returned dict is written out below. Adding a rung, a
    metric or a specimen field to the engine cannot widen this: it has to be
    typed in here to get out.

    `warnings` is the one thing carried over from the audit itself, and it is
    not diagnosis — it says how many of the caller's own rows were unreadable
    and therefore how much of their file these numbers describe. Withholding
    that would mean quietly auditing 470 of 500 rows and reporting a
    percentage as if it covered the file.
    """
    inf = inflation(result.get("verdict") or {})
    return {
        "tier": "free",
        "inflation_pct": inf["pct"],
        "inflation_basis": inf["basis"],
        "inflation_state": inf["state"],
        "warnings": result.get("warnings") or [],
        "contact": {
            "email": CONTACT_EMAIL,
            "message": contact_message(inf),
            "withheld": list(WITHHELD),
        },
    }


def reserved_view(result: dict) -> dict:
    """The audit as the engine built it, marked with the tier it came from."""
    return {"tier": "reserved", **result}


def _is_subscriber(subscriber) -> bool:
    """Does this object actually carry a subscription, or is it merely not None?

    `is not None` is too weak a test to hang the diagnosis on. A dependency
    that was never resolved, a sentinel, a stray truthy default — all of them
    pass it, and each one would hand a stranger the full audit. A subscriber
    is a row with a key hash in it; anything that cannot produce one is not
    one, and the failure direction is the free tier.
    """
    if subscriber is None:
        return False
    try:
        return bool(subscriber["key_hash"])
    except Exception:
        return False


def view_for(result: Optional[dict], subscriber) -> Optional[dict]:
    """Which tier this caller gets. A live subscription buys the diagnosis;
    everything else — anonymous, free, paywall switched off — gets the
    verdict and the address to write to."""
    if result is None:
        return None
    return reserved_view(result) if _is_subscriber(subscriber) else public_view(result)
