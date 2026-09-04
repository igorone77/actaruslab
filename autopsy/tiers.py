"""
MODEL AUTOPSY — what an audit says in public, and what it keeps back.

The engine computes the whole ladder every time. This module decides how much
of it leaves the process, and it is the only place that decision is made.

Two tiers
---------
full       the audit as the engine built it — the ladder, the per-fold
           metrics, the scaffold composition, the readout cards, the
           reproducible validation report. Where the leakage comes from, not
           just that it is there. **This is the default.**
verdict    the synthetic verdict alone: how much the reported score is
           inflated over the honest one, as a single percentage, plus an
           invitation to ask for the rest.

Two switches, and they are not the same switch
----------------------------------------------
    AUTOPSY_PAYWALL_ENABLED   may an audit run at all without paying.
                              Default false — see billing.paywall_enabled.
    AUTOPSY_VERDICT_ONLY      what an audit hands back to someone who is not
                              a subscriber. Default false: everything.

They were one switch, and that was a design mistake worth naming here so it
is not repeated. "Turn the paywall off" plainly reads as "give people the
product", but the flag only stopped the 402 and left the output withheld, so
a deployment with the paywall visibly off still served a percentage and
nothing else. One flag was answering two questions. Now each question has its
own, and either can be set without touching the other:

    (default)                      free, and the whole diagnosis
    AUTOPSY_VERDICT_ONLY=true      free to run, percentage only — the
                                   lead-generating showcase
    AUTOPSY_PAYWALL_ENABLED=true   the paid product: 402 without a
                                   subscription, whole diagnosis with one

A live subscription always reads the full audit, whatever AUTOPSY_VERDICT_ONLY
says: it is what was paid for.

`public_view` builds the verdict tier from scratch. That is deliberate and it
is the whole security argument: a whitelist that copies four values out of the
result cannot leak a fifth, whereas a filter that pops keys off the full dict
leaks the day someone adds a field to the engine and forgets this file exists.
The withheld part is never serialised at all — it is not hidden inside the
response under another name, not truncated, not encoded. There is nothing in
the JSON to inspect, because it was never put there. That property is intact
and one variable away; it is switched off, not removed.

The CLI is not affected by any of this: it runs the engine locally on a file
the operator already has, and prints everything.
"""
from __future__ import annotations

import os
from typing import Optional

# Same spellings the paywall flag accepts, so the two read alike in a deploy
# config and neither surprises someone who has met the other.
_TRUE = {"1", "true", "yes", "on"}


def verdict_only() -> bool:
    """Is the diagnosis withheld from callers without a subscription?

    Default false: an audit returns everything it found. Set

        AUTOPSY_VERDICT_ONLY=true

    to serve the synthetic verdict alone and invite the reader to write in —
    the lead-generating showcase.

    Read from the environment on every call, not bound at import, so a
    deployment's setting cannot be shadowed by whatever happened to be set
    when this module was first loaded.
    """
    return os.getenv("AUTOPSY_VERDICT_ONLY", "false").strip().lower() in _TRUE

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
    """The verdict tier, built by hand from the full audit.

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
        "tier": "verdict",
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


def full_view(result: dict) -> dict:
    """The audit as the engine built it, marked with the tier it came from."""
    return {"tier": "full", **result}


# The name this used to have, kept because it reads well at the call site and
# because renaming a function is not a reason to break anything importing it.
reserved_view = full_view


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
    """Which tier this caller gets.

    A live subscription always reads the whole audit — that is what it bought,
    and no deployment setting takes it away. Everyone else reads the whole
    audit too, unless this deployment has asked to withhold it.

    Note which switch is *not* consulted here: the paywall decides whether an
    audit may run, not what it says once it has. Keeping the two apart is the
    entire point of AUTOPSY_VERDICT_ONLY existing separately.
    """
    if result is None:
        return None
    if _is_subscriber(subscriber):
        return full_view(result)
    return public_view(result) if verdict_only() else full_view(result)
