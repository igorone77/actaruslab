"""
The two tiers, and who owns a result.

Autopsy is free and the engine is open. What it hands back is not: the free
tier is the synthetic verdict — one percentage — and the diagnosis behind it
is reserved. Two properties are worth a test each, because both are the kind
that rot silently:

  · the withheld part must not be in the response *at all*. Not hidden under
    another key, not rounded, not encoded. Inspecting the raw JSON must not
    get anybody past the split.
  · a free audit is still someone's data. The payment gate is off by default;
    the ownership check is not, and never is.

tests/test_billing.py covers the paywall itself, in the on position.
"""
import json
import time
import uuid

import pytest
from fastapi import HTTPException

from autopsy import api, tiers

# A full audit, shaped exactly as the engine returns one. Hand-built rather
# than computed: these tests are about what leaves the process, and a
# 20-second BACE run would only make them slower, not stronger.
FULL = {
    "specimen": {"n_compounds": 300, "n_rows_in": 300, "n_unparseable_dropped": 0,
                 "n_activity_dropped": 0, "n_dropped_total": 0, "dropped_pct": 0.0,
                 "n_scaffold_series": 217, "n_singleton_series": 183,
                 "largest_series": 21, "largest_series_pct": 7.0,
                 "target_mean": 6.482, "target_sd": 1.348,
                 "exact_duplicate_smiles": 0, "has_dates": False},
    "ladder": [
        {"model": "XGBoost", "condition": "random split", "r2": 0.709,
         "rmse": 0.727, "spearman": 0.843, "kind": "reported", "note": None},
        {"model": "1-NN lookup", "condition": "random split", "r2": 0.572,
         "rmse": 0.881, "spearman": 0.771, "kind": "lookup", "note": None},
        {"model": "XGBoost", "condition": "scaffold split · 217 series", "r2": 0.597,
         "rmse": 0.856, "spearman": 0.782, "kind": "survives", "note": None},
        {"model": "1-NN lookup", "condition": "scaffold split", "r2": 0.451,
         "rmse": 0.999, "spearman": 0.684, "kind": "lookup", "note": None},
        {"model": "Permutation", "condition": "shuffled target", "r2": -0.222,
         "rmse": 1.489, "spearman": 0.011, "kind": "floor", "note": None},
    ],
    "verdict": {"reported": 0.709, "lookup_random": 0.572, "survives_scaffold": 0.597,
                "lookup_scaffold": 0.451, "learned_beyond_lookup": 0.146,
                "temporal": None, "permutation_floor": -0.222,
                "lookup_pct_of_reported": 81,
                "headline": "81% of the reported score is reproducible by a "
                            "nearest-neighbour lookup."},
    "readout": [{"signal": "Learned structure", "flag": "THIN", "value": 0.146,
                 "value_pct": None,
                 "note": "the model adds 0.146 over a lookup table."}],
    "warnings": [],
    "meta": {"featurisation": "ECFP4 · 2048 bit", "model": "XGBoost (400 trees, depth 6)",
             "k_folds": 5, "seed": 0,
             "columns": {"smiles": "smiles", "target": "pIC50", "date": None}},
}


# ── what the free tier is ─────────────────────────────────────────────

def test_the_free_tier_is_the_percentage_and_nothing_else():
    free = tiers.public_view(FULL)
    assert set(free) == {"tier", "inflation_pct", "inflation_basis",
                         "inflation_state", "warnings", "contact"}
    assert free["tier"] == "free"
    assert free["inflation_pct"] == 19          # 0.709 / 0.597 - 1
    assert free["inflation_basis"] == "scaffold split"


def test_the_diagnosis_is_not_in_the_response_to_be_found():
    """The point of the split, stated as a property of the bytes on the wire.

    Not 'the UI does not show it' and not 'the keys are renamed' — the
    detailed audit is never serialised, so there is nothing in the payload to
    inspect, decode or reconstruct.
    """
    free = tiers.public_view(FULL)
    raw = json.dumps(free)

    # One methodological name is exposed on purpose: `inflation_basis` says
    # which split the percentage is measured against. The method is published
    # in the README and demonstrated in full on the demo, so naming it is not
    # naming the finding — and nothing this dataset produced travels with it.
    basis = free["inflation_basis"]

    # every number the ladder measured, and every model that measured it
    for rung in FULL["ladder"]:
        for metric in ("r2", "rmse", "spearman"):
            assert str(rung[metric]) not in raw, f"{rung['kind']}.{metric} leaked"
        assert rung["model"] not in raw
        if rung["condition"] != basis:
            assert rung["condition"] not in raw, f"{rung['kind']}.condition leaked"

    assert "217" not in raw, "the scaffold series count leaked"

    # every number and sentence the verdict composed
    for key, value in FULL["verdict"].items():
        if value is None:
            continue
        assert str(value) not in raw, f"verdict.{key} leaked"

    # the composition of the dataset, and the readout cards
    for key in ("n_scaffold_series", "n_singleton_series", "largest_series",
                "target_mean", "target_sd"):
        assert str(FULL["specimen"][key]) not in raw, f"specimen.{key} leaked"
    for card in FULL["readout"]:
        assert card["note"] not in raw and card["flag"] not in raw

    # and no branch of the full result survives under any name
    for key in ("specimen", "ladder", "verdict", "readout", "meta"):
        assert key not in json.loads(raw)


def test_the_free_view_is_built_not_filtered():
    """A whitelist cannot leak a field that did not exist when it was written.
    A filter can, and this is the test that tells the two apart: give the
    engine a new branch and see whether it comes out."""
    grown = dict(FULL, secret_new_branch={"per_fold": [0.61, 0.58, 0.60]})
    grown["verdict"] = dict(FULL["verdict"], new_metric=0.99)
    raw = json.dumps(tiers.public_view(grown))
    assert "secret_new_branch" not in raw and "per_fold" not in raw
    assert "new_metric" not in raw and "0.99" not in raw


def test_dropped_rows_stay_visible_because_they_describe_the_upload():
    """How much of their own file was unreadable is the caller's business, not
    ours — a percentage computed on 470 of 500 rows must say so."""
    warned = dict(FULL, warnings=[{"level": "severe", "text": "30 of 500 rows dropped.",
                                   "n_dropped": 30, "n_rows_in": 500, "pct": 6.0}])
    assert tiers.public_view(warned)["warnings"][0]["n_dropped"] == 30


# ── the number itself ─────────────────────────────────────────────────

@pytest.mark.parametrize("verdict,pct,state", [
    ({"reported": 0.709, "survives_scaffold": 0.597}, 19, "inflated"),
    ({"reported": 0.90, "survives_scaffold": 0.30}, 200, "inflated"),
    ({"reported": 0.50, "survives_scaffold": 0.60}, -17, "clean"),
    ({"reported": 0.60, "survives_scaffold": 0.60}, 0, "clean"),
    ({"reported": 0.70, "survives_scaffold": -0.05}, None, "collapse"),
    ({"reported": 0.70, "survives_scaffold": 0.0}, None, "collapse"),
    ({"reported": -0.1, "survives_scaffold": 0.4}, None, "unmeasurable"),
    ({"reported": 0.70, "survives_scaffold": None, "temporal": None}, None, "unmeasurable"),
])
def test_inflation_reads_every_shape_an_audit_can_take(verdict, pct, state):
    inf = tiers.inflation(verdict)
    assert (inf["pct"], inf["state"]) == (pct, state)


def test_the_temporal_split_stands_in_when_there_are_no_scaffold_series():
    inf = tiers.inflation({"reported": 0.70, "survives_scaffold": None, "temporal": 0.40})
    assert inf["basis"] == "temporal split" and inf["pct"] == 75


# ── the invitation ────────────────────────────────────────────────────

def test_the_contact_message_carries_the_real_percentage():
    """X interpolated, never literal, and never a number the audit did not
    produce."""
    msg = tiers.public_view(FULL)["contact"]["message"]
    assert "+19%" in msg
    assert "X%" not in msg
    assert "actaruslab@proton.me" in msg
    assert "disponibile su richiesta" in msg


def test_the_message_says_something_true_when_there_is_no_percentage():
    for verdict in ({"reported": 0.70, "survives_scaffold": -0.05},
                    {"reported": 0.50, "survives_scaffold": 0.60},
                    {"reported": 0.70, "survives_scaffold": None, "temporal": None}):
        msg = tiers.contact_message(tiers.inflation(verdict))
        assert "X%" not in msg and "None" not in msg
        assert "actaruslab@proton.me" in msg


def test_the_free_response_offers_no_way_to_pay():
    """No Checkout link, no price, no subscribe button while the paywall is
    off — the call to action is an email address."""
    raw = json.dumps(tiers.public_view(FULL)).lower()
    for word in ("checkout", "stripe", "subscribe", "199", "€"):
        assert word not in raw


# ── the reserved tier ─────────────────────────────────────────────────

def test_a_subscriber_gets_the_audit_whole():
    out = tiers.view_for(FULL, {"key_hash": "abc"})
    assert out["tier"] == "reserved"
    assert out["ladder"] == FULL["ladder"]
    assert out["verdict"]["headline"] == FULL["verdict"]["headline"]


def test_anything_that_is_not_a_subscription_gets_the_free_tier():
    """`is not None` is not the test. An unresolved dependency, a sentinel, a
    stray truthy default — none of them buy the diagnosis."""
    for impostor in (None, object(), "yes", 1, {}, {"email": "a@b.c"}):
        assert tiers.view_for(FULL, impostor)["tier"] == "free"


# ── ownership, which the paywall does not govern ──────────────────────

def _job(result=None, key_hash=None):
    """A finished job in the service's own dict, with the owner submit mints."""
    jid, token = uuid.uuid4().hex, api.issue_job_token()
    api._jobs[jid] = {"status": "done", "progress": "complete",
                      "result": result or FULL, "error": None,
                      "created": time.time(), "started": time.time(),
                      "finished": time.time(), "rows": 300,
                      "key_hash": key_hash, "owner_hash": api._token_hash(token)}
    return jid, token


def test_the_owner_of_a_free_audit_can_read_it(monkeypatch):
    monkeypatch.delenv("AUTOPSY_PAYWALL_ENABLED", raising=False)
    jid, token = _job()
    out = api.job_status(jid, sub=None, x_job_token=token)
    assert out["result"]["inflation_pct"] == 19


def test_nobody_else_can_read_it_even_with_the_paywall_off(monkeypatch):
    """The job id is the only thing an attacker could have, and it is not
    enough. 404 and not 403: a 403 would confirm the id belongs to someone."""
    monkeypatch.delenv("AUTOPSY_PAYWALL_ENABLED", raising=False)
    jid, _ = _job()
    for wrong in ("", "not-the-token", api.issue_job_token()):
        with pytest.raises(HTTPException) as e:
            api.job_status(jid, sub=None, x_job_token=wrong)
        assert e.value.status_code == 404
        assert "unknown job" in e.value.detail


def test_one_subscriber_cannot_read_another_subscribers_job():
    jid, _ = _job(key_hash="hash_of_alices_key")
    with pytest.raises(HTTPException) as e:
        api.job_status(jid, sub={"key_hash": "hash_of_bobs_key"}, x_job_token="")
    assert e.value.status_code == 404
    api.job_status(jid, sub={"key_hash": "hash_of_alices_key"}, x_job_token="")


def test_a_job_nobody_owns_is_readable_by_nobody():
    """The safe direction to fail. An entry point that forgets to record an
    owner locks its own results rather than publishing everyone else's."""
    jid = uuid.uuid4().hex
    api._jobs[jid] = {"status": "done", "progress": "complete", "result": FULL,
                      "error": None, "created": time.time(), "started": time.time(),
                      "finished": time.time(), "rows": 300,
                      "key_hash": None, "owner_hash": None}
    with pytest.raises(HTTPException) as e:
        api.job_status(jid, sub=None, x_job_token=api.issue_job_token())
    assert e.value.status_code == 404


def test_the_token_is_never_kept_in_the_clear():
    """Same rule as the API key: this process holds a hash, so a memory dump
    or a leaked job record does not hand over the result."""
    _, token = _job()
    assert all(token != j.get("owner_hash") for j in api._jobs.values())
    assert any(api._token_hash(token) == j.get("owner_hash") for j in api._jobs.values())
