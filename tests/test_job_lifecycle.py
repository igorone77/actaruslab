"""
Submit, then poll — over real HTTP, in both paywall positions.

This file exists because of a bug the rest of the suite could not see. The
other tests call the endpoint functions directly and pass `sub` themselves,
so nothing was exercising what a client actually sends: headers, cookies, and
the fact that the two requests are separate. A job could be alive and
computing while every poll answered 404, and 102 green tests said nothing.

So these drive the app through TestClient, which keeps a cookie jar the way a
browser does. The property under test is the one the product rests on: a job
submitted is a job that can immediately be read back by the client that
submitted it — and by nobody else.
"""
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autopsy import api, billing

BACE = pytest.importorskip("pathlib").Path(__file__).resolve().parent.parent / "bace.csv"

ALIVE = {"queued", "running", "done"}


@pytest.fixture(scope="module")
def csv_bytes():
    """Small enough that the audit finishes in about a second, large enough
    that the engine accepts it."""
    df = pd.read_csv(BACE, usecols=["smiles", "pIC50"]).head(60)
    return df.to_csv(index=False).encode()


def _submit(client, csv_bytes):
    """Always assert the 202 here. These tests all share one process-wide
    queue, and a submit rejected for depth would otherwise surface much later
    as a missing cookie or a mystery 404 — which is exactly the class of
    confusion this file was written to end."""
    res = client.post("/autopsy/jobs",
                      files={"file": ("s.csv", csv_bytes, "text/csv")},
                      data={"smiles": "smiles", "y": "pIC50", "k": "3"})
    assert res.status_code == 202, f"{res.status_code}: {res.text}"
    return res


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """Two things these tests must not inherit from whatever ran before.

    The queue ceiling is a capacity rule tested elsewhere; here it would only
    couple these tests to how fast the worker drains between them.

    AUTOPSY_PUBLIC_URL decides whether the session cookie is marked Secure,
    and TestClient speaks http — so a leaked https value silently drops every
    cookie and these tests fail as 404s with no hint why. That is not
    hypothetical either: test_billing.py reloads the billing module with an
    https URL patched in, and a reloaded module constant outlives the
    monkeypatch that set it.
    """
    monkeypatch.setattr(api, "QUEUE_DEPTH", 100)
    monkeypatch.delenv("AUTOPSY_PUBLIC_URL", raising=False)

    # One partition per rung. Every test in this file queues a real audit onto
    # a single worker, so they serialise; at the default five partitions the
    # backlog outran the poll budget and three tests failed for being behind a
    # queue rather than for anything they assert. What is tested here is the
    # HTTP path — tiers, ownership, the mode report — and none of it depends on
    # how many partitions the statistics were measured over. The bands
    # themselves are tests/test_methodology.py and tests/test_bace_smoke.py.
    monkeypatch.setenv("AUTOPSY_REPEATS", "1")


@pytest.fixture()
def free(monkeypatch):
    """The deployed configuration: no flag, no key, no subscription."""
    monkeypatch.delenv("AUTOPSY_PAYWALL_ENABLED", raising=False)
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    with TestClient(api.app) as c:
        yield c


@pytest.fixture()
def paid(tmp_path, monkeypatch):
    """The paid configuration, and a live key to go with it. DB_PATH is read
    at connect time, so pointing the module at a tmp file is enough — no
    reload, which would leave api.py holding the old dependency."""
    monkeypatch.setenv("AUTOPSY_PAYWALL_ENABLED", "true")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setattr(billing, "DB_PATH", str(tmp_path / "subs.db"))
    billing.init_db()
    key = billing.create_subscriber("t@t.t", "cus_t", "sub_t", "active",
                                    time.time() + 86400)
    with TestClient(api.app) as c:
        c.headers.update({"Authorization": f"Bearer {key}"})
        yield c


# ── the regression ────────────────────────────────────────────────────

def test_submit_then_immediate_poll_finds_the_job_paywall_off(free, csv_bytes):
    """202 then 404 was the report. The job never died: the poll arrived
    without the claim that owns it, and the answer for that is the same 404
    as for a job that never existed."""
    sub = _submit(free, csv_bytes)
    job_id = sub.json()["job_id"]

    poll = free.get(f"/autopsy/jobs/{job_id}",
                    headers={"X-Job-Token": sub.json()["job_token"]})
    assert poll.status_code == 200, poll.text
    assert poll.json()["status"] in ALIVE


def test_submit_then_immediate_poll_finds_the_job_paywall_on(paid, csv_bytes):
    sub = _submit(paid, csv_bytes)
    job_id = sub.json()["job_id"]

    poll = paid.get(f"/autopsy/jobs/{job_id}",
                    headers={"X-Job-Token": sub.json()["job_token"]})
    assert poll.status_code == 200, poll.text
    assert poll.json()["status"] in ALIVE


def test_the_browser_polls_on_its_cookie_alone(free, csv_bytes):
    """The actual fix. A page that does not know to send X-Job-Token — an
    older bundle out of a browser cache, a reload that dropped the token, a
    second tab — still reads its own audit, because the cookie set at submit
    goes back on its own."""
    sub = _submit(free, csv_bytes)
    assert api.SESSION_COOKIE in sub.cookies or api.SESSION_COOKIE in free.cookies

    poll = free.get(f"/autopsy/jobs/{sub.json()['job_id']}")   # no header at all
    assert poll.status_code == 200, poll.text
    assert poll.json()["status"] in ALIVE


def test_the_cookie_survives_a_reload_and_a_second_tab(free, csv_bytes):
    """Same browser, new page object: the audit is still readable. This is
    what the token in a closure could not do."""
    job_id = _submit(free, csv_bytes).json()["job_id"]
    session = free.cookies.get(api.SESSION_COOKIE)

    with TestClient(api.app, cookies={api.SESSION_COOKIE: session}) as tab2:
        assert tab2.get(f"/autopsy/jobs/{job_id}").status_code == 200


def test_one_session_reads_every_audit_it_started(free, csv_bytes):
    first = _submit(free, csv_bytes).json()["job_id"]
    second = _submit(free, csv_bytes).json()["job_id"]
    for job_id in (first, second):
        assert free.get(f"/autopsy/jobs/{job_id}").status_code == 200


# ── and still nobody else ─────────────────────────────────────────────

def test_a_different_browser_still_gets_404(free, csv_bytes):
    """The fix must not have turned ownership off. A client with no cookie,
    no token and no key holds nothing, and the job id alone is not a claim."""
    job_id = _submit(free, csv_bytes).json()["job_id"]

    with TestClient(api.app) as stranger:
        out = stranger.get(f"/autopsy/jobs/{job_id}")
    assert out.status_code == 404
    assert "unknown job" in out.json()["detail"]


def test_a_stolen_job_id_with_a_wrong_cookie_gets_404(free, csv_bytes):
    job_id = _submit(free, csv_bytes).json()["job_id"]
    forged = {api.SESSION_COOKIE: api.issue_job_token()}
    with TestClient(api.app, cookies=forged) as attacker:
        assert attacker.get(f"/autopsy/jobs/{job_id}").status_code == 404


def test_the_404_no_longer_explains_the_wrong_cause(free):
    """It used to say the job had expired or the service had restarted, which
    sent the reader to look for a bug in the job store. It has to name the
    claim too — and say the same thing for every id, or the message itself
    becomes the leak."""
    out = free.get("/autopsy/jobs/definitely-not-a-job")
    assert out.status_code == 404
    assert "did not carry the claim" in out.json()["detail"]


def test_the_session_cookie_is_not_readable_by_script(free, csv_bytes):
    """HttpOnly and SameSite, or the cookie would be a downgrade on the
    header it backs up rather than a repair."""
    sub = _submit(free, csv_bytes)
    raw = sub.headers["set-cookie"].lower()
    assert "httponly" in raw and "samesite=lax" in raw


def test_the_cookie_is_secure_on_an_https_deployment(free, csv_bytes, monkeypatch):
    """And not on a localhost one, or the laptop install loses every cookie
    it sets. Both positions pinned, because getting either wrong is silent."""
    assert "secure" not in _submit(free, csv_bytes).headers["set-cookie"].lower()

    monkeypatch.setenv("AUTOPSY_PUBLIC_URL", "https://autopsy.example.org")
    assert "secure" in _submit(free, csv_bytes).headers["set-cookie"].lower()


# ── what an audit hands back, over HTTP, in each configuration ────────
# The switch that decides this is not the paywall switch, and conflating the
# two is what put a deployment with the paywall visibly off into verdict-only
# mode with no way to see why. Each position is pinned here on a real audit
# of a real file, so "the full diagnosis" means the rungs of *that* file
# rather than a shape assertion that a demo payload could satisfy.

# Every test in this file shares one worker thread and one process-wide queue,
# so a job waits behind whatever was submitted before it. Budget in wall-clock
# seconds, generously, and only for the two tests that genuinely need a real
# audit — a flaky test is worse than no test, and an earlier version of this
# helper counted iterations and failed whenever the machine was busy.
_AUDIT_BUDGET_S = 180


def _finish(client, csv_bytes):
    """Submit and poll to completion. Returns the result the client sees."""
    sub = _submit(client, csv_bytes)
    token = {"X-Job-Token": sub.json()["job_token"]}
    deadline = time.monotonic() + _AUDIT_BUDGET_S
    while time.monotonic() < deadline:
        out = client.get(f"/autopsy/jobs/{sub.json()['job_id']}", headers=token)
        assert out.status_code == 200, out.text
        body = out.json()
        if body["status"] == "done":
            return body["result"]
        assert body["status"] != "failed", body
        time.sleep(0.2)
    raise AssertionError(f"the audit did not finish in {_AUDIT_BUDGET_S}s")


def _finished_job(client, result):
    """A job already done, read back through the endpoint.

    The tier decision is made when the result is served, not when it is
    computed, so anything about *which* tier a caller gets can be asserted on
    a planted result. Only the two tests that check the audit's own contents
    need to pay for a real one.
    """
    import uuid
    jid, token = uuid.uuid4().hex, api.issue_job_token()
    now = time.time()
    api._jobs[jid] = {"status": "done", "progress": "complete", "result": result,
                      "error": None, "created": now, "started": now,
                      "finished": now, "rows": 60, "key_hash": None,
                      "owner_hash": api._token_hash(token), "session_hash": None}
    out = client.get(f"/autopsy/jobs/{jid}", headers={"X-Job-Token": token})
    assert out.status_code == 200, out.text
    return out.json()["result"]


PLANTED = {"specimen": {"n_compounds": 60}, "ladder": [{"kind": "reported", "r2": 0.7}],
           "verdict": {"reported": 0.7, "survives_scaffold": 0.6, "headline": "h"},
           "readout": [], "warnings": [], "meta": {"seed": 0}}


def test_paywall_off_serves_the_whole_diagnosis(free, csv_bytes, monkeypatch):
    """The default, and the deployed configuration: no flags at all. An audit
    comes back as the engine built it — every rung of the uploaded file, the
    scaffold composition, the readout, the reproducible metadata."""
    monkeypatch.delenv("AUTOPSY_VERDICT_ONLY", raising=False)
    res = _finish(free, csv_bytes)

    assert res["tier"] == "full"
    assert set(res) == {"tier", "specimen", "ladder", "verdict", "readout",
                        "warnings", "meta", "limitations"}

    # the rungs of *this* file, not a benchmark: 60 compounds went in
    assert res["specimen"]["n_compounds"] == 60
    assert {r["kind"] for r in res["ladder"]} >= {"reported", "lookup", "floor"}
    assert res["verdict"]["reported"] is not None
    assert res["verdict"]["headline"]
    assert res["readout"] and res["meta"]["seed"] == 0


def test_verdict_only_withholds_it_again(free, csv_bytes, monkeypatch):
    """One variable restores the lead-generating showcase, and it withholds by
    building the response rather than by trimming it."""
    monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", "true")
    res = _finish(free, csv_bytes)

    assert res["tier"] == "verdict"
    assert set(res) == {"tier", "inflation_pct", "inflation_basis",
                        "inflation_state", "warnings", "contact"}
    assert "ladder" not in res and "specimen" not in res


def test_the_two_switches_are_independent(free, monkeypatch):
    """The paywall being off must not imply withholding, and vice versa. This
    is the pairing that was wrong: one flag answering two questions."""
    monkeypatch.delenv("AUTOPSY_PAYWALL_ENABLED", raising=False)
    monkeypatch.delenv("AUTOPSY_VERDICT_ONLY", raising=False)
    assert _finished_job(free, PLANTED)["tier"] == "full"

    monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", "true")
    assert _finished_job(free, PLANTED)["tier"] == "verdict"

    monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", "false")
    assert _finished_job(free, PLANTED)["tier"] == "full"


@pytest.mark.parametrize("value,withheld", [("true", True), ("1", True), ("on", True),
                                            ("yes", True), ("false", False),
                                            ("0", False), ("", False), ("maybe", False)])
def test_the_switch_reads_the_obvious_spellings(monkeypatch, value, withheld):
    monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", value)
    from autopsy import tiers
    assert tiers.verdict_only() is withheld


def test_the_switch_is_read_at_request_time_not_at_import(free, monkeypatch):
    """Flipping it must take effect without restarting the process, or a
    deployment's setting can be shadowed by whatever was set at load."""
    monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", "true")
    assert _finished_job(free, PLANTED)["tier"] == "verdict"
    monkeypatch.delenv("AUTOPSY_VERDICT_ONLY")
    assert _finished_job(free, PLANTED)["tier"] == "full"


def test_a_subscriber_reads_everything_even_when_withholding_is_on(paid, monkeypatch):
    """Withholding is aimed at visitors, not at the people who paid."""
    monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", "true")
    res = _finished_job(paid, PLANTED)
    assert res["tier"] == "full" and "ladder" in res


# ── the mode, published ───────────────────────────────────────────────
# A deployment that withholds its findings looks exactly like a broken one
# until you can ask which it is. These pin the answer, because the question
# has now cost three debugging sessions and the field is the whole fix.

def test_health_says_what_an_audit_will_return_by_default(free):
    mode = free.get("/health").json()["mode"]
    assert mode == {"paywall_enabled": False, "verdict_only": False,
                    "audits_return": "full",
                    "note": "free to run; returns the full diagnosis"}


def test_health_says_so_when_the_deployment_withholds(free, monkeypatch):
    monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", "true")
    mode = free.get("/health").json()["mode"]
    assert mode["verdict_only"] is True and mode["audits_return"] == "verdict"
    assert "percentage only" in mode["note"]


def test_health_says_so_when_the_paywall_is_on(paid):
    mode = paid.get("/health").json()["mode"]
    assert mode["paywall_enabled"] is True
    assert "subscription required" in mode["note"]


def test_the_mode_tracks_the_environment_per_request(free, monkeypatch):
    """Read live, not bound at import — otherwise the report could disagree
    with the behaviour it is there to explain, which is worse than no report."""
    monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", "true")
    assert free.get("/health").json()["mode"]["audits_return"] == "verdict"
    monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", "false")
    assert free.get("/health").json()["mode"]["audits_return"] == "full"


def test_the_reported_mode_matches_what_an_audit_actually_returns(free, csv_bytes,
                                                                  monkeypatch):
    """The report is only worth having if it cannot drift from the behaviour.
    Both positions checked against a real audit, not against the flag."""
    for value, expected in (("false", "full"), ("true", "verdict")):
        monkeypatch.setenv("AUTOPSY_VERDICT_ONLY", value)
        assert free.get("/health").json()["mode"]["audits_return"] == expected
        assert _finish(free, csv_bytes)["tier"] == expected


# ── the cost knob ─────────────────────────────────────────────────────

def test_repeats_is_a_deployment_knob_and_the_result_says_which_way(free, csv_bytes,
                                                                    monkeypatch):
    """The bands cost what they measure, so the trade is available and named.
    Either way the audit must be honest about which it did: one partition
    reports no spread *and* declares that it has none, rather than quietly
    dropping the bands and leaving the numbers looking as firm as before."""
    monkeypatch.setenv("AUTOPSY_REPEATS", "1")
    lone = _finish(free, csv_bytes)
    assert all(r.get("r2_sd") is None for r in lone["ladder"])
    assert "single_partition" in {l["code"] for l in lone["limitations"]}
    assert lone["meta"]["repeats"] == 1

    monkeypatch.setenv("AUTOPSY_REPEATS", "2")
    banded = _finish(free, csv_bytes)
    assert banded["meta"]["repeats"] == 2
    assert any(r.get("r2_sd") is not None for r in banded["ladder"])
    assert "single_partition" not in {l["code"] for l in banded["limitations"]}


def test_a_nonsense_repeats_value_falls_back_instead_of_crashing(free, monkeypatch):
    monkeypatch.setenv("AUTOPSY_REPEATS", "not-a-number")
    assert api._repeats() == api.engine_repeats
    monkeypatch.setenv("AUTOPSY_REPEATS", "0")
    assert api._repeats() == 1
