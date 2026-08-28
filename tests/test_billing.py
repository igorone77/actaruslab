"""
The commercial layer: paywall, Stripe webhook, quota.

Stripe's own servers are not reachable from a test, but almost nothing here
needs them. The webhook is verified with a real signature — HMAC-SHA256 over
the payload with the endpoint secret — so these exercise the same code path
production does, including the rejection of an unsigned body. Only the two
calls that mint a Stripe URL are stubbed.
"""
import hashlib
import hmac
import importlib
import json
import time

import pytest
from fastapi import HTTPException

WEBHOOK_SECRET = "whsec_test_secret_for_local_verification"


@pytest.fixture()
def billing(tmp_path, monkeypatch):
    """A fresh database per test, so quota arithmetic cannot leak between."""
    monkeypatch.setenv("AUTOPSY_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("STRIPE_PRICE_ID", "price_x")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("AUTOPSY_PUBLIC_URL", "https://autopsy.example.org")
    import autopsy.billing as b
    importlib.reload(b)
    b.init_db()
    return b


def _signed(payload: dict, secret: str = WEBHOOK_SECRET):
    """Stripe's scheme: t=<ts>,v1=<hmac of "ts.body">."""
    body = json.dumps(payload).encode()
    ts = str(int(time.time()))
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return body, f"t={ts},v1={sig}"


# ── the paywall ───────────────────────────────────────────────────────

def test_no_key_is_402_with_the_way_to_fix_it(billing):
    with pytest.raises(HTTPException) as e:
        billing.require_subscription(authorization="", autopsy_key="")
    assert e.value.status_code == 402, "402 says unpaid; 401 would say unknown"
    assert "€199" in e.value.detail and "20 audits" in e.value.detail
    assert "demo stays free" in e.value.detail
    assert e.value.headers["X-Checkout-URL"].endswith("/billing/checkout")


def test_an_active_key_passes(billing):
    key = billing.create_subscriber("a@b.c", "cus_1", "sub_1", "active", time.time() + 86400)
    row = billing.require_subscription(authorization=f"Bearer {key}", autopsy_key="")
    assert row["email"] == "a@b.c"
    assert billing.require_subscription(authorization="", autopsy_key=key)["email"] == "a@b.c"


def test_a_cancelled_subscription_is_refused(billing):
    key = billing.create_subscriber("a@b.c", "cus_1", "sub_1", "canceled", 0)
    with pytest.raises(HTTPException) as e:
        billing.require_subscription(authorization=f"Bearer {key}", autopsy_key="")
    assert e.value.status_code == 402
    assert "canceled" in e.value.detail


def test_the_key_is_never_stored_in_the_clear(billing, tmp_path):
    key = billing.create_subscriber("a@b.c", "cus_1", "sub_1", "active", 0)
    blob = (tmp_path / "t.db").read_bytes()
    assert key.encode() not in blob, "the plaintext key reached the database"
    assert hashlib.sha256(key.encode()).hexdigest().encode() in blob


# ── quota ─────────────────────────────────────────────────────────────

def test_twenty_audits_then_blocked_until_renewal(billing):
    key = billing.create_subscriber("a@b.c", "cus_1", "sub_1", "active", 0)
    kh = billing._hash(key)

    for n in range(1, billing.QUOTA + 1):
        assert billing.reserve_audit(kh) == n

    with pytest.raises(billing.QuotaExhausted):
        billing.reserve_audit(kh)


def test_renewal_returns_the_allowance(billing):
    key = billing.create_subscriber("a@b.c", "cus_1", "sub_1", "active", 0)
    kh = billing._hash(key)
    for _ in range(billing.QUOTA):
        billing.reserve_audit(kh)

    billing.update_subscription("sub_1", "active", time.time() + 2592000, reset_usage=True)
    assert billing.reserve_audit(kh) == 1, "a paid invoice restores the cycle"


def test_a_failure_on_our_side_is_refunded(billing):
    key = billing.create_subscriber("a@b.c", "cus_1", "sub_1", "active", 0)
    kh = billing._hash(key)
    billing.reserve_audit(kh)
    billing.refund_audit(kh)
    assert billing.by_key(key)["used"] == 0


def test_refund_cannot_go_below_zero(billing):
    key = billing.create_subscriber("a@b.c", "cus_1", "sub_1", "active", 0)
    kh = billing._hash(key)
    billing.refund_audit(kh)
    assert billing.by_key(key)["used"] == 0


# ── webhook ───────────────────────────────────────────────────────────

def _post(billing, body, sig):
    """Drive the endpoint the way Starlette would."""
    import asyncio

    class Req:
        headers = {"stripe-signature": sig}
        async def body(self):
            return body

    return asyncio.get_event_loop().run_until_complete(billing.webhook(Req()))


def test_an_unsigned_webhook_is_refused(billing):
    body, _ = _signed({"type": "checkout.session.completed", "data": {"object": {}}})
    with pytest.raises(HTTPException) as e:
        _post(billing, body, "t=1,v1=deadbeef")
    assert e.value.status_code == 400
    assert "signature" in e.value.detail


def test_a_forged_secret_is_refused(billing):
    body, sig = _signed({"type": "invoice.paid", "data": {"object": {}}},
                        secret="whsec_attacker")
    with pytest.raises(HTTPException):
        _post(billing, body, sig)


def test_checkout_completed_creates_a_subscriber_and_issues_one_key(billing):
    event = {"type": "checkout.session.completed", "data": {"object": {
        "subscription": "sub_new", "customer": "cus_new",
        "customer_details": {"email": "buyer@lab.org"}, "expires_at": 0}}}
    out = _post(billing, *_signed(event))

    assert out["received"] is True
    key = out["api_key"]
    assert billing.by_key(key)["email"] == "buyer@lab.org"

    # Stripe retries webhooks; a retry must not mint a second key.
    again = _post(billing, *_signed(event))
    assert "api_key" not in again


def test_invoice_paid_extends_the_period_and_resets_usage(billing):
    key = billing.create_subscriber("a@b.c", "cus_1", "sub_1", "active", 0)
    for _ in range(5):
        billing.reserve_audit(billing._hash(key))

    end = time.time() + 2592000
    event = {"type": "invoice.paid", "data": {"object": {
        "subscription": "sub_1",
        "lines": {"data": [{"period": {"end": end}}]}}}}
    _post(billing, *_signed(event))

    row = billing.by_key(key)
    assert row["used"] == 0 and row["status"] == "active"
    assert row["period_end"] == pytest.approx(end)


def test_cancellation_revokes_access_without_wiping_the_record(billing):
    key = billing.create_subscriber("a@b.c", "cus_1", "sub_1", "active", 0)
    event = {"type": "customer.subscription.deleted", "data": {"object": {
        "id": "sub_1", "status": "canceled", "current_period_end": 0}}}
    _post(billing, *_signed(event))

    assert billing.by_key(key)["status"] == "canceled"
    with pytest.raises(HTTPException):
        billing.require_subscription(authorization=f"Bearer {key}", autopsy_key="")


# ── the plan, and what stays free ─────────────────────────────────────

def test_plan_is_public_and_states_the_offer(billing):
    p = billing.plan()
    assert p["price_eur_month"] == 199 and p["audits_per_cycle"] == 20
    assert p["currency"] == "EUR" and p["billing_configured"] is True


def test_billing_endpoints_say_so_when_stripe_is_not_configured(billing, monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY")
    with pytest.raises(HTTPException) as e:
        billing.checkout()
    assert e.value.status_code == 503
    assert "STRIPE_SECRET_KEY" in e.value.detail
