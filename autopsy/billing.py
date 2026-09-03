"""
MODEL AUTOPSY — the commercial layer.

€199/month, 20 audits per billing cycle, blocked until renewal past that.
Everything here is mechanism: the Stripe account, the price and the deploy
are configured outside, through the environment.

    STRIPE_SECRET_KEY        sk_live_… / sk_test_…
    STRIPE_PRICE_ID          price_… for the €199/month recurring price
    STRIPE_WEBHOOK_SECRET    whsec_… — signature key for /billing/webhook
    AUTOPSY_PUBLIC_URL       where this service is reachable, for Stripe's
                             redirects. Defaults to http://127.0.0.1:8000
    AUTOPSY_DB               SQLite file. Defaults to ./autopsy.db
    AUTOPSY_QUOTA            audits per cycle. Defaults to 20
    AUTOPSY_PAYWALL_ENABLED  the master switch. Defaults to false — see
                             `paywall_enabled()`. Everything in this file
                             works either way; the flag only decides whether
                             the engine is behind it.

Why SQLite and not the in-process dict the job queue uses: a restart may
forget a running audit, but it may never forget who paid. One file, one
container, no infrastructure. A horizontally scaled deployment needs Postgres
behind the same functions — the schema is four columns and a counter.

Who is who
----------
Checkout issues an API key once, shown once, stored only as a SHA-256 hash.
Callers present it as `Authorization: Bearer <key>` or, for the browser, the
`autopsy_key` cookie set on return from Stripe. There is no password to
reset and no session to hijack; losing the key means rotating it from the
billing page.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Cookie, Header, HTTPException, Request, Response

PRICE_EUR = 199
QUOTA = int(os.getenv("AUTOPSY_QUOTA", "20"))
DB_PATH = os.getenv("AUTOPSY_DB", "autopsy.db")
PUBLIC_URL = os.getenv("AUTOPSY_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/")

# Statuses Stripe reports that still entitle the holder to run audits.
LIVE_STATUSES = {"active", "trialing"}


_TRUE = {"1", "true", "yes", "on"}


def paywall_enabled() -> bool:
    """Is the engine behind the paywall? One flag decides, and it is off.

    Autopsy is a free showcase: an audit runs for anyone, and what it returns
    is the synthetic verdict — see autopsy/tiers.py. The commercial layer in
    this file is intact and inert, not deleted, because the business model is
    a switch and not a rewrite. Set

        AUTOPSY_PAYWALL_ENABLED=true

    and every computing endpoint is behind a subscription again, exactly as
    it was: same 402, same quota, same Checkout link, same webhook.

    Deliberately *not* the presence of STRIPE_SECRET_KEY. That coupling made
    the paywall a side effect of configuration — set a key to test a webhook
    and the engine silently locked. Turning a deployment paid is now one
    explicit decision, written down in one variable.
    """
    return os.getenv("AUTOPSY_PAYWALL_ENABLED", "false").strip().lower() in _TRUE


def selling() -> bool:
    """Can this deployment actually take a payment?

    A paywall in front of a Stripe account that does not exist is a locked
    door with no handle: `paywall_enabled()` decides whether to charge,
    this decides whether charging is possible. Both must hold, and when the
    flag is on without the key `require_subscription` says so rather than
    quietly serving free audits to a deployment that believes it is paid.
    """
    return bool(os.getenv("STRIPE_SECRET_KEY"))

router = APIRouter(prefix="/billing", tags=["billing"])


# ── store ─────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS subscribers (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    email                  TEXT    NOT NULL,
    key_hash               TEXT    NOT NULL UNIQUE,
    stripe_customer_id     TEXT,
    stripe_subscription_id TEXT UNIQUE,
    status                 TEXT    NOT NULL DEFAULT 'incomplete',
    period_end             REAL    NOT NULL DEFAULT 0,
    used                   INTEGER NOT NULL DEFAULT 0,
    created_at             REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sub_customer ON subscribers (stripe_customer_id);
"""


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _db() as conn:
        conn.executescript(SCHEMA)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def issue_key() -> str:
    """Shown once, never stored in the clear."""
    return "ma_" + secrets.token_urlsafe(32)


# ── subscriber lifecycle ──────────────────────────────────────────────
def create_subscriber(email: str, customer_id: str, subscription_id: str,
                      status: str, period_end: float) -> str:
    """Returns the plaintext API key — the only moment it exists."""
    key = issue_key()
    with _db() as conn:
        conn.execute(
            "INSERT INTO subscribers (email, key_hash, stripe_customer_id, "
            "stripe_subscription_id, status, period_end, used, created_at) "
            "VALUES (?,?,?,?,?,?,0,?)",
            (email, _hash(key), customer_id, subscription_id, status,
             period_end, time.time()))
    return key


def by_key(key: str) -> Optional[sqlite3.Row]:
    if not key:
        return None
    with _db() as conn:
        return conn.execute("SELECT * FROM subscribers WHERE key_hash = ?",
                            (_hash(key),)).fetchone()


def by_subscription(subscription_id: str) -> Optional[sqlite3.Row]:
    with _db() as conn:
        return conn.execute(
            "SELECT * FROM subscribers WHERE stripe_subscription_id = ?",
            (subscription_id,)).fetchone()


def update_subscription(subscription_id: str, status: str,
                        period_end: float, reset_usage: bool) -> None:
    """Stripe is the authority on status and on when the cycle rolls over.
    A paid invoice both extends the period and returns the 20 audits."""
    with _db() as conn:
        if reset_usage:
            conn.execute(
                "UPDATE subscribers SET status=?, period_end=?, used=0 "
                "WHERE stripe_subscription_id=?",
                (status, period_end, subscription_id))
        else:
            conn.execute(
                "UPDATE subscribers SET status=?, period_end=? "
                "WHERE stripe_subscription_id=?",
                (status, period_end, subscription_id))


# ── quota ─────────────────────────────────────────────────────────────
class QuotaExhausted(Exception):
    def __init__(self, used: int, period_end: float):
        self.used, self.period_end = used, period_end


def reserve_audit(key_hash: str) -> int:
    """Take one audit from the cycle, atomically. Returns the new count.

    The UPDATE ... WHERE used < QUOTA is the lock: two requests racing for the
    last audit cannot both win, because SQLite serialises the write and the
    loser matches no row.
    """
    with _db() as conn:
        cur = conn.execute(
            "UPDATE subscribers SET used = used + 1 "
            "WHERE key_hash = ? AND used < ?", (key_hash, QUOTA))
        if cur.rowcount == 0:
            row = conn.execute(
                "SELECT used, period_end FROM subscribers WHERE key_hash = ?",
                (key_hash,)).fetchone()
            raise QuotaExhausted(row["used"] if row else QUOTA,
                                 row["period_end"] if row else 0)
        return conn.execute("SELECT used FROM subscribers WHERE key_hash = ?",
                            (key_hash,)).fetchone()["used"]


def refund_audit(key_hash: str) -> None:
    """A failure on our side must not cost a subscriber one of their 20."""
    with _db() as conn:
        conn.execute("UPDATE subscribers SET used = MAX(used - 1, 0) "
                     "WHERE key_hash = ?", (key_hash,))


# ── stripe ────────────────────────────────────────────────────────────
def _stripe():
    """Imported late and only when needed, so the service still starts — and
    serves the free demo — on a machine with no Stripe credentials at all."""
    key = os.getenv("STRIPE_SECRET_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="billing is not configured on this deployment. Set "
                   "STRIPE_SECRET_KEY, STRIPE_PRICE_ID and "
                   "STRIPE_WEBHOOK_SECRET to enable subscriptions.")
    try:
        import stripe
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="the stripe package is not installed on this deployment.")
    stripe.api_key = key
    return stripe


def checkout_url() -> str:
    """A subscribe link, for the 402 body and the paywall page."""
    return f"{PUBLIC_URL}/billing/checkout"


# ── endpoints ─────────────────────────────────────────────────────────
@router.get("/plan")
def plan():
    """Public: what the subscription costs and includes."""
    return {"price_eur_month": PRICE_EUR, "audits_per_cycle": QUOTA,
            "currency": "EUR", "checkout": checkout_url(),
            "billing_configured": bool(os.getenv("STRIPE_SECRET_KEY"))}


@router.get("/checkout")
@router.post("/checkout")
def checkout():
    """Sends the buyer to Stripe. Stripe collects the card and the VAT
    details; nothing sensitive is handled here."""
    stripe = _stripe()
    price = os.getenv("STRIPE_PRICE_ID")
    if not price:
        raise HTTPException(status_code=503, detail="STRIPE_PRICE_ID is not set.")
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price, "quantity": 1}],
        success_url=f"{PUBLIC_URL}/billing/return?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{PUBLIC_URL}/?checkout=cancelled",
        allow_promotion_codes=True,
    )
    return {"checkout_url": session.url}


@router.get("/return")
def checkout_return(session_id: str, response: Response):
    """Where Stripe sends the buyer back. The webhook is what actually grants
    access — this only reads the result and hands over the key."""
    stripe = _stripe()
    session = stripe.checkout.Session.retrieve(session_id)
    sub_id = session.get("subscription")
    if not sub_id:
        raise HTTPException(
            status_code=402,
            detail="that checkout did not complete. Nothing has been charged.")

    row = by_subscription(sub_id)
    if row is None:
        # The webhook has not landed yet. Stripe retries it, so the honest
        # answer is "not yet", not a second key for the same subscription.
        raise HTTPException(
            status_code=202,
            detail="payment received — your access key is being issued. "
                   "Reload this page in a few seconds.")
    return {"status": row["status"], "audits_per_cycle": QUOTA,
            "used": row["used"],
            "note": "your key was shown once when it was issued. Rotate it "
                    "from the billing portal if you have lost it."}


@router.post("/webhook")
async def webhook(request: Request):
    """Stripe's word on who is a subscriber. Unsigned bodies are refused:
    without verification this endpoint would hand out free subscriptions to
    anyone who can POST JSON."""
    stripe = _stripe()
    secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=503,
                            detail="STRIPE_WEBHOOK_SECRET is not set.")
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(
            payload, request.headers.get("stripe-signature", ""), secret)
    except Exception:
        raise HTTPException(status_code=400,
                            detail="the webhook signature did not verify.")

    kind = event["type"]
    obj = event["data"]["object"]

    if kind == "checkout.session.completed":
        sub_id = obj.get("subscription")
        if sub_id and by_subscription(sub_id) is None:
            key = create_subscriber(
                email=(obj.get("customer_details") or {}).get("email", ""),
                customer_id=obj.get("customer", ""),
                subscription_id=sub_id,
                status="active",
                period_end=float(obj.get("expires_at") or 0),
            )
            # Delivered to the buyer by whatever channel the deployment
            # chooses; it is never written to a log.
            return {"received": True, "api_key": key}

    elif kind in ("invoice.paid", "invoice.payment_succeeded"):
        sub_id = obj.get("subscription")
        if sub_id:
            period_end = float(((obj.get("lines") or {}).get("data") or [{}])[0]
                               .get("period", {}).get("end") or 0)
            update_subscription(sub_id, "active", period_end, reset_usage=True)

    elif kind in ("customer.subscription.updated", "customer.subscription.deleted"):
        update_subscription(obj["id"], obj.get("status", "canceled"),
                            float(obj.get("current_period_end") or 0),
                            reset_usage=False)

    return {"received": True}


@router.get("/portal")
def portal(authorization: str = Header(default=""),
           autopsy_key: str = Cookie(default="")):
    """Cancel, change card, download invoices — all of it is Stripe's, not
    ours, so we only mint the link."""
    row = by_key(_bearer(authorization) or autopsy_key)
    if row is None:
        raise HTTPException(status_code=401, detail=_UNKNOWN_KEY)
    stripe = _stripe()
    session = stripe.billing_portal.Session.create(
        customer=row["stripe_customer_id"], return_url=f"{PUBLIC_URL}/")
    return {"portal_url": session.url}


@router.get("/me")
def me(authorization: str = Header(default=""), autopsy_key: str = Cookie(default="")):
    """What the holder of this key has left in the cycle."""
    row = by_key(_bearer(authorization) or autopsy_key)
    if row is None:
        raise HTTPException(status_code=401, detail=_UNKNOWN_KEY)
    return {"email": row["email"], "status": row["status"],
            "audits_per_cycle": QUOTA, "used": row["used"],
            "remaining": max(QUOTA - row["used"], 0),
            "renews_at": row["period_end"]}


# ── the paywall ───────────────────────────────────────────────────────
_UNKNOWN_KEY = ("no active subscription for this key. Subscribe at "
                f"{checkout_url()} — €{PRICE_EUR}/month, {QUOTA} audits per "
                "cycle. The BACE-1 demo stays free.")


def _bearer(authorization: str) -> str:
    parts = (authorization or "").split()
    return parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else ""


def require_subscription(authorization: str = Header(default=""),
                         autopsy_key: str = Cookie(default="")):
    """FastAPI dependency guarding every paid endpoint.

    Returns None while the paywall is switched off — the default, and the free
    showcase — and the caller then meters nothing and gets the free tier.
    Otherwise a subscriber row, or a refusal.

    This is the payment gate and nothing else. Whether a caller may read a
    particular audit result is a separate question with a separate answer:
    api.py::_authorize_job, which stays on in every configuration. A free
    audit still belongs to whoever launched it.

    402 rather than 401 throughout: the caller is not unknown, they are
    unpaid, and the body carries the link that fixes it.
    """
    if not paywall_enabled():
        return None
    if not selling():
        # Fail loudly rather than serve free audits from a deployment whose
        # operator believes it is charging for them.
        raise HTTPException(
            status_code=503,
            detail="this deployment has AUTOPSY_PAYWALL_ENABLED set but no "
                   "STRIPE_SECRET_KEY, so it can refuse audits but not sell "
                   "them. Configure Stripe, or unset the flag to run free.")

    key = _bearer(authorization) or autopsy_key
    row = by_key(key)
    if row is None:
        raise HTTPException(status_code=402, detail=_UNKNOWN_KEY,
                            headers={"X-Checkout-URL": checkout_url()})
    if row["status"] not in LIVE_STATUSES:
        raise HTTPException(
            status_code=402,
            detail=f"this subscription is {row['status']}. Reactivate it from "
                   f"the billing portal to run audits again.",
            headers={"X-Checkout-URL": checkout_url()})
    return row
