# Setting up billing — the part only you can do

The mechanism is built and tested. What is left needs your Stripe account and
your deployment, and none of it touches code.

---

## 1. In Stripe (about 10 minutes)

**Create the product and price.** Stripe Dashboard → *Product catalogue* →
*Add product*.

- Name: `MODEL AUTOPSY — subscription`
- Description: `20 forensic audits per month`
- Pricing model: **Recurring**, `199.00 EUR`, billing period **Monthly**
- Save, then copy the **price ID** — it looks like `price_1Ab2Cd…`.
  This is `STRIPE_PRICE_ID`. It is not the product ID; the price ID is the one
  under the price, not the product header.

**Copy the secret key.** *Developers → API keys → Secret key* (`sk_live_…`).
Use the **test** key (`sk_test_…`) until you have run a test purchase.
This is `STRIPE_SECRET_KEY`. It is a password — it goes in the deployment's
secret store, never in the repo.

**Create the webhook.** *Developers → Webhooks → Add endpoint*.

- Endpoint URL: `https://<your-domain>/billing/webhook`
- Events to send — exactly these four:
  - `checkout.session.completed` — issues the access key
  - `invoice.paid` — renews the cycle and returns the 20 audits
  - `customer.subscription.updated` — status changes
  - `customer.subscription.deleted` — cancellation
- Save, then reveal the **signing secret** (`whsec_…`).
  This is `STRIPE_WEBHOOK_SECRET`.

> The signing secret is what makes the webhook trustworthy. Without it anyone
> who can POST JSON at your server could mint themselves a free subscription,
> so the endpoint refuses every unsigned body. If you rotate the secret in
> Stripe, update the variable and redeploy, or subscriptions stop being
> recorded.

**Turn on the customer portal.** *Settings → Billing → Customer portal* →
enable, and allow customers to cancel and to update their payment method.
`/billing/portal` mints links into it. Without this, cancellation has to go
through you by hand.

**Tax.** If you are selling into the EU, enable *Stripe Tax* or configure VAT
rates on the price before you take a real payment. Getting this wrong after
the fact means reissuing invoices.

---

## 2. In the deployment

Seven environment variables. The three Stripe ones are secrets.

**Start with the switch.** Autopsy ships as a free showcase: the paywall is
off, the engine runs for anyone, and every step below is inert until the flag
is set. Nothing in this document has any effect without it.

| variable | value | notes |
|---|---|---|
| `AUTOPSY_PAYWALL_ENABLED` | `true` | **required.** Default `false` — free showcase. Set it and everything below switches on. |
| `STRIPE_SECRET_KEY` | `sk_live_…` | secret |
| `STRIPE_PRICE_ID` | `price_…` | the €199/month recurring price |
| `STRIPE_WEBHOOK_SECRET` | `whsec_…` | secret |
| `AUTOPSY_PUBLIC_URL` | `https://autopsy.actaruslab.org` | no trailing slash — Stripe redirects here |
| `AUTOPSY_DB` | `/data/autopsy.db` | **must be on a persistent volume** |
| `AUTOPSY_QUOTA` | `20` | optional; this is the default |

> The flag and the key must both be set. `AUTOPSY_PAYWALL_ENABLED=true`
> without `STRIPE_SECRET_KEY` is refused with a 503 on every audit rather than
> quietly serving them free — the worst failure here would be a silent one,
> where the deployment believes it is charging and is not.

> `AUTOPSY_DB` is the one that will bite you. It holds who has paid. On a
> platform with an ephemeral filesystem — a plain container, most PaaS default
> configurations — a redeploy wipes it and every subscriber loses access while
> still being charged. Mount a volume, or move the store to Postgres before
> going live.

Then deploy as before: `docker build -t model-autopsy . && docker run -p 8000:8000 …`
with those variables set. The container already includes the `stripe` package.

---

## 3. Verify, in this order

0. **The switch is on** — with `AUTOPSY_PAYWALL_ENABLED` unset, an upload
   runs and returns a result (`tier: "full"` by default, or `tier: "verdict"`
   where `AUTOPSY_VERDICT_ONLY` is set). Either way the audit ran without
   payment, so the paywall is off and steps 2–5 cannot pass. Note that
   `AUTOPSY_VERDICT_ONLY` is a separate switch and does not paywall anything:
   it decides what a free audit says, not whether one may run.
1. **Free things still free** — open the site logged out. The BACE-1 demo
   renders. `GET /health` and `GET /autopsy/demo` return 200.
2. **Paywall bites** — try an upload without subscribing. You get **402** and
   a link to checkout, not a crash and not a free audit.
3. **Test purchase** — with the *test* keys, subscribe using Stripe's test
   card `4242 4242 4242 4242`, any future expiry, any CVC.
4. **Webhook landed** — Stripe Dashboard → Webhooks → your endpoint shows
   `checkout.session.completed` with a **200**. If it shows 400, the signing
   secret does not match. If it shows nothing, the URL is wrong or not
   reachable from the internet.
5. **Access works** — the audit runs, and `GET /billing/me` shows
   `used: 1, remaining: 19`.
6. **Cancellation works** — cancel from the portal, then try another audit.
   You get 402 naming the status.
7. **Switch to live keys** and repeat steps 3–5 with a real card, once.

---

## 4. What you still have to decide

Three things I could not decide for you.

**Key delivery.** When a subscription starts, the webhook issues an API key,
shown once and stored only as a hash. Right now it is returned in the webhook
response, which Stripe sees but the buyer does not. **Before launch, pick how
the buyer receives it** — an email from your side on
`checkout.session.completed`, or a one-time page after the Stripe redirect.
Until then, subscribers pay and get nothing they can use.

**The terms.** `docs/TERMS.md` is a draft with the clauses that need a lawyer
marked `[LEGAL REVIEW]` — VAT, the EU right of withdrawal, data retention,
the liability cap, governing law. Do not publish it as it stands.

**Who may reach the service.** Anyone with the URL can attempt checkout. That
is what self-service means, but it also means the audit endpoints face the
open internet on a €199 plan. The rate limits already in place
(`AUTOPSY_WORKERS`, `AUTOPSY_QUEUE_DEPTH`, upload and row caps) bound the
damage; whether you also want an allowlist or a manual approval step is a
commercial call.

---

## 5. How the pieces fit

```
  visitor ──► GET /autopsy/demo        free, precomputed BACE-1 audit
          └─► POST /autopsy/jobs  ──►  402 + X-Checkout-URL
                                       │
  buyer   ──► GET /billing/checkout ──►│ Stripe Checkout (card, VAT)
                                       ▼
              Stripe ──► POST /billing/webhook   signature verified
                              │
                              ├─ checkout.session.completed → subscriber + API key
                              ├─ invoice.paid               → period extended, 20 audits back
                              └─ subscription.deleted       → access revoked
                                       │
  subscriber ─► POST /autopsy/jobs ───►│ one audit spent, refunded if we fail
               GET /billing/me         │ used / remaining / renews_at
               GET /billing/portal ───►│ Stripe portal: cancel, card, invoices
```

The key is presented as `Authorization: Bearer ma_…` or the `autopsy_key`
cookie. Audit results are readable only by the key that submitted them.
