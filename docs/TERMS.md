# Terms of Service — MODEL AUTOPSY

**DRAFT. Not legal advice, and not ready to publish as written.**
It was written by a developer, not a lawyer. Before this goes in front of a
paying customer, have a lawyer qualified in your jurisdiction review it —
particularly the sections marked **[LEGAL REVIEW]**, which turn on facts only
you know: where ActarusLab is established, whether you sell to businesses
only, and how you handle EU VAT.

**Scope: this document covers the €199/month subscription, which is currently
switched off** (`AUTOPSY_PAYWALL_ENABLED=false`). The deployed service is a
free showcase: it runs an audit for anyone and returns the synthetic verdict
only. Nothing below describes that free tier — it has no fee, no quota and no
account — and terms for it still need to be written and reviewed. **[LEGAL
REVIEW: the free tier needs its own liability, data-retention and
acceptable-use clauses before the site goes live.]**

Last updated: _[date]_ · Provider: _[legal entity, address, VAT number]_

---

## 1. What the service does

MODEL AUTOPSY measures how much of a predictive model's reported accuracy
survives validation that a similarity lookup cannot fake. Given a dataset of
molecules and measured activities, it runs a fixed sequence of tests — random
split, nearest-neighbour lookup baseline, scaffold-disjoint split, temporal
split where dates are supplied, and a shuffled-target control — and reports
what each one scores.

**It reports measurements. It does not certify a model, approve a compound,
or predict an outcome.**

## 2. What the service does not do

The audit is a statistical measurement of a dataset you supply. It is not:

- a validation, certification or regulatory qualification of any model;
- advice — scientific, medical, clinical, regulatory, investment or otherwise;
- a prediction of how any model will behave on data it has not seen;
- a substitute for your own scientific judgement or your own experiments.

A favourable audit does not mean a model is fit for any purpose. An
unfavourable one does not mean it is worthless. **Every decision you take
after reading an audit is yours.**

## 3. Accuracy, and its limits

The audit is computed by the software as documented, and is reproducible: the
same dataset yields the same numbers. That is a statement about the
computation, not about your data.

Results depend entirely on the dataset you upload. Measurement error,
mislabelled activities, undisclosed duplicates, censored values or units mixed
between rows will all be reflected in the audit, which cannot detect them for
you. Rows the engine cannot read are reported and excluded; the audit then
describes what remains.

We do not warrant that the audit is fit for any particular purpose, and we
give no warranties beyond those that cannot lawfully be excluded.

## 4. Subscription, billing and cancellation

- €199 per month, excluding VAT where applicable. **[LEGAL REVIEW: VAT
  treatment, B2B reverse charge, and whether prices are shown inclusive.]**
- Each billing cycle includes **20 audits**. The counter resets when the
  subscription renews. Unused audits do not carry over.
- Past 20 audits the service returns a clear refusal until the next renewal.
  It does not charge overage.
- Payment is taken by Stripe. We never see or store your card details.
- Cancel at any time from the billing portal. Cancellation takes effect at the
  end of the paid cycle; access continues until then. We do not refund part
  cycles unless required by law. **[LEGAL REVIEW: EU right of withdrawal — if
  you sell to consumers rather than businesses only, a 14-day withdrawal right
  may apply and this clause is likely unenforceable as written.]**

## 5. Your data

- You keep all rights in the datasets you upload. We claim none.
- Datasets are used only to compute your audit.
- Audit results are held in the serving process and expire; they are not
  shared with other subscribers. **[LEGAL REVIEW: state the actual retention
  period once the deployment is fixed, and align this with your privacy
  notice and GDPR obligations, including any processor agreement.]**
- Do not upload data you are not entitled to process.

## 6. Acceptable use

Do not use the service to break the law, to infringe anyone's rights, or to
attack the service — including attempts to bypass the subscription check, the
usage limit, or another subscriber's data. We may suspend an account that
does, and will say why.

## 7. Availability

The service is provided as-is and as-available. We do not commit to an uptime
figure and may change or withdraw features. Where an outage prevents you from
using audits you have paid for, contact us. **[LEGAL REVIEW: decide whether to
offer a service credit, and say so here if you do.]**

## 8. Liability

To the fullest extent permitted by law, our total liability to you for any
claim arising from the service is limited to the subscription fees you paid in
the **twelve months** before the claim arose.

We are not liable for indirect or consequential loss, for lost profits, for
lost or wasted research effort, or for decisions taken on the basis of an
audit — including decisions to continue, abandon, license, acquire or publish
work relating to a model or a compound.

Nothing here excludes liability for death or personal injury caused by
negligence, for fraud, or for anything else that cannot lawfully be excluded.
**[LEGAL REVIEW: liability caps are read strictly and vary by jurisdiction;
some are unenforceable against consumers.]**

## 9. Changes to these terms

We may update these terms. Material changes will be notified to the email on
the subscription at least 30 days before they take effect; continuing to use
the service after that means you accept them. If you do not, cancel before
they take effect.

## 10. Governing law

**[LEGAL REVIEW: governing law and jurisdiction — for an Italian entity,
typically Italian law and a named court, but consumer rules may override the
choice.]**

## 11. Contact

_[support email]_ · _[legal entity and registered address]_
