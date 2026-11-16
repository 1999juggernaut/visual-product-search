"""The usage report: read usage.jsonl and add it up ("four readers, one row").

WHAT: totals per tenant and per model: requests, refusals (by code), tokens, cost in USD and INR.
      Then RECONCILES: the sum of the per-tenant totals must equal the grand total. Requests, tokens and cost
      are ADDITIVE, so they must agree to the last paisa; a mismatch means a row was double-counted or lost.
      Models with no price in shared/usage.py are listed, so a missing price is visible, not a silent 0.
Run:  python tasks.py report
"""

from shared.usage import USD_INR, price, read_usage


def add_one(counts: dict, key: str) -> None:
    """counts[key] += 1, starting from 0 the first time a key is seen."""
    if key not in counts:
        counts[key] = 0
    counts[key] += 1


def report(rows: list) -> dict:
    by_tenant = {}           # {tenant: {"requests", "refused", "tokens", "cost_usd"}}
    by_model = {}            # {model: number of calls}
    refusals = {}            # {refusal code: count}
    unpriced = []
    total_cost = 0.0
    total_tokens = 0

    for row in rows:
        tenant = row.get("tenant", "-")
        if tenant not in by_tenant:
            by_tenant[tenant] = {"requests": 0, "refused": 0, "tokens": 0, "cost_usd": 0.0}
        tokens = row.get("tokens_in", 0) + row.get("tokens_out", 0)
        cost = row.get("cost_usd", 0.0)

        by_tenant[tenant]["requests"] += 1
        by_tenant[tenant]["tokens"] += tokens
        by_tenant[tenant]["cost_usd"] += cost
        total_tokens += tokens
        total_cost += cost

        if row.get("refused"):
            by_tenant[tenant]["refused"] += 1
            add_one(refusals, row["refused"])

        model = row.get("model", "none")
        if model != "none":
            add_one(by_model, model)
            _cost, known = price(model, 0, 0)
            if not known and model not in unpriced:
                unpriced.append(model)

    # Reconcile: add the per-tenant totals back up; they must equal the grand totals.
    sum_requests = 0
    sum_tokens = 0
    sum_cost = 0.0
    for totals in by_tenant.values():
        sum_requests += totals["requests"]
        sum_tokens += totals["tokens"]
        sum_cost += totals["cost_usd"]
    reconciles = (sum_requests == len(rows) and sum_tokens == total_tokens
                  and round(sum_cost, 8) == round(total_cost, 8))

    return {"requests": len(rows), "cost_usd": round(total_cost, 6), "cost_inr": round(total_cost * USD_INR, 2),
            "tokens": total_tokens, "by_tenant": by_tenant, "by_model": by_model, "refusals": refusals,
            "unpriced_models": sorted(unpriced), "reconciles": reconciles}


def main() -> None:
    result = report(read_usage())
    print(f"requests {result['requests']}   tokens {result['tokens']:,}   "
          f"cost ${result['cost_usd']} (Rs {result['cost_inr']})")
    print("\nper tenant:")
    for tenant, totals in result["by_tenant"].items():
        print(f"  {tenant:<12} requests {totals['requests']:>5}  refused {totals['refused']:>4}  "
              f"tokens {totals['tokens']:>8,}  ${totals['cost_usd']:.6f}")
    print("\nre-rank calls per model:", result["by_model"] or "none")
    print("refusals by code:", result["refusals"] or "none")
    if result["unpriced_models"]:
        print("WARNING: no price for", result["unpriced_models"], "- add them to PRICES in shared/usage.py")
    if result["reconciles"]:
        print("\nreconciles (per-tenant sums == totals): YES")
    else:
        print("\nreconciles (per-tenant sums == totals): NO - investigate")


if __name__ == "__main__":
    main()
