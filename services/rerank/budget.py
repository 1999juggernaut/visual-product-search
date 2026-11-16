"""The budget breaker: "degrade, never stop".

WHAT: ``Breaker.mode()`` says how freely we may call a paid/limited re-ranker right now:
    normal     anything allowed; the router may pick the strong model for close calls
    strict     spend has reached strict_at_pct (80%) of the budget: only the default re-ranker, never the strong one
    exhausted  daily request limit or 100% of budget used: no re-rank calls; answers come from search order
  ``Breaker.take()`` asks for ONE request slot (per-minute and per-day clocks). False = don't call; the caller
  falls back to search order and logs it. Nothing ever returns an error to the customer because of budget.
WHY clocks AND money: OpenRouter's free models cost $0 but allow 20 requests/minute and 50/day; paid models
  cost money. One breaker handles both.
REPLAY WITHOUT SPENDING: VPI_SPEND_PCT=85 pretends 85% is spent, so you can watch strict mode switch on
  without paying for it.
WHERE THE NUMBERS COME FROM: today's request count and this month's spend are read from usage.jsonl at
  start (the usage row is the single truth), then counted in memory.
"""

import datetime
import threading
import time


class Breaker:
    def __init__(self, settings, usage_rows=None, clock=time.time):
        """``clock`` is the function that tells the time. Tests pass a fake clock so they don't have to wait."""
        self.settings = settings
        self.clock = clock
        self.recent_calls = []               # times (in seconds) of the calls made in the last 60 seconds
        self.lock = threading.Lock()         # two requests at once must not both take the last slot

        today = datetime.date.today().isoformat()        # e.g. "2026-10-19"
        this_month = today[:7]                           # e.g. "2026-10"
        self.day_count = 0
        self.month_spend = 0.0
        for row in usage_rows or []:
            if row.get("model", "none") == "none" or row.get("cached"):
                continue                                 # only real (not cached) model calls count
            timestamp = row.get("ts", "")
            if timestamp.startswith(today):
                self.day_count += 1
            if timestamp.startswith(this_month):
                self.month_spend += float(row.get("cost_usd", 0))

    def spend_pct(self) -> float:
        """How much of this month's budget is used, in percent."""
        if self.settings.spend_pct:                      # the replay setting wins
            return float(self.settings.spend_pct)
        if self.settings.budget_usd <= 0:
            return 0.0
        return 100.0 * self.month_spend / self.settings.budget_usd

    def mode(self) -> str:
        if self.day_count >= self.settings.daily_request_limit or self.spend_pct() >= 100:
            return "exhausted"
        if self.spend_pct() >= self.settings.strict_at_pct:
            return "strict"
        return "normal"

    def take(self) -> bool:
        """Reserve one call. Checked BEFORE spending (check before you spend). False = don't call."""
        with self.lock:
            now = self.clock()
            # Forget calls older than 60 seconds.
            still_recent = []
            for call_time in self.recent_calls:
                if now - call_time <= 60:
                    still_recent.append(call_time)
            self.recent_calls = still_recent

            if self.mode() == "exhausted":
                return False
            if len(self.recent_calls) >= self.settings.requests_per_minute:
                return False
            self.recent_calls.append(now)
            self.day_count += 1
            return True

    def charge(self, cost_usd: float) -> None:
        with self.lock:
            self.month_spend += cost_usd
