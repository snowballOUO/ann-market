"""
Buyer: Week 1 placeholder.

A real buyer model lives in src/data/buyer_simulator.py (Week 2 work).
This stub gives a deterministic-ish accept rule so the pipeline can run end-to-end.

Decision rule:
  accept iff (price <= budget) AND (latency <= sla) AND (price_value_ok)
where price_value_ok is a noisy check that price is reasonable given budget.
"""
import random
from src.system.types import Query


class StubBuyer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def respond(self, query: Query, results: list, price: float, latency: float):
        """Returns (A_t, S_t)."""
        # Hard checks
        if price > query.budget_t:
            return False, 0.0
        if latency > query.sla_t:
            return False, 0.0

        # Noisy price acceptance
        price_ratio = price / max(query.budget_t, 1e-9)
        # Lower price → higher accept prob; near budget → ~50/50
        accept_prob = 1.0 - 0.5 * price_ratio
        accept = self.rng.random() < accept_prob

        # Satisfaction: full if accepted and fast, partial otherwise
        if accept:
            S_t = 1.0 - 0.5 * (latency / max(query.sla_t, 1e-9))
        else:
            S_t = 0.0

        return accept, max(S_t, 0.0)
