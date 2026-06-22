"""
ContextCache: maintains a small rolling-window summary of recent outcomes,
used as h_t feature into the policy.

Week 1: tracks accept rate and mean latency over the last window_size queries.
"""
from collections import deque


class ContextCache:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.recent_accept: deque[int] = deque(maxlen=window_size)
        self.recent_latency: deque[float] = deque(maxlen=window_size)
        self.recent_revenue: deque[float] = deque(maxlen=window_size)
        self.total_count = 0

    def get_features(self) -> dict:
        return {
            "recent_accept_rate": (sum(self.recent_accept) / len(self.recent_accept)) if self.recent_accept else 0.5,
            "recent_mean_latency": (sum(self.recent_latency) / len(self.recent_latency)) if self.recent_latency else 0.0,
            "recent_mean_revenue": (sum(self.recent_revenue) / len(self.recent_revenue)) if self.recent_revenue else 0.0,
            "total_count": self.total_count,
        }

    def update(self, outcome):
        self.recent_accept.append(1 if outcome.A_t else 0)
        self.recent_latency.append(outcome.L_t)
        self.recent_revenue.append(outcome.R_t)
        self.total_count += 1
