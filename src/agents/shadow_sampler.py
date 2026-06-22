"""
Agent 4: ShadowSampler

For a small random fraction of queries, asynchronously run exact KNN to
get ground-truth Q_t (recall@k). Used as unbiased reward signal for
off-policy learning. Runs OFF the hot path.

This is one of the core systems contributions of the paper. It turns an
otherwise-unobservable quantity (true recall) into a controlled, samplable
signal at known cost.
"""
import random
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Callable
import numpy as np
from src.system.types import Query


class ShadowSampler:
    def __init__(
        self,
        base_vectors: np.ndarray,
        sample_rate: float = 0.02,
        max_workers: int = 2,
        on_recall_computed: Optional[Callable[[str, float], None]] = None,
        seed: int = 42,
    ):
        """
        Args:
            base_vectors: the FULL base set used for exact search, shape (N, dim)
            sample_rate: fraction of queries to shadow-sample
            max_workers: thread pool size (these threads contend with serving)
            on_recall_computed: callback invoked with (query_id, recall) when ready
        """
        self.base_vectors = base_vectors
        self.sample_rate = sample_rate
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.on_recall_computed = on_recall_computed
        self.rng = random.Random(seed)
        self.pending: dict[str, Future] = {}

    def maybe_sample(self, query: Query, approximate_results: list) -> bool:
        """Fire-and-forget. Returns True if this query was sampled."""
        if self.rng.random() >= self.sample_rate:
            return False

        approx_ids = [r[0] for r in approximate_results]
        fut = self.executor.submit(
            self._compute_recall, query.id, query.v_t, query.k_t, approx_ids
        )
        self.pending[query.id] = fut
        return True

    def _compute_recall(self, query_id: str, v: np.ndarray, k: int, approx_ids: list) -> float:
        """Exact KNN by brute force. Slow but correct."""
        # Vectorized squared L2 distance to every base vector
        diffs = self.base_vectors - v
        dists_sq = np.einsum("ij,ij->i", diffs, diffs)
        exact_ids = np.argpartition(dists_sq, k)[:k].tolist()

        approx_set = set(approx_ids)
        exact_set = set(exact_ids)
        if len(exact_set) == 0:
            recall = 0.0
        else:
            recall = len(approx_set & exact_set) / len(exact_set)

        if self.on_recall_computed is not None:
            self.on_recall_computed(query_id, recall)
        return recall

    def drain(self, timeout: float = 60.0):
        """Block until all pending shadow searches complete. Use at end of run."""
        for qid, fut in list(self.pending.items()):
            try:
                fut.result(timeout=timeout)
            except Exception as e:
                print(f"Shadow recall for {qid} failed: {e}")
        self.pending.clear()

    def shutdown(self):
        self.executor.shutdown(wait=True)
