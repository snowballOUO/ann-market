"""
Agent 3: ExecutionAgent

Thin wrapper over FAISS. Its job is to:
  - Apply the chosen search parameters z_t
  - Measure latency L_t and cost C_t
  - Return results

No ML here. This is the workhorse.
"""
import time
import numpy as np
from typing import Tuple
from src.system.types import Query


class ExecutionAgent:
    def __init__(self, faiss_index, cost_model: dict):
        """
        Args:
            faiss_index: a FAISS index, typically IndexIVFPQ
            cost_model: dict with keys 'base_per_ms', 'fixed_overhead'
        """
        self.index = faiss_index
        self.cost_model = cost_model

    def search(self, query: Query, z_t: dict) -> Tuple[list, float, float]:
        """
        Returns:
            results: list of (neighbor_id, distance) tuples
            L_t: latency in seconds
            C_t: cost in USD
        """
        # Apply search parameters
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = z_t.get("nprobe", 16)

        # Prepare query
        v = np.ascontiguousarray(query.v_t.reshape(1, -1), dtype=np.float32)

        # rerank_k: search a wider candidate set, then keep top-k
        # for IVF-PQ this naturally happens by searching more candidates
        k_search = max(z_t.get("rerank_k", query.k_t * 4), query.k_t)

        # Time the actual search
        t0 = time.perf_counter()
        D, I = self.index.search(v, k_search)
        L_t = time.perf_counter() - t0

        # Take top-k results
        top_k = min(query.k_t, k_search)
        results = list(zip(I[0][:top_k].tolist(), D[0][:top_k].tolist()))

        # Compute cost
        L_ms = L_t * 1000.0
        C_t = self.cost_model["fixed_overhead"] + self.cost_model["base_per_ms"] * L_ms

        return results, L_t, C_t
