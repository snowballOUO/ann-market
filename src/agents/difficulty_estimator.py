"""
Agent 1: DifficultyEstimator

Produces U_t — the observable confounder used for causal correction.

Week 1 version uses a heuristic, not a trained model. The interface is
identical to the eventual MLP version, so swapping it in later is one line.

CRITICAL invariant: U_t must depend ONLY on the query, never on anything
downstream (results, latency, buyer behaviour). If you violate this, the
causal correction story falls apart.
"""
import numpy as np
from typing import Optional
from src.system.types import Query


class DifficultyEstimator:
    """Heuristic-based difficulty estimator (Week 1)."""

    def __init__(self, sample_vectors: Optional[np.ndarray] = None):
        """
        Args:
            sample_vectors: Optional (N, dim) array of representative vectors,
                used to estimate per-query local density. If None, we skip
                density estimation and use only intrinsic features.
        """
        self.sample_vectors = sample_vectors
        if sample_vectors is not None:
            self.global_mean_norm = float(np.linalg.norm(sample_vectors, axis=1).mean())
        else:
            self.global_mean_norm = 1.0

    def estimate(self, query: Query) -> float:
        """Return U_t in [0, 1] where 0 = easy, 1 = hard."""
        v = query.v_t

        # Feature 1: vector norm relative to global mean
        # (queries far from the centroid tend to be harder)
        norm = float(np.linalg.norm(v))
        norm_ratio = min(norm / max(self.global_mean_norm, 1e-6), 3.0) / 3.0

        # Feature 2: filter selectivity (more restrictive filter = harder)
        # filter_t example: {"category": "shoes"} → selectivity ~0.1
        # we approximate selectivity from filter dict size for Week 1
        n_filter_keys = len(query.filter_t) if query.filter_t else 0
        filter_difficulty = min(n_filter_keys * 0.2, 0.6)

        # Feature 3: requested k (larger k is marginally harder)
        k_difficulty = min(query.k_t / 100.0, 1.0) * 0.3

        # Feature 4: local density if available
        density_difficulty = 0.0
        if self.sample_vectors is not None:
            # quick density estimate via distance to nearest sample
            dists = np.linalg.norm(self.sample_vectors[:1000] - v, axis=1)
            nearest = float(np.min(dists))
            # higher distance = sparser region = harder
            density_difficulty = min(nearest / max(self.global_mean_norm, 1e-6), 1.0) * 0.4

        # Combine and clip to [0, 1]
        U_t = 0.3 * norm_ratio + 0.3 * filter_difficulty + 0.2 * k_difficulty + 0.2 * density_difficulty
        return float(np.clip(U_t, 0.0, 1.0))


class MLPDifficultyEstimator:
    """MLP-based difficulty estimator (Week 4).

    Replaces the heuristic weighted sum with a pre-trained 3-layer MLP.
    Input:  6 statistical features extracted from the query vector
    Output: U_t ∈ [0, 1] where 0 = easy, 1 = hard

    Uses ONNX runtime for inference — p99 < 0.5ms on CPU.
    """

    def __init__(self, onnx_path="models/difficulty_v1.onnx",
                 sample_vectors=None):
        """
        Args:
            onnx_path:      path to the exported ONNX model
            sample_vectors: reference vectors for density estimation
        """
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)

        self.sample_vectors = sample_vectors
        if sample_vectors is not None:
            self.global_mean_norm = float(
                np.linalg.norm(sample_vectors, axis=1).mean()
            )
        else:
            self.global_mean_norm = 1.0

    def _extract_features(self, query: Query) -> np.ndarray:
        """Extract the same 6 features used in training."""
        v = query.v_t

        # 1. Norm ratio
        norm = float(np.linalg.norm(v))
        norm_ratio = min(norm / max(self.global_mean_norm, 1e-6), 3.0) / 3.0

        # 2. Local density
        if self.sample_vectors is not None:
            dists = np.linalg.norm(self.sample_vectors[:1000] - v, axis=1)
            nearest = float(np.min(dists))
            density = min(nearest / max(self.global_mean_norm, 1e-6), 1.0)
        else:
            density = 0.0

        # 3. Filter complexity
        n_filter_keys = len(query.filter_t) if query.filter_t else 0
        filter_diff = min(n_filter_keys * 0.2, 0.6)

        # 4. Requested k (normalised)
        k_norm = min(query.k_t / 100.0, 1.0)

        # 5. SLA tightness
        sla_norm = 1.0 - min(query.sla_t / 0.1, 1.0)

        # 6. Bias intercept
        bias = 1.0

        return np.array(
            [norm_ratio, density, filter_diff, k_norm, sla_norm, bias],
            dtype=np.float32,
        )

    def estimate(self, query: Query) -> float:
        """Return U_t ∈ [0, 1]."""
        features = self._extract_features(query)

        # ONNX inference: (6,) → (1, 6) → forward → (1, 1) → scalar
        inputs = {"features": features.reshape(1, -1)}
        output = self.session.run(None, inputs)[0]

        return float(np.clip(output[0].ravel()[0], 0.0, 1.0))