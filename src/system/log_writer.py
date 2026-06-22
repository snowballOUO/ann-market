"""
Append-only trajectory log writer.

Buffers in memory, flushes to parquet every N records or on close.
parquet is chosen because:
  - columnar layout → cheap to read just propensity/U_t/R_t when doing analysis
  - schema enforcement → catches bugs where you forget a field
  - small files (this is single-machine prototype, not a streaming pipeline)
"""
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
from threading import Lock
from src.system.types import Trajectory


class LogWriter:
    def __init__(self, output_dir: str, flush_every_n: int = 100):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.flush_every_n = flush_every_n
        self.buffer: list[dict] = []
        self.lock = Lock()
        self.shard_id = 0
        self.run_id = f"run_{int(time.time())}"
        self.recall_pending: dict[str, float] = {}  # query_id -> recall

    def write_async(self, traj: Trajectory):
        """Append a trajectory. Flushes when buffer full."""
        row = traj.flatten_for_log()
        with self.lock:
            self.buffer.append(row)
            if len(self.buffer) >= self.flush_every_n:
                self._flush()

    def record_recall(self, query_id: str, recall: float):
        """Late-arriving recall from shadow sampler. Buffer it; we'll attach on flush."""
        with self.lock:
            self.recall_pending[query_id] = recall

    def _flush(self):
        """Caller must hold self.lock."""
        if not self.buffer:
            return
        # Attach any late-arriving recall values
        for row in self.buffer:
            qid = row["query_id"]
            if qid in self.recall_pending:
                row["Q_t"] = self.recall_pending.pop(qid)
        table = pa.Table.from_pylist(self.buffer)
        path = os.path.join(self.output_dir, f"{self.run_id}_shard{self.shard_id:04d}.parquet")
        pq.write_table(table, path)
        self.shard_id += 1
        self.buffer.clear()

    def close(self):
        """Flush remaining buffer. Call at end of run."""
        with self.lock:
            self._flush()
            # Also write any orphan recalls to their own file (these arrived after rows were flushed)
            if self.recall_pending:
                orphan = [{"query_id": qid, "Q_t": r} for qid, r in self.recall_pending.items()]
                table = pa.Table.from_pylist(orphan)
                path = os.path.join(self.output_dir, f"{self.run_id}_orphan_recall.parquet")
                pq.write_table(table, path)
                self.recall_pending.clear()
