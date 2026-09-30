"""Dense retrieval: thin wrapper around a FAISS IndexIDMap (inner product on L2-normalised vectors).

IndexIDMap is used from Phase 1 so vectors can later be added/removed by id
(docs/PROJECT_PLAN.md section 3.2) without changing the index type.
"""
from __future__ import annotations

import faiss
import numpy as np


class DenseIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    @staticmethod
    def _prep(vectors) -> np.ndarray:
        v = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if v.ndim == 1:
            v = v[None, :]
        faiss.normalize_L2(v)
        return v

    def add(self, ids, vectors) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        self.index.add_with_ids(self._prep(vectors), ids)

    def remove(self, ids) -> int:
        return self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def search(self, queries, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids), each shape (n_queries, k). Missing hits have id -1."""
        k = max(1, min(k, self.index.ntotal)) if self.index.ntotal else 1
        return self.index.search(self._prep(queries), k)

    def __len__(self) -> int:
        return self.index.ntotal

    def save(self, path: str) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str) -> "DenseIndex":
        index = faiss.read_index(str(path))
        obj = cls(index.d)
        obj.index = index
        return obj
