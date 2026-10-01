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

    def search(self, queries, k: int = 10, allowed_ids=None) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids), each shape (n_queries, k). Missing hits have id -1.

        allowed_ids restricts the search to those vector ids (a faiss IDSelectorBatch), which
        is how a versioned index searches "as of" a version without rebuilding anything."""
        k = max(1, min(k, self.index.ntotal)) if self.index.ntotal else 1
        if allowed_ids is None:
            return self.index.search(self._prep(queries), k)
        allowed = np.ascontiguousarray(np.asarray(list(allowed_ids), dtype=np.int64))
        n = np.atleast_2d(np.asarray(queries)).shape[0]
        if allowed.size == 0:
            return np.full((n, k), -np.inf, dtype=np.float32), np.full((n, k), -1, dtype=np.int64)
        params = faiss.SearchParameters()
        params.sel = faiss.IDSelectorBatch(allowed)
        return self.index.search(self._prep(queries), k, params=params)

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
