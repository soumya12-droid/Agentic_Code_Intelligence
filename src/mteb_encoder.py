"""MTEB encoder for the pipeline (Phase 1: dense-only baseline, see docs/PROJECT_PLAN.md section 7)."""
from __future__ import annotations

import logging

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType  # noqa: F401
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "jinaai/jina-embeddings-v2-base-code"
FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"


class PrePostPipelineEncoder(AbsEncoder):
    """Wraps a single dense embedding model. Later phases will put the full
    retrieve/fuse/rerank pipeline behind this same interface."""

    def __init__(self, model_name: str = PRIMARY_MODEL, max_seq_length: int = 512,
                 batch_size: int = 64, device: str = "cpu", **kwargs):
        self.batch_size = batch_size
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
            self.model_name = model_name
        except Exception as e:  # download/remote-code/dependency failure
            logger.warning("Failed to load %s (%s: %s); falling back to %s",
                           model_name, type(e).__name__, e, FALLBACK_MODEL)
            self.model = SentenceTransformer(FALLBACK_MODEL, device=device)
            self.model_name = FALLBACK_MODEL
        self.model.max_seq_length = max_seq_length
        self.mteb_model_meta = ModelMeta.create_empty(
            {"name": f"prepost-pipeline/{self.model_name.split('/')[-1]}",
             "revision": "local"}
        )

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs) -> np.ndarray:
        texts = [t for batch in inputs for t in batch["text"]]
        emb = self.model.encode(texts, batch_size=self.batch_size,
                                convert_to_numpy=True, normalize_embeddings=True,
                                show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)
