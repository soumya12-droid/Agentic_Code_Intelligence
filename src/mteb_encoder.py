"""MTEB encoder for the pipeline (Phase 1: dense-only baseline, see docs/PROJECT_PLAN.md section 7)."""
from __future__ import annotations

import logging

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# bge-small is the default: jina-embeddings-v2-base-code (pass it via model_name) was
# too slow on CPU for the full AppsRetrieval run.
PRIMARY_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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
        # bge models want an instruction prefix on the query side only
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        self.mteb_model_meta = ModelMeta.create_empty(
            {"name": f"prepost-pipeline/{self.model_name.split('/')[-1]}",
             "revision": "local"}
        )

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs) -> np.ndarray:
        texts = [t for batch in inputs for t in batch["text"]]
        if prompt_type == PromptType.query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        emb = self.model.encode(texts, batch_size=self.batch_size,
                                convert_to_numpy=True, normalize_embeddings=True,
                                show_progress_bar=True)
        return np.asarray(emb, dtype=np.float32)
