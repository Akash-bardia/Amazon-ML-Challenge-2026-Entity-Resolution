"""Sentence embeddings (multilingual-e5 / bge-m3) and chunked exact k-NN search.

Memory-safe on a 6 GB GPU: the pool is streamed through the GPU in chunks of
`knn_p_chunk` rows and queries in chunks of `knn_q_chunk`, keeping a running
top-k per query. Results are identical to a full exact search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .utils import LOG


def pick_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


class Encoder:
    def __init__(self, model_name: str, device: str, max_len: int, prefix: str):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.max_len = max_len
        self.prefix = prefix
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device == "cuda":
            self.model = self.model.half()
        self.model.to(device).eval()
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts: list[str], batch: int, desc: str = "") -> np.ndarray:
        """Return L2-normalised float16 embeddings; identical texts are encoded once."""
        n = len(texts)
        out = np.zeros((n, self.dim), dtype=np.float16)
        if n == 0:
            return out
        inv, uniq = pd.factorize(pd.Series(texts, dtype=object), sort=False)
        uniq = np.asarray(uniq, dtype=object)
        order = np.argsort(np.fromiter((len(t) for t in uniq), dtype=np.int32, count=len(uniq)))[::-1]
        uemb = np.zeros((len(uniq), self.dim), dtype=np.float16)
        n_batches = (len(uniq) + batch - 1) // batch
        for b in range(n_batches):
            idx = order[b * batch:(b + 1) * batch]
            enc = self.tok([self.prefix + uniq[i] for i in idx], padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            hid = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
            emb = (hid * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            uemb[idx] = emb.cpu().numpy().astype(np.float16)
            if (b + 1) % 500 == 0 or b + 1 == n_batches:
                LOG.info(f"  embed {desc}: {min((b + 1) * batch, len(uniq)):,}/{len(uniq):,} unique texts")
        out[:] = uemb[inv]
        return out


@torch.inference_mode()
def knn(Q: np.ndarray, P: np.ndarray, k: int, device: str, q_chunk: int, p_chunk: int
        ) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k of each row of Q over rows of P.

    Returns (idx [nq,k] int32 with -1 for missing, score [nq,k] float32).
    """
    nq, npool = len(Q), len(P)
    k_eff = min(k, npool)
    if nq == 0 or npool == 0:
        return np.full((nq, k), -1, np.int32), np.full((nq, k), -np.inf, np.float32)
    dt = torch.float16 if device == "cuda" else torch.float32
    Qt = torch.from_numpy(np.array(Q)).to(device=device, dtype=dt)          # np.array: copy out of memmap
    best_s = torch.full((nq, k_eff), -float("inf"), device=device, dtype=torch.float32)
    best_i = torch.full((nq, k_eff), -1, device=device, dtype=torch.int64)
    for p0 in range(0, npool, p_chunk):
        Pc = torch.from_numpy(np.array(P[p0:p0 + p_chunk])).to(device=device, dtype=dt)
        kk = min(k_eff, len(Pc))
        for q0 in range(0, nq, q_chunk):
            s = Qt[q0:q0 + q_chunk] @ Pc.T                      # fp16 on GPU: q_chunk x p_chunk
            v, i = torch.topk(s, kk, dim=1)
            del s
            cat_s = torch.cat([best_s[q0:q0 + q_chunk], v.float()], dim=1)
            cat_i = torch.cat([best_i[q0:q0 + q_chunk], i + p0], dim=1)
            v2, j = torch.topk(cat_s, k_eff, dim=1)
            best_s[q0:q0 + q_chunk] = v2
            best_i[q0:q0 + q_chunk] = torch.gather(cat_i, 1, j)
        del Pc
        LOG.info(f"  knn: pool rows {min(p0 + p_chunk, npool):,}/{npool:,}")
    idx = best_i.cpu().numpy().astype(np.int32)
    sc = best_s.cpu().numpy()
    if k_eff < k:
        idx = np.pad(idx, ((0, 0), (0, k - k_eff)), constant_values=-1)
        sc = np.pad(sc, ((0, 0), (0, k - k_eff)), constant_values=-np.inf)
    if device == "cuda":
        torch.cuda.empty_cache()
    return idx, sc


def pair_cosine(Eq: np.ndarray, Ep: np.ndarray, q: np.ndarray, t: np.ndarray, chunk: int = 2_000_000
                ) -> np.ndarray:
    """Cosine for arbitrary (q, t) index pairs (embeddings already normalised)."""
    out = np.empty(len(q), dtype=np.float32)
    for i in range(0, len(q), chunk):
        a = Eq[q[i:i + chunk]].astype(np.float32)
        b = Ep[t[i:i + chunk]].astype(np.float32)
        out[i:i + chunk] = np.einsum("ij,ij->i", a, b)
    return out
