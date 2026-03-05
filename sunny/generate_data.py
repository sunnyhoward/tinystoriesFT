"""Standalone data generation + model loading for the burst experiment.

Generates the same bijection-composition data used during training and
provides helpers to load the pretrained/burst-trained models.

Dimension key:
    B: batch_size
    L: doc_len (sequence length of a full document)
    V: vocab_size

Usage:
    python generate_data.py              # prints summary + runs a quick demo
    python generate_data.py --generate   # regenerate _data.pkl from scratch
"""
import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_SEED = 999
N_A = 3
CLASS_OTHER = "other"
CLASS_BURST = "burst"


class DepthNData:
    """Pure-bijection depth-N composition data generator.

    bijections[0] = identity, bijections[1..n_a] = other-class functions,
    bijections[n_a+1] = b* (novel burst function).

    Token format (depth=3):
      S [FN ... F1] ' ' [input] ' ' [after F1] ' ' ... ' ' [after FN]
    """

    def __init__(self, n_alph: int, seq_len: int, n_a: int, depth: int,
                 burst_pos: int, seed: int):
        assert 1 <= burst_pos <= depth
        self.n_alph = n_alph
        self.seq_len = seq_len
        self.n_a = n_a
        self.depth = depth
        self.burst_pos = burst_pos
        rng = np.random.RandomState(seed)

        self.bijections = [np.arange(n_alph)]
        for _ in range(n_a + 1):
            self.bijections.append(rng.permutation(n_alph))

        self._build_vocab()
        self._build_splits(rng)

    def _build_vocab(self):
        self.token, self.token_idx, self.fn_tok = {}, {}, {}
        idx = 0
        for i in range(self.n_alph):
            self.token[idx] = f"X{i}"
            self.token_idx[f"X{i}"] = idx
            idx += 1
        for i in range(len(self.bijections)):
            self.token[idx] = f"F{i}"
            self.token_idx[f"F{i}"] = idx
            self.fn_tok[i] = idx
            idx += 1
        for sp in (' ', '<PAD>', 'S'):
            self.token[idx] = sp
            self.token_idx[sp] = idx
            idx += 1
        self.vocab_size = idx

    def _build_splits(self, rng):
        na, b_star = self.n_a, self.n_a + 1
        r = list(range(1, na + 1))
        D, bp = self.depth, self.burst_pos

        other_combos = list(itertools.product(r, repeat=D))
        rng.shuffle(other_combos)
        self.other_train = [(CLASS_OTHER,) + combo for combo in other_combos]

        remaining_combos = list(itertools.product(r, repeat=D - 1))
        burst_tasks = []
        for combo in remaining_combos:
            fns = list(combo)
            fns.insert(D - bp, b_star)
            burst_tasks.append((CLASS_BURST,) + tuple(fns))
        self.burst_train = burst_tasks

    def _make_doc(self, task: tuple) -> np.ndarray:
        fns = task[1:]
        inp = np.random.choice(self.n_alph, size=self.seq_len, replace=True)
        sp = np.array([self.token_idx[' ']])

        cur = inp.copy()
        outs = []
        for fn_idx in reversed(fns):
            cur = self.bijections[fn_idx][cur]
            outs.append(cur.copy())

        doc = [np.array([self.token_idx['S']]),
               np.array([self.fn_tok[f] for f in fns]),
               sp, inp]
        for o in outs:
            doc.extend([sp, o])
        return np.concatenate(doc)

    def gen_pool(self, tasks: list, n: int) -> dict:
        return {t: np.array([self._make_doc(t) for _ in range(n)]) for t in tasks}

    def decode(self, doc: np.ndarray) -> str:
        return " ".join(self.token.get(int(t), f"?{t}") for t in doc)


def _set_seed(seed: int):
    rng = np.random.default_rng(seed)
    true_seed = int(rng.integers(2**30))
    import random
    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.manual_seed(true_seed)
    torch.cuda.manual_seed_all(true_seed)


def pad_pools_to_same_length(*pools):
    max_len = 0
    for pool in pools:
        for docs in pool.values():
            max_len = max(max_len, docs.shape[1])
    padded = []
    for pool in pools:
        new_pool = {}
        for key, docs in pool.items():
            if docs.shape[1] < max_len:
                pad_width = max_len - docs.shape[1]
                padding = np.full((docs.shape[0], pad_width), 0, dtype=docs.dtype)
                docs = np.concatenate([docs, padding], axis=1)
            new_pool[key] = docs
        padded.append(new_pool)
    return padded


def build_data(n_alphabets: int = 10, seq_len: int = 6, n_a: int = N_A,
               depth: int = 3, burst_pos: int = 1,
               n_docs: int = 500, n_eval: int = 500):
    """Build training and eval data pools, returns everything needed."""
    _set_seed(DATA_SEED)
    d = DepthNData(n_alphabets, seq_len, n_a, depth, burst_pos, DATA_SEED)

    bg_pool = d.gen_pool(d.other_train, n_docs)
    target_pool = d.gen_pool(d.burst_train, n_docs)

    eval_pools = {
        CLASS_OTHER: d.gen_pool(d.other_train[:min(8, len(d.other_train))], n_eval),
        CLASS_BURST: d.gen_pool(d.burst_train, n_eval),
    }

    all_pools = [bg_pool, target_pool] + list(eval_pools.values())
    padded = pad_pools_to_same_length(*all_pools)
    bg_pool, target_pool = padded[0], padded[1]
    for i, k in enumerate(eval_pools):
        eval_pools[k] = padded[i + 2]

    def _cat(pool):
        if not pool:
            return np.zeros((1, list(bg_pool.values())[0].shape[1]), dtype=np.int64)
        return np.concatenate(list(pool.values()))

    eval_docs = {k: _cat(v) for k, v in eval_pools.items()}

    ref = eval_docs[CLASS_OTHER]
    sp_positions = np.where(ref[0] == d.token_idx[' '])[0]
    prompt_len = int(sp_positions[0]) + 1 + d.seq_len + 1

    return {
        "target_pool": target_pool,
        "bg_pool": bg_pool,
        "eval_docs": eval_docs,
        "prompt_len": prompt_len,
        "data_obj": d,
    }


def load_model(ckpt_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a nanoGPT model from a state_dict checkpoint.

    Uses the same architecture config as the training run:
    6 layers, 120 embed dim, 4 heads, vocab_size=128, context_size=80.
    """
    import math
    import torch.nn as nn
    from torch.nn import functional as F

    class LayerNorm(nn.Module):
        def __init__(self, ndim, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

        def forward(self, input):
            return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.dropout = config.dropout

        def forward(self, x, kv_cache=None, return_kv=False):
            B, T, C = x.size()
            head_dim = C // self.n_head
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
            q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
            if kv_cache is not None:
                k_prev, v_prev = kv_cache
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
            is_causal = kv_cache is None
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=is_causal)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            if kv_cache is not None or return_kv:
                return y, (k, v)
            return y

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.config = config
            if config.mlp:
                self.mlp = MLP(config)

        def forward(self, x, kv_cache=None, return_kv=False):
            collect_kv = kv_cache is not None or return_kv
            attn_out = self.attn(self.ln_1(x), kv_cache=kv_cache, return_kv=return_kv)
            if collect_kv:
                attn_out, new_kv = attn_out
                x = x + attn_out
                if self.config.mlp:
                    x = x + self.mlp(self.ln_2(x))
                return x, new_kv
            x = x + attn_out
            if self.config.mlp:
                x = x + self.mlp(self.ln_2(x))
            return x

    class nanoGPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.context_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ))
            self.LM_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.LM_head.weight
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx):
            b, t = idx.size()
            tok_emb = self.transformer.wte(idx)
            pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            return self.LM_head(x)

        @torch.no_grad()
        def generate(self, prompt_BT: torch.Tensor, n_new: int) -> torch.Tensor:
            B, T_prompt = prompt_BT.shape
            device = prompt_BT.device
            generated = torch.empty(B, T_prompt + n_new, dtype=torch.long, device=device)
            generated[:, :T_prompt] = prompt_BT
            tok_emb = self.transformer.wte(prompt_BT)
            pos = torch.arange(0, T_prompt, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            kv_caches = []
            for block in self.transformer.h:
                x, kv = block(x, return_kv=True)
                kv_caches.append(kv)
            x = self.transformer.ln_f(x)
            next_tok = self.LM_head(x)[:, -1, :].argmax(dim=-1)
            generated[:, T_prompt] = next_tok
            next_tok = next_tok.unsqueeze(-1)
            for i in range(1, n_new):
                t_cur = T_prompt + i
                tok_emb = self.transformer.wte(next_tok)
                pos = torch.tensor([t_cur - 1], dtype=torch.long, device=device)
                pos_emb = self.transformer.wpe(pos)
                x = self.transformer.drop(tok_emb + pos_emb)
                new_kv_caches = []
                for block, kv in zip(self.transformer.h, kv_caches):
                    x, new_kv = block(x, kv_cache=kv)
                    new_kv_caches.append(new_kv)
                kv_caches = new_kv_caches
                x = self.transformer.ln_f(x)
                next_tok = self.LM_head(x)[:, -1, :].argmax(dim=-1)
                generated[:, t_cur] = next_tok
                next_tok = next_tok.unsqueeze(-1)
            return generated

    cfg = OmegaConf.create({
        "compile": False, "vocab_size": 128, "context_size": 80,
        "n_layer": 6, "n_head": 4, "n_embd": 120,
        "dropout": 0.0, "bias": False, "mlp": True,
    })
    net = nanoGPT(cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def load_saved_data(path: str = None):
    """Load the pre-generated _data.pkl."""
    if path is None:
        path = str(SCRIPT_DIR / "_data.pkl")
    with open(path, "rb") as f:
        target_pool, bg_pool, eval_docs, prompt_len, _ = pickle.load(f)
    return {
        "target_pool": target_pool,
        "bg_pool": bg_pool,
        "eval_docs": eval_docs,
        "prompt_len": prompt_len,
    }


@torch.no_grad()
def eval_accuracy(net, docs_BL: np.ndarray, prompt_len: int,
                  device: str = "cpu") -> float:
    """Evaluate free-generation accuracy on the last 6 tokens."""
    if docs_BL.shape[0] == 0:
        return 0.0
    net.eval()
    n_new = docs_BL.shape[1] - prompt_len
    correct, total = 0, 0
    bs = 256
    for start in range(0, len(docs_BL), bs):
        batch = docs_BL[start:start + bs]
        dat = torch.as_tensor(batch, dtype=torch.long, device=device)
        full = net.generate(dat[:, :prompt_len], n_new)
        gen = full[:, prompt_len:]
        ref = dat[:, prompt_len:]
        ml = min(gen.shape[1], ref.shape[1])
        last6 = max(0, ml - 6)
        correct += (gen[:, last6:ml] == ref[:, last6:ml]).float().sum().item()
        total += ref[:, last6:ml].numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Regenerate _data.pkl from scratch")
    args = parser.parse_args()

    if args.generate:
        print("Generating data from scratch...")
        result = build_data(depth=3, burst_pos=1)
        out_path = SCRIPT_DIR / "_data.pkl"
        with open(out_path, "wb") as f:
            pickle.dump((result["target_pool"], result["bg_pool"],
                         result["eval_docs"], result["prompt_len"], None), f)
        print(f"Saved to {out_path}")
        d = result["data_obj"]
    else:
        print("Loading pre-generated data...")
        data = load_saved_data()
        _set_seed(DATA_SEED)
        d = DepthNData(10, 6, N_A, 3, 1, DATA_SEED)

    print(f"\nVocab size: {d.vocab_size}")
    print(f"Other-class tasks: {len(d.other_train)}")
    print(f"Burst-class tasks: {len(d.burst_train)}")
    print(f"\nBijections (b* = F{d.n_a + 1}):")
    for i, b in enumerate(d.bijections):
        label = "identity" if i == 0 else ("b*" if i == d.n_a + 1 else f"other_{i}")
        print(f"  F{i} ({label}): {b}")

    print(f"\nSample other-class doc:")
    doc = d._make_doc(d.other_train[0])
    print(f"  Task: {d.other_train[0]}")
    print(f"  Tokens: {d.decode(doc)}")

    print(f"\nSample burst-class doc:")
    doc = d._make_doc(d.burst_train[0])
    print(f"  Task: {d.burst_train[0]}")
    print(f"  Tokens: {d.decode(doc)}")

    models_dir = SCRIPT_DIR / "models"
    available = sorted(models_dir.glob("*.pt")) if models_dir.exists() else []
    if not available:
        print("\nNo model checkpoints found in models/")
        return

    print(f"\nAvailable models:")
    for p in available:
        print(f"  {p.name}")

    data = load_saved_data()
    prompt_len = data["prompt_len"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEvaluating on {device}...")

    for p in available:
        net = load_model(str(p), device=device)
        acc_other = eval_accuracy(net, data["eval_docs"]["other"], prompt_len, device)
        acc_burst = eval_accuracy(net, data["eval_docs"]["burst"], prompt_len, device)
        print(f"  {p.name:25s}  other={acc_other:.3f}  burst={acc_burst:.3f}")


if __name__ == "__main__":
    main()
