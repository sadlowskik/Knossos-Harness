"""Two-tier memory.

Mnemosyne  -- lossy gist compression of high-level context into a few vectors
              (Gisting, Mu et al. 2023; Recurrent Memory Transformer, Bulatov
              et al. 2022). Good for fuzzy context.
Scribe     -- an exact, never-compressed symbol table parsed from source via
              Python's `ast`. Good for facts a compiler cares about
              (identifiers, signatures, imports, line numbers). Non-neural.

The split is deliberate: approximate the prose, but keep anything a compiler
cares about bit-exact.
"""
from __future__ import annotations
import ast
from typing import Dict, List, Optional, Set
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Embeddings, Block


class Mnemosyne(nn.Module):
    """Compress a segment of token states (B, T, C) into `n_gist` vectors via
    learned-query cross-attention."""

    def __init__(self, n_embd: int, n_gist: int = 16, n_head: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_gist, n_embd) * 0.02)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        gist, _ = self.attn(q, x, x)
        return self.ln(gist)


class MemoryModel(nn.Module):
    """Reference demo: encode segment A -> gist -> prepend gist to segment B and
    predict B. Set `use_memory=False` to zero the gist (ablation)."""

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 n_gist: int = 16, block_size: int = 128):
        super().__init__()
        self.emb = Embeddings(vocab_size, n_embd, block_size)
        self.encoder = Block(n_embd, n_head, block_size)
        self.mnemosyne = Mnemosyne(n_embd, n_gist, n_head)
        self.decoder = Block(n_embd, n_head, block_size + n_gist)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.n_gist = n_gist

    def forward(self, seg_a: torch.Tensor, seg_b: torch.Tensor,
                targets_b: torch.Tensor, use_memory: bool = True) -> torch.Tensor:
        gist = self.mnemosyne(self.encoder(self.emb(seg_a)))
        if not use_memory:
            gist = torch.zeros_like(gist)
        seq = torch.cat([gist, self.emb(seg_b)], dim=1)
        logits = self.lm_head(self.ln_f(self.decoder(seq)))[:, self.n_gist:, :]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets_b.reshape(-1))


class Scribe:
    """Exact, never-compressed symbol table extracted from Python source via AST.

    Not a neural network -- it parses. Returns ground truth, not a best guess.
    """

    def __init__(self) -> None:
        self.functions: Dict[str, dict] = {}
        self.classes: Dict[str, dict] = {}
        self.imports: List[str] = []
        self.assignments: Set[str] = set()

    def ingest(self, code: str, filepath: str = "<unknown>") -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                self.functions[node.name] = {
                    "signature": f"{node.name}({', '.join(args)})",
                    "file": filepath, "lineno": node.lineno,
                }
            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                self.classes[node.name] = {"methods": methods, "file": filepath, "lineno": node.lineno}
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imports.append(node.module)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assignments.add(target.id)

    def lookup(self, name: str) -> Optional[dict]:
        return self.functions.get(name) or self.classes.get(name)
