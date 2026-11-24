"""Structured cache model: explicit residency + a learned replacement unit.

See architecture_spec.md. Three units, run once per access:
  1. set decoder  -- address -> set. Single-set data: every access is set 0
                     (``sets`` may be passed per access for multi-set data).
  2. residency    -- which line sits in each way of each set. Pure bookkeeping:
                     hit <=> the accessed line is in a way of its set.
  3. replacement  -- a learned state per way h[s, w] plus a global state g, updated
                     by GRU cells whose weights are SHARED across ways and sets (one
                     policy for the whole cache). On a miss a scorer picks the victim.

Design rules (each has a reason; see the spec):
  * way identity  -- each way has a learned position vector; without it all ways stay
                     identical and the model can only do random replacement.
  * used-way broadcast -- every way is told the identity of the way just hit/filled;
                     the set summary is a mean, which forgets *which* way was used,
                     and position-dependent policies (tree-PLRU) need it.
  * fill-order naming -- empty ways are filled lowest-index first, in training and
                     evaluation alike. Which physical way the hardware uses for an
                     empty fill is unobservable from hit/miss; naming ways by fill
                     order is a relabelling, so nothing is lost as long as the
                     hardware's fill order is fixed. The learned way identities
                     absorb the relabelling.

Victims are always hard (argmax), so residency is exact and P(hit) is 0 or 1. A
trainer may override the victim decision through ``chooser`` (hindsight training,
cpl.train_hindsight). Returns logits of P(hit).
"""
from __future__ import annotations

import torch
from torch import nn

from cpl.model.hit_gru import NUM_BLOCKS

LOGIT = 13.8   # |logit| for a certain hit/miss (P = 1 - 1e-6)


class SlotCacheModel(nn.Module):
    def __init__(self, num_blocks: int = NUM_BLOCKS, ways: int = 16, sets: int = 1,
                 d_way: int = 32, d_glob: int = 32, d_mlp: int = 64, d_pos: int = 8) -> None:
        super().__init__()
        self.N, self.W, self.S = num_blocks, ways, sets
        self.d_way, self.d_glob = d_way, d_glob
        self.way_pos = nn.Parameter(torch.randn(ways, d_pos) * 0.5)     # way identity
        self.h0 = nn.Parameter(torch.randn(ways, d_way) * 0.1)          # per-way initial state
        # way input: [hit here, filled here, was empty, own id, used-way id, set mean, global]
        self.way_cell = nn.GRUCell(3 + 2 * d_pos + d_way + d_glob, d_way)
        # global input: [mean of new way states, hit]
        self.glob_cell = nn.GRUCell(d_way + 1, d_glob)
        # scorer input: [way state, own id, set mean, global, was empty]
        self.scorer = nn.Sequential(nn.Linear(d_way + d_pos + d_way + d_glob + 1, d_mlp),
                                    nn.ReLU(), nn.Linear(d_mlp, 1))

    def forward(self, accesses: torch.Tensor, sets: torch.Tensor | None = None,
                chooser=None, return_contents: bool = False):
        """``accesses`` [B, T] block ids (pad = N). ``chooser(t, way_lines, score) ->
        (v [B,W] one-hot victim, hit [B] 0/1)`` lets a trainer make the decision.
        Returns logits [B, T]; with ``return_contents`` also the line held by each way
        of the accessed set after every access, [B, T, W] (-1 = empty)."""
        B, T = accesses.shape
        dev = accesses.device
        N, W, S = self.N, self.W, self.S
        ar = torch.arange(B, device=dev)
        idx = torch.arange(W, device=dev).float()
        if sets is None:
            sets = torch.zeros_like(accesses)
        lines = torch.full((B, S, W), N, dtype=torch.long, device=dev)      # N = empty
        h = self.h0.expand(B, S, W, self.d_way).contiguous()
        pos = self.way_pos[None].expand(B, -1, -1)
        g = torch.zeros(B, self.d_glob, device=dev)
        out, contents = [], []
        for t in range(T):
            b = accesses[:, t]
            valid = b != N
            s = sets[:, t].clamp(0, S - 1)
            ls, hs = lines[ar, s], h[ar, s]                                 # [B,W], [B,W,d]
            here = ls == b[:, None]                                        # [B,W]
            hit = here.any(-1).float()
            out.append((2 * hit - 1) * LOGIT)

            empty = (ls == N).float()
            summ = hs.mean(1, keepdim=True).expand(-1, W, -1)
            gx = g[:, None].expand(-1, W, -1)
            score = self.scorer(torch.cat([hs, pos, summ, gx, empty[..., None]], -1)).squeeze(-1)
            if chooser is not None:
                v, hit = chooser(t, ls, score)
            else:
                lowest_empty = torch.where(empty > 0, -idx, torch.full_like(empty, -1e9))
                pick = torch.where(empty.any(-1, keepdim=True), lowest_empty, score).argmax(-1)
                v = nn.functional.one_hot(pick, W).float()               # fill-order naming
            hit = hit * valid.float()
            e_hit = here.float() * hit[:, None]
            e_fill = v * ((1 - hit) * valid.float())[:, None]

            ls_new = torch.where(e_fill > 0, b[:, None].expand(-1, W), ls)
            used = torch.einsum("bw,bwd->bd", e_hit + e_fill, pos)[:, None].expand(-1, W, -1)
            x = torch.cat([e_hit[..., None], e_fill[..., None], empty[..., None], pos, used, summ, gx], -1)
            hs_new = self.way_cell(x.reshape(B * W, -1), hs.reshape(B * W, -1)).view(B, W, -1)
            g_new = self.glob_cell(torch.cat([hs_new.mean(1), hit[:, None]], -1), g)

            m = nn.functional.one_hot(s, S).bool() & valid[:, None]        # write only set s
            lines = torch.where(m[..., None], ls_new[:, None], lines)
            h = torch.where(m[..., None, None], hs_new[:, None], h)
            g = torch.where(valid[:, None], g_new, g)
            if return_contents:
                contents.append(torch.where(ls_new == N, torch.full_like(ls_new, -1), ls_new))
        logits = torch.stack(out, 1)
        return (logits, torch.stack(contents, 1)) if return_contents else logits


def build_slot_model(cfg: dict) -> SlotCacheModel:
    return SlotCacheModel(num_blocks=cfg.get("num_blocks", NUM_BLOCKS), ways=cfg.get("ways", 16),
                          sets=cfg.get("sets", 1), d_way=cfg.get("d_way", 32),
                          d_glob=cfg.get("d_glob", 32), d_mlp=cfg.get("d_mlp", 64))
