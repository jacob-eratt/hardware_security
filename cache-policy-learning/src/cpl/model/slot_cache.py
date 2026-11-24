"""Structured cache model: explicit residency slots + a learned replacement GRU.

Three parts (see the plan in the repo discussion):
  1. set decoder   -- address -> set. Here a given per-access set id (``sets``);
                      single-set data uses set 0 everywhere.
  2. residency     -- R[b, s, w, :] = probability that way w of set s holds each
                      line (last index = "empty / some foreign line"). Pure
                      bookkeeping, nothing learned: P(hit) = sum_w R[s, w, line].
  3. replacement   -- per-way states h[s, w] and a global state g, all updated by
                      GRU cells whose weights are SHARED across ways and sets, so
                      the model learns one policy for the whole cache. On a miss a
                      scorer over the ways (from h, g, emptiness) picks the victim.

Training is soft: the victim is a softmax, so a fill writes the new line into each
way in proportion to its victim probability and everything is differentiable.
``tf`` (teacher forcing, 0..1) mixes the *measured* outcome into the hit/miss
branch so residency follows the real history early in training; evaluation
always uses tf=0 (the model never sees outcomes). ``hard=True`` takes the argmax
victim, turning the model into a deterministic cache simulator.

Returns logits of P(hit) so it drops into cpl.train_hits unchanged.
"""
from __future__ import annotations

import torch
from torch import nn

from cpl.model.hit_gru import NUM_BLOCKS

EPS = 1e-6


class SlotCacheModel(nn.Module):
    uses_targets = True

    def __init__(self, num_blocks: int = NUM_BLOCKS, ways: int = 16, sets: int = 1,
                 d_way: int = 32, d_glob: int = 32, d_mlp: int = 64, victim: str = "soft") -> None:
        super().__init__()
        # victim="soft": victim is a softmax (residency blurs early in training).
        # victim="st":   straight-through -- forward takes ONE way (argmax, + Gumbel
        #                exploration noise while training) so residency stays crisp
        #                like a real cache; backward uses the softmax gradient.
        assert victim in ("soft", "st")
        self.victim = victim
        # Victim softmax temperature, annealed by the trainer (1 = soft -> small =
        # nearly hard). A buffer, so a checkpoint keeps the tau it was selected at.
        self.register_buffer("tau", torch.tensor(1.0))
        # Bonus added to a way's score in proportion to how "empty" it is (holds a
        # pre-stream line); >0 means fill those first. Also saved with the model.
        self.register_buffer("empty_bonus", torch.tensor(0.0))
        self.N, self.W, self.S = num_blocks, ways, sets
        self.d_way, self.d_glob = d_way, d_glob
        # Each way has a learned identity: without it all ways start identical, get
        # identical inputs, and the model can never break the symmetry (it degenerates
        # to "evict uniformly at random"). Real ways have fixed positions anyway.
        self.d_pos = 8
        self.way_pos = nn.Parameter(torch.randn(ways, self.d_pos) * 0.5)
        # per-way input: [hit-in-way, filled-in-way, was-empty, way id, id of the way that
        # was just hit/filled (broadcast), set summary, global]. The broadcast id matters:
        # the set summary is a mean, which forgets WHICH way was used, and policies like
        # tree-PLRU update each way depending on where the used way sits relative to it.
        self.way_cell = nn.GRUCell(3 + 2 * self.d_pos + d_way + d_glob, d_way)
        # global input: [mean of new way states, P(hit) of this access]
        self.glob_cell = nn.GRUCell(d_way + 1, d_glob)
        self.scorer = nn.Sequential(
            nn.Linear(d_way + self.d_pos + d_way + d_glob + 1, d_mlp), nn.ReLU(), nn.Linear(d_mlp, 1))
        self.h0 = nn.Parameter(torch.randn(ways, d_way) * 0.1)   # per-way initial state

    def forward(self, accesses: torch.Tensor, targets: torch.Tensor | None = None,
                tf: float = 0.0, sets: torch.Tensor | None = None, hard: bool = False,
                victims: torch.Tensor | None = None, return_scores: bool = False, chooser=None):
        """``victims`` [B,T] long (diagnostic only, simulated data): the true way
        filled at each miss (-1 on hits). When given, residency follows the true
        hit/miss and victims exactly, so the scorer/state can be trained by direct
        supervision. ``return_scores`` also returns the per-step victim scores [B,T,W].
        ``chooser(t, way_lines, score) -> (v [B,W] one-hot victim, ph [B] hit flags)``
        lets a trainer make the victim decision (hindsight training); ``way_lines``
        [B,W] is the line held by each way of the accessed set (N = empty)."""
        B, T = accesses.shape
        dev = accesses.device
        N, W, S = self.N, self.W, self.S
        ar = torch.arange(B, device=dev)
        if sets is None:
            sets = torch.zeros_like(accesses)
        R = torch.zeros(B, S, W, N + 1, device=dev); R[..., N] = 1.0      # all ways empty
        h = self.h0.expand(B, S, W, self.d_way).contiguous()
        pos = self.way_pos[None].expand(B, -1, -1)                # [B,W,d_pos]
        g = torch.zeros(B, self.d_glob, device=dev)
        out, scores = [], []
        for t in range(T):
            b = accesses[:, t]
            valid = (b != N)                                   # pad id == N
            bb = torch.where(valid, b, torch.zeros_like(b))
            s = sets[:, t].clamp(0, S - 1)
            oh_s = nn.functional.one_hot(s, S).float()          # [B, S]
            Rs, hs = R[ar, s], h[ar, s]                          # [B,W,N+1], [B,W,d]

            q = Rs[ar, :, bb]                                    # [B,W] P(way holds b)
            p = q.sum(-1).clamp(EPS, 1 - EPS)                    # P(hit)
            out.append(torch.log(p) - torch.log1p(-p))

            empty = Rs[..., N]                                   # [B,W]
            summ = hs.mean(1, keepdim=True).expand(-1, W, -1)
            gx = g[:, None].expand(-1, W, -1)
            score = self.scorer(torch.cat([hs, pos, summ, gx, empty[..., None]], -1)).squeeze(-1)
            score = (score + self.empty_bonus * empty) / self.tau
            scores.append(score)
            if hard:
                v = nn.functional.one_hot(score.argmax(-1), W).float()
            elif self.victim == "st":
                z = score
                if self.training:                                # Gumbel exploration
                    z = z - torch.log(-torch.log(torch.rand_like(z).clamp(EPS, 1 - EPS)))
                soft = torch.softmax(z, -1)
                v = nn.functional.one_hot(z.argmax(-1), W).float() - soft.detach() + soft
            else:
                v = torch.softmax(score, -1)                     # victim distribution

            ph = p
            if targets is not None and tf > 0:
                ph = tf * targets[:, t].clamp(0, 1) + (1 - tf) * p
            if hard:
                ph = (p > 0.5).float()
            if chooser is not None:                              # trainer decides
                v, ph = chooser(t, Rs.argmax(-1), score)
                q = q.detach()
            if victims is not None:                              # oracle: exact history
                vt = victims[:, t]
                ph = (vt < 0).float()
                v = nn.functional.one_hot(vt.clamp_min(0), W).float() * (vt >= 0).float()[:, None]
                q = q.detach()
            e_hit = ph[:, None] * q / p[:, None]                 # touched this way
            e_fill = (1 - ph)[:, None] * v                       # line b filled here

            Rs_new = Rs * (1 - e_fill[..., None]) + e_fill[..., None] * nn.functional.one_hot(bb, N + 1).float()[:, None]
            used = torch.einsum("bw,bwd->bd", e_hit + e_fill, pos)[:, None].expand(-1, W, -1)
            x = torch.cat([e_hit[..., None], e_fill[..., None], empty[..., None], pos, used, summ, gx], -1)
            hs_new = self.way_cell(x.reshape(B * W, -1), hs.reshape(B * W, -1)).view(B, W, -1)
            g_new = self.glob_cell(torch.cat([hs_new.mean(1), ph[:, None]], -1), g)

            keep = valid.float()
            m4 = (oh_s * keep[:, None])[:, :, None, None]            # write only set s, valid steps
            R = R * (1 - m4) + m4 * Rs_new[:, None]
            h = h * (1 - m4) + m4 * hs_new[:, None]
            g = torch.where(valid[:, None], g_new, g)
        logits = torch.stack(out, 1)
        return (logits, torch.stack(scores, 1)) if return_scores else logits


def build_slot_model(cfg: dict) -> SlotCacheModel:
    return SlotCacheModel(
        num_blocks=cfg.get("num_blocks", NUM_BLOCKS), ways=cfg.get("ways", 16),
        sets=cfg.get("sets", 1), d_way=cfg.get("d_way", 32), d_glob=cfg.get("d_glob", 32),
        d_mlp=cfg.get("d_mlp", 64), victim=cfg.get("victim", "soft"))
