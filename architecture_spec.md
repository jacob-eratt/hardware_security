# Whole-Cache Model — Architecture Specification

**Project:** Learning and Identifying Cache Replacement Policies with Neural Sequence Models
**Team:** Henry Tudor · Dhwanit Kumawat · Jacob Erattuparambil
**Scope:** the structured model that predicts per-access hit/miss for a whole cache (many sets) from measured hardware streams, how it is trained without victim labels, how it is validated, and the per-processor pipeline it runs in. It supersedes the plain GRU (`cpl.model.hit_gru`) as the main model; the plain GRU stays as the baseline.
**Audience:** a human or an LLM coding agent implementing or extending this model. **MUST** = required; **SHOULD** = default unless there is a clear reason not to.
**Status (2026-11-24, evening):** see §9. Parts 2 and 3 exist for a single set and pass the single-set simulator suite (LRU, FIFO, bit-PLRU, tree-PLRU, SRRIP) trained from hit/miss labels only. Part 1, multi-set data, DRRIP and hardware training do not exist yet.

---

## 0. TL;DR

The model has three units, run once per memory access:

1. **Set decoder** (learned): line address → set identity. Pretrained on "same set or not" pairs measured on the machine, then frozen.
2. **Residency tracker** (not learned): for every set, which line sits in each of its W ways. Hit ⇔ the line is in a slot of its set.
3. **Replacement unit** (learned): a small state per way, a global state, and a victim scorer, all with weights **shared across ways and sets** — one replacement policy for the whole cache.

Training uses **hindsight victim candidates** derived from hit/miss labels (§5). The model is **validated on a suite of simulated policies** (§6) before any hardware result is reported, and every processor goes through the **same pipeline** with only an auto-generated config differing (§7).

---

## 1. Goal and non-goals

**Goal:** a model that, given only the sequence of line addresses a program accesses, predicts whether each access hits or misses in a given cache level of a real processor, and that works the same way on any processor we measure (currently an Intel Celeron N3450; next, an Intel i7 of unknown model, both on Ubuntu).

The model MUST NOT assume a known replacement policy. Fixed policies (LRU, PLRU, RRIP, …) appear only as baselines and as simulated validation targets.

**Non-goals (for now):** coherence traffic, other cores' activity, prefetcher modelling (prefetch effects are suppressed in measurement, §8), writes/dirty state, and inclusive/exclusive multi-level interactions beyond the L1 caveat in §10.

---

## 2. Terminology (MUST)

| Term | Meaning |
|---|---|
| **line** | a 64-byte cache line, identified by its line address `a >> 6` |
| **W** | associativity of the modelled cache level (read from sysfs, §7) |
| **S_hw** | number of sets in that level (read from sysfs) |
| **stream** | one measured sequence of accesses, length 32–96 (SHOULD), with a per-access hit probability `p_hit` from R timed repeats |
| **label** | majority outcome: hit ⇔ `p_hit > 0.5` |
| **local set slot** | the index of a set *within one stream*; a stream touching k distinct sets has slots 0..k−1 (§3.4) |
| **way identity** | a learned vector per way index 0..W−1 |
| **victim** | the way overwritten at a miss |
| **free-running** | evaluation mode: hard victims, no labels used |

---

## 3. Architecture

### 3.1 Unit 1 — set decoder (learned)

- **Input (MUST):** the line address as a bit vector. For L2 with 2 MiB huge pages: address bits 6..20 (15 bits), which are identical in virtual and physical addresses. For later L3 work: physical address bits 6..(≥33).
- **Output:** a distribution over `K = S_hw` set classes, `q(a) ∈ Δ^K`, plus a set embedding `e(a) ∈ R^{d_set}` (SHOULD `d_set = 16`) computed from the decoder's last hidden layer.
- **Network (SHOULD):** MLP, 2 hidden layers of 128, ReLU.
- **Training (MUST):** pairwise congruence. For an address pair `(a, b)` with label `y = 1` if they map to the same set: `P(same) = Σ_k q_k(a) q_k(b)`; binary cross-entropy on `y`. Positive pairs come from eviction pools found by the engine (§8.2); negatives from addresses in different pools. Class labels are arbitrary up to permutation; only grouping is supervised.
- **At cache-model time (MUST, v1):** frozen; the hard set class `argmax q(a)` defines which set an access goes to, and `e(a)` is the set identity fed to the replacement unit.
- **Validation (MUST, L2):** on held-out addresses, the decoder's grouping must agree with the bit-slice formula `set = (a >> 6) mod S_hw` on ≥ 99.9% of pairs (both positive and negative). This tests the decoder, and — independently — confirms the hardware's L2 mapping is the plain slice. If agreement fails, investigate before training anything downstream; it is a hardware finding or a decoder bug.

### 3.2 Unit 2 — residency tracker (not learned)

- **State (MUST):** `R[b, j, w, :]` for stream `b`, local set slot `j`, way `w`: a distribution over `N + 1` line tokens (stream-local line ids 0..N−1, plus token `N` = "empty / holds a pre-stream line"). Initialised to token `N` everywhere.
- **Hit probability (MUST):** for an access to line `x` in slot `j`: `p_hit = Σ_w R[b, j, w, x]`.
- **Update:** on a fill of way `w` with line `x`, `R[b, j, w, :] ← onehot(x)`. Only slot `j` changes.
- Victims are always hard (MUST; soft and straight-through victims were tried and removed, §9), so `R` is one-hot and `p_hit ∈ {0, 1}`. The implementation stores `R` as an integer line id per way.
- **Fill-order naming (MUST):** when the accessed set has an empty way, the fill goes to the **lowest-index empty way**, with no score and no loss, in training and evaluation alike. Which physical way hardware uses for an empty fill cannot be observed from hit/miss. Naming ways by fill order is a relabelling, so nothing is lost as long as the hardware's fill order is fixed; the learned way identities absorb the relabelling. Without this rule, hindsight training on tree-PLRU stalled (§9).
- **Line ages (MUST):** per way, two exact counters derived from the access sequence, nothing learned:
  - `rec[w]` = recency rank of the way's line (0 = most recently used, normalised by `W−1`);
  - `ins[w]` = insertion rank (0 = most recently filled, normalised by `W−1`).
  Empty ways have time −1, so they rank oldest. These are bookkeeping, like `R`, and live in the residency tracker.

### 3.3 Unit 3 — replacement unit (learned)

**State (MUST):**
- per way: `h[b, j, w] ∈ R^{d_way}` (SHOULD `d_way = 32`), initialised from a learned per-way vector `h0[w]`;
- global: `g[b] ∈ R^{d_glob}` (SHOULD `d_glob = 32`), initialised to 0.

**Inputs that MUST be present** (each was found necessary experimentally, §9):
- **way identity** `pos[w] ∈ R^{8}` — without it all ways stay identical and the model degenerates to random replacement;
- **used-way broadcast** `u = Σ_w (e_hit[w] + e_fill[w]) · pos[w]` — the identity of the way just hit or filled, given to every way; without it tree-PLRU is unrepresentable (the set summary is a mean, which forgets which way was used);
- **set summary** `m = mean_w h[b, j, w]` (SHOULD be upgraded to a learned set-level vector if the validation suite finds a policy the mean cannot express);
- **set identity** `e(a)` from the decoder — lets the model distinguish e.g. set-dueling leader sets from followers; MUST be fed to the scorer, the way cell and the global cell;
- **global state** `g`;
- **line ages** `age[w] = [rec[w], ins[w], rec[0..W−1], ins[0..W−1]]` (dimension `2 + 2W`): the way's own ranks plus the **whole set's** rank vectors in way order, given to every way. Rules such as tree-PLRU's "which pair in this quarter was used less recently" compare *other* ways' ages, so a way's own ranks are not enough. Without ages, hindsight training on tree-PLRU reached only 0.985 accuracy and 0.637 exact-stream; with them, 1.000 and 0.999 (§9).

**Victim scorer (MUST):** for each way of the accessed set,
`score[w] = MLP([h[w], pos[w], m, g, e(a), empty[w], age[w]])`, MLP 64 hidden, ReLU. Victim = `argmax_w score[w]` over full sets (empty ways are handled by fill-order naming, §3.2).

**Way update (MUST):** one shared `GRUCell` over all W ways of the accessed set:
`h[w] ← GRUCell([e_hit[w], e_fill[w], empty[w], pos[w], u, m, g, e(a), age[w]], h[w])`,
where `e_hit[w]` = 1 if this way held the hit line, `e_fill[w]` = 1 if this way was filled. Ways of other sets are not updated.

**Global update (MUST):** `g ← GRUCell_g([mean_w h_new[w], hit, e(a)], g)` on every access.

**Weight sharing (MUST):** the way cell, scorer and global cell are the same for every way and every set. The only per-way parameters are `pos` and `h0`; there are no per-set parameters (set differences enter only through `e(a)`).

### 3.4 Batching and memory

Streams touch few sets, so states are kept per **local set slot**, not for all `S_hw` sets: before a batch, each stream's accesses are decoded and the distinct sets numbered 0..k−1 (k ≤ `S_max`, SHOULD 32). Tensors are `[B, S_max, W, ·]`; each step gathers slot `j`, updates it, and writes it back with a one-hot mask. The set *identity* that distinguishes sets across streams is `e(a)`, never the slot number.

### 3.5 One access, step by step

1. `j, e = decoder(a)` → local slot, set embedding.
2. `p_hit = Σ_w R[j, w, x]` — model output for this access.
3. Hit: `e_hit = onehot(way holding x)`. Miss: if set `j` has an empty way, `v` = lowest empty way; else compute `score` and pick victim `v`. `e_fill = onehot(v)`, write `x` into `R[j, v]`.
4. Update `h[j, ·]` with the shared way cell, then `g` with the global cell.
5. Update the ages of set `j`: last-use time of the hit/filled way, and insertion time of a filled way.

---

## 4. Modes of the forward pass

| Mode | Victim | Used for |
|---|---|---|
| **free-running** | `argmax score` | all evaluation and reported numbers (MUST) |
| **hindsight** | best-scoring *allowed* way (§5) | training on hardware and simulated data |
| **victim labels** | true victim supplied, cross-entropy on it | capacity tests on simulated data only (`--victim-labels`); MUST NOT be used for reported results |

Soft victims, straight-through victims, a temperature `tau` and an `empty_bonus` were tried, failed or did not help, and have been removed from the code (§9).

---

## 5. Training the replacement unit (hindsight candidates)

Hardware gives hit/miss per access, never the victim. Labels still constrain evictions, **per set** (MUST — only lines and misses of the same set count):

- line `y`'s next access in this set **hits** → `y` was not evicted in between (**protected**);
- next access **misses** → `y` was evicted before it (**must go**, deadline = that access);
- never accessed again, or way empty / pre-stream line → **free**.

At a miss in set `j` (MUST):
0. If set `j` has an empty way, fill the lowest one (§3.2), with no loss. If the labels do not allow that fill, count it as inconsistent.
1. Classify the resident ways of `j` as protected / must / free.
2. Earliest-deadline-first feasibility: with must-lines sorted by deadline `d_1 ≤ d_2 ≤ …`, require `k ≤ #misses in set j strictly between now and d_k` for all k after the choice. Allowed = free ∪ must if feasible; otherwise only the must-lines whose removal makes it feasible.
3. Victim = best-scoring allowed way (+ Gumbel exploration on non-empty ways only, fading 1 → 0 over training; no noise on empty-way fills).
4. Loss: `−log Σ_{w ∈ allowed} softmax(score)[w]`, only when allowed ⊊ all ways; SHOULD weight by label confidence `|2 p_hit − 1|` on hardware data.
5. Inconsistent steps (a label the residency cannot satisfy) MUST be counted and reported; on simulated data they MUST be 0.

Model selection on the validation split's **free-running** accuracy. Independent test set used **once**. Reference implementation for one set: `cpl/train_hindsight.py`.

---

## 6. Validation suite (MUST pass before hardware results count)

**Data (MUST).** Simulated datasets come from `cpl.data.sim_stream_dataset`, which runs the repo's reference simulators (`hardware_security_project/src/cpl/policies/`, tree-PLRU conventions per `simulation_spec.md` §3.3) from an empty set. Each dataset records full ground truth: hit/miss, evicted line, filled way, contents after every access and policy state. Train/val and test are generated with **different seeds** (current convention: train seed 1, test seed 2). Its tests (`tests/test_sim_stream_dataset.py`) check the bookkeeping, and check tree-PLRU against an independent recency-based oracle.
The single-set suite so far uses synthetic streams (pool of 24 congruent lines, W = 16, length 32–96). Once multi-set streams exist (§8.1), the suite SHOULD also be run on the target machine's own measured streams relabelled by the simulators, so the address, set mix and length distributions match the hardware.

**Grading (MUST)** with `cpl.eval_sim` on the independent test set, model free-running:
- hit/miss accuracy, overall and excluding first touches;
- accuracy by reuse distance (0–7, 8–15, 16+);
- exact-stream rate (every access right);
- **contents accuracy**: the predicted set of resident lines equals the true one after each access (way positions ignored);
- contents exact-stream rate.
Baselines are LRU, FIFO and random replacement. On tree-PLRU data, an independent tree-PLRU predictor MUST score 1.0, which checks the grader itself. Hit/miss accuracy alone hides wrong evictions that rarely affect later hits, so contents exact-stream is the headline metric for "does the model track the cache."

**Diagnostics** (`cpl.analyze_sim`) run on the **validation** split only; the tool refuses the test split. It reports same-state decision accuracy by eviction index and, for tree-PLRU, the tree level of each stream's first wrong decision.

**Plain-GRU baseline (MUST):** every suite policy also gets a plain GRU (`cpl.train_hits`) on the same data, to show that the structure is what helps.

| Policy | Purpose | Pass (free-running, independent sim test) |
|---|---|---|
| LRU, FIFO | sanity. Near-trivial with line ages: the answer is the oldest `rec` / `ins` | ≥ 0.99 |
| bit-PLRU | reference-bit family; state not readable from ages | ≥ 0.99 |
| tree-PLRU | position-dependent state | ≥ 0.99 |
| SRRIP (HP/FP) | re-reference prediction; state not readable from ages | ≥ 0.99 |
| DRRIP (set dueling, multi-set) | global state + set identity | ≥ 0.99, and ablating `e(a)` or `g` MUST lose accuracy |
| random replacement | control | reported vs its expected-accuracy ceiling; must not exceed it |

The pass thresholds are on hit/miss accuracy. Contents exact-stream MUST also be reported next to them. Also report accuracy on streams 2× and 4× the training length. The decoder (§3.1) is validated separately.

---

## 7. Per-processor pipeline (MUST be identical across machines)

1. **Bring-up** (script to be written): read `/sys/devices/system/cpu/cpuN/cache/indexK/{level,type,size,ways_of_associativity,number_of_sets,shared_cpu_list}` per core type; choose the pinned core and siblings to offline; write the config (W, S_hw, core); reserve huge pages; set the performance governor; verify and **record all of it in `meta.json`**.
2. **Measure:** decoder pairs (§8.2) and multi-set streams (§8.1), R = 64, a training set and an independent test set with a fresh seed.
3. **Data check:** consistency, label ceiling, hit rate vs reuse distance, fixed-policy baselines.
4. **Validate:** decoder agreement (§3.1) and the policy suite (§6) on this machine's streams.
5. **Train:** plain GRU baseline and this model, fixed hyperparameters.
6. **Report:** independent test once — accuracy, log-loss, base rate, ceiling, errors by reuse distance, GRU vs this model, and the `e(a)`/`g` ablations.

---

## 8. Measurement requirements

### 8.1 Multi-set streams (engine mode to be added)
- Find P congruent pools (one per set), P varying across streams, sets **sampled widely across all S_hw sets** (needed to discover leader sets, if any).
- Each stream interleaves accesses across its pools; ≥ ~W+8 lines per pool so replacement is exercised.
- Suppress prefetch: disable hardware prefetchers if the platform allows, else choose sets whose lines are far apart in memory; record which in `meta.json`.
- Output per access: line address (bits 6..20 at least) and `p_hit` over R repeats.

### 8.2 Decoder pairs
- Positives: address pairs within one engine-found eviction pool; negatives: pairs across pools. SHOULD cover ≥ 512 distinct sets.

---

## 9. Status and evidence (2026-11-24)

The early rows (way identity, used-way broadcast) were measured on relabelled Celeron streams with the older code. Everything from "simulator generator" down was measured on `cpl.data.sim_stream_dataset` data: 100k train streams (5% val, seed 1) and an independent 8k-stream test set (seed 2), W = 16, pool 24, length 32–96. Those rows are free-running on the test set unless they say "val".

| Item | State | Evidence |
|---|---|---|
| Residency + replacement, one set | built (`cpl/model/slot_cache.py`, 30.6k params with ages) | — |
| Way identity | built, required | without it: stuck at random replacement (0.916) |
| Used-way broadcast | built, required | tree-PLRU oracle victim accuracy 0.245 → 0.924 |
| Soft / straight-through victims, `tau`, `empty_bonus` | removed | soft plateaued at 0.93–0.94; straight-through collapsed; annealing destabilised |
| Simulator ground-truth generator + grader | built (`sim_stream_dataset`, `eval_sim`, `analyze_sim`) | tree-PLRU truth check scores 1.0 |
| Plain GRU, tree-PLRU | baseline | acc 0.9407, exact-stream 0.114 |
| Hindsight, no ages, tree-PLRU | superseded | acc 0.9854, contents 0.835, exact-stream 0.637 |
| Capacity A: victim labels, no ages | passed | acc 1.0000, contents 0.9998, exact-stream 0.9995 |
| Capacity B: victim labels + ages | passed | acc 1.0000, contents 0.9999, exact-stream 0.9994 |
| **C: hit/miss only + ages + fill-order naming** (current recipe) | **passed on all 5 single-set policies** | table below |
| Set decoder | not built | — |
| Set identity input `e(a)` | not built | — |
| Multi-set residency/batching | partly (set dimension exists, untested) | — |
| Engine multi-set + pairs modes | not built | — |
| Bring-up script | not built | — |
| Hardware baseline | — | plain GRU 0.9706 on Celeron v2 independent test (R = 64); label ceiling ≈ 0.997 |

**Recipe C on the single-set suite** (15 epochs, lr 1e-3, batch 256, d_way = d_glob = 32, exploration 1 → 0, model seed 1234 unless noted; ≈ 4–6 min/epoch on a laptop CPU):

| Policy | acc | exact-stream | contents acc | contents exact-stream | val: streams with a wrong eviction | best fixed baseline (acc) |
|---|---|---|---|---|---|---|
| tree-PLRU (seed 1234) | 1.0000 | 0.9990 | 0.9996 | 0.9981 | 15 / 4,966 | LRU 0.9385 |
| tree-PLRU (seed 2) | 0.9999 | 0.9966 | 0.9986 | 0.9942 | — | |
| tree-PLRU (seed 3) | 0.9999 | 0.9984 | 0.9992 | 0.9962 | — | |
| LRU | 1.0000 | 0.9999 | 1.0000 | 0.9999 | 1 / 4,966 | FIFO 0.9109 |
| FIFO | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 / 4,966 | LRU 0.9109 |
| bit-PLRU | 1.0000 | 0.9998 | 1.0000 | 0.9995 | 4 / 4,966 | LRU 0.9522 |
| SRRIP | 0.9999 | 0.9952 | 0.9984 | **0.9769** | 114 / 4,966 | LRU 0.9360 |

Inconsistent steps were 0 in every simulated run. SRRIP is the weakest: its validation accuracy was still rising at the last epoch. Plain-GRU baselines for LRU, FIFO, bit-PLRU and SRRIP have not been run yet.

---

## 10. Open problems

1. ~~tree-PLRU under hindsight training~~ **Solved** by fill-order naming plus line ages (recipe C, §9).
2. **SRRIP residual errors.** 2.3% of validation streams have at least one wrong eviction (contents exact-stream 0.977). Try longer training first, since validation was still rising. Then check with `analyze_sim` whether the errors cluster after many evictions, which would point to state the per-way GRU loses.
3. **L1 filtering, and its effect on ages.** Accesses that hit in L1 never reach L2, so L2 replacement state is not updated although a hit is recorded. This also makes the line ages, which are computed from the access sequence, wrong on hardware. If hardware accuracy stalls, add a small L1 layer to the residency tracker (and compute ages from L1 misses only), or make the measurement avoid L1 hits.
4. **Fill-order assumption.** Fill-order naming is exact only if the hardware's fill order for empty ways is fixed (it need not be lowest-first, just fixed). Hardware streams do not start from an empty set, so pre-stream lines behave as "free" residents. Check the hardware inconsistent-step count, which should be near 0 apart from label noise.
5. **Set summary expressiveness.** If a suite policy fails, replace the mean with a learned set-level state.
6. **Decoder generality.** L3 slice hashing will need more address bits and more pairs; the interface (§3.1) is unchanged.
7. **Length generalisation.** Not yet tested on streams longer than in training (§6).
