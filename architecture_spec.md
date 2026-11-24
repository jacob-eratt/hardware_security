# Whole-Cache Model — Architecture Specification

**Project:** Learning and Identifying Cache Replacement Policies with Neural Sequence Models
**Team:** Henry Tudor · Dhwanit Kumawat · Jacob Erattuparambil
**Scope:** the structured model that predicts per-access hit/miss for a whole cache (many sets) from measured hardware streams, how it is trained without victim labels, how it is validated, and the per-processor pipeline it runs in. It supersedes the plain GRU (`cpl.model.hit_gru`) as the main model; the plain GRU stays as the baseline.
**Audience:** a human or an LLM coding agent implementing or extending this model. **MUST** = required; **SHOULD** = default unless there is a clear reason not to.
**Status (2026-11-24):** see §9. Parts 2 and 3 exist for a single set; part 1, multi-set data and the validation suite do not.

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
- **Update:** on a fill of way `w` with line `x`, `R[b, j, w, :] ← onehot(x)` (hard modes) or the convex mix used by soft mode. Only slot `j` changes.
- Under hard victims (free-running and hindsight training), `R` is one-hot and `p_hit ∈ {0, 1}`.

### 3.3 Unit 3 — replacement unit (learned)

**State (MUST):**
- per way: `h[b, j, w] ∈ R^{d_way}` (SHOULD `d_way = 32`), initialised from a learned per-way vector `h0[w]`;
- global: `g[b] ∈ R^{d_glob}` (SHOULD `d_glob = 32`), initialised to 0.

**Inputs that MUST be present** (each was found necessary experimentally, §9):
- **way identity** `pos[w] ∈ R^{8}` — without it all ways stay identical and the model degenerates to random replacement;
- **used-way broadcast** `u = Σ_w (e_hit[w] + e_fill[w]) · pos[w]` — the identity of the way just hit or filled, given to every way; without it tree-PLRU is unrepresentable (the set summary is a mean, which forgets which way was used);
- **set summary** `m = mean_w h[b, j, w]` (SHOULD be upgraded to a learned set-level vector if the validation suite finds a policy the mean cannot express);
- **set identity** `e(a)` from the decoder — lets the model distinguish e.g. set-dueling leader sets from followers; MUST be fed to the scorer, the way cell and the global cell;
- **global state** `g`.

**Victim scorer (MUST):** for each way of the accessed set,
`score[w] = MLP([h[w], pos[w], m, g, e(a), empty[w]])`, MLP 64 hidden, ReLU. Victim = `argmax_w score[w]` (hard) or `softmax(score)` (soft mode only).

**Way update (MUST):** one shared `GRUCell` over all W ways of the accessed set:
`h[w] ← GRUCell([e_hit[w], e_fill[w], empty[w], pos[w], u, m, g, e(a)], h[w])`,
where `e_hit[w]` = 1 if this way held the hit line, `e_fill[w]` = 1 if this way was filled. Ways of other sets are not updated.

**Global update (MUST):** `g ← GRUCell_g([mean_w h_new[w], hit, e(a)], g)` on every access.

**Weight sharing (MUST):** the way cell, scorer and global cell are the same for every way and every set. The only per-way parameters are `pos` and `h0`; there are no per-set parameters (set differences enter only through `e(a)`).

### 3.4 Batching and memory

Streams touch few sets, so states are kept per **local set slot**, not for all `S_hw` sets: before a batch, each stream's accesses are decoded and the distinct sets numbered 0..k−1 (k ≤ `S_max`, SHOULD 32). Tensors are `[B, S_max, W, ·]`; each step gathers slot `j`, updates it, and writes it back with a one-hot mask. The set *identity* that distinguishes sets across streams is `e(a)`, never the slot number.

### 3.5 One access, step by step

1. `j, e = decoder(a)` → local slot, set embedding.
2. `p_hit = Σ_w R[j, w, x]` — model output for this access.
3. Hit: `e_hit = onehot(way holding x)`. Miss: compute `score`, pick victim `v`, `e_fill = onehot(v)`, write `x` into `R[j, v]`.
4. Update `h[j, ·]` with the shared way cell, then `g` with the global cell.

---

## 4. Modes of the forward pass

| Mode | Victim | Used for |
|---|---|---|
| **free-running** | `argmax score` | all evaluation and reported numbers (MUST) |
| **hindsight** | best-scoring *allowed* way (§5) | training on hardware and simulated data |
| **oracle** | true victim supplied | diagnostics on simulated data only; MUST NOT be used for reported results |
| soft | `softmax(score)` | legacy; plateaus at 0.93–0.94 on simulation; SHOULD NOT be used |

The current code also contains straight-through victims, a temperature `tau` and an `empty_bonus`; all three failed or did not help (§9) and SHOULD be removed once this spec is implemented.

---

## 5. Training the replacement unit (hindsight candidates)

Hardware gives hit/miss per access, never the victim. Labels still constrain evictions, **per set** (MUST — only lines and misses of the same set count):

- line `y`'s next access in this set **hits** → `y` was not evicted in between (**protected**);
- next access **misses** → `y` was evicted before it (**must go**, deadline = that access);
- never accessed again, or way empty / pre-stream line → **free**.

At a miss in set `j` (MUST):
1. Classify the resident ways of `j` as protected / must / free.
2. Earliest-deadline-first feasibility: with must-lines sorted by deadline `d_1 ≤ d_2 ≤ …`, require `k ≤ #misses in set j strictly between now and d_k` for all k after the choice. Allowed = free ∪ must if feasible; otherwise only the must-lines whose removal makes it feasible.
3. Victim = best-scoring allowed way (+ Gumbel exploration on non-empty ways only, fading 1 → 0 over training; no noise on empty-way fills).
4. Loss: `−log Σ_{w ∈ allowed} softmax(score)[w]`, only when allowed ⊊ all ways; SHOULD weight by label confidence `|2 p_hit − 1|` on hardware data.
5. Inconsistent steps (a label the residency cannot satisfy) MUST be counted and reported; on simulated data they MUST be 0.

Model selection on the validation split's **free-running** accuracy. Independent test set used **once**. Reference implementation for one set: `cpl/train_hindsight.py`.

---

## 6. Validation suite (MUST pass before hardware results count)

Simulated datasets are made by relabelling the target machine's own measured access streams (same addresses, set mix and lengths) with a reference simulator, using that machine's `W` and set mapping. Simulators MUST be the repo's reference implementations (`hardware_security_project/src/cpl/policies/`, tree-PLRU conventions per `simulation_spec.md` §3.3).

| Policy | Purpose | Pass (free-running, independent sim test) |
|---|---|---|
| LRU, FIFO | sanity | ≥ 0.99 |
| bit-PLRU | reference-bit family | ≥ 0.99 |
| tree-PLRU | position-dependent state | ≥ 0.99 |
| SRRIP (HP/FP) | re-reference prediction | ≥ 0.99 |
| DRRIP (set dueling, multi-set) | global state + set identity | ≥ 0.99, and ablating `e(a)` or `g` MUST lose accuracy |
| random replacement | control | reported vs its expected-accuracy ceiling; must not exceed it |

Also report accuracy on streams 2× and 4× the training length. The decoder (§3.1) is validated separately.

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

| Item | State | Evidence (simulated relabelled Celeron streams unless noted) |
|---|---|---|
| Residency + replacement, one set | built (`cpl/model/slot_cache.py`) | — |
| Way identity | built, required | without it: stuck at random replacement (0.916) |
| Used-way broadcast | built, required | tree-PLRU oracle victim accuracy 0.245 → 0.924 |
| Oracle representability | passed | bit-PLRU 0.997, tree-PLRU 0.980 (free-running) |
| Hindsight training, one set | built (`cpl/train_hindsight.py`) | bit-PLRU ~0.995; **tree-PLRU stuck at ~0.936** |
| Set decoder | not built | — |
| Set identity input `e(a)` | not built | — |
| Multi-set residency/batching | partly (set dimension exists) | — |
| Engine multi-set + pairs modes | not built | — |
| Bring-up script, validation suite | not built | — |
| Hardware baseline | — | plain GRU 0.9706 on Celeron v2 independent test; label ceiling ≈ 0.997 |

---

## 10. Open problems

1. **tree-PLRU under hindsight training** (blocker for §6). The model represents it given true victims but does not learn it from hit/miss alone; candidate sets are as tight as for bit-PLRU (mean 9.3 of 16 allowed), and removing exploration on empty-way fills did not help. Leading hypothesis: victim choice and line layout are mutually dependent, so training settles on a self-consistent wrong policy. Next idea: curriculum from short streams.
2. **L1 filtering.** Accesses that hit in L1 never reach L2, so L2 replacement state is not updated although a hit is recorded. If hardware accuracy stalls, add a small L1 layer to the residency tracker.
3. **Set summary expressiveness.** If a suite policy fails, replace the mean with a learned set-level state.
4. **Decoder generality.** L3 slice hashing will need more address bits and more pairs; the interface (§3.1) is unchanged.
