# Simulation Phase — Implementation Specification

**Project:** Learning and Identifying Cache Replacement Policies with Neural Sequence Models
**Team:** Henry Tudor · Dhwanit Kumawat · Jacob Erattuparambil
**Scope of this document:** the **simulation-only** part of the project. Hardware measurement is out of scope here.
**Audience:** a human or an LLM coding agent implementing this phase. Every definition, convention, interface and acceptance test needed to write correct code is stated explicitly. When this document says **MUST**, the code has to follow it exactly. Where it says **SHOULD**, use the stated default unless there is a clear reason not to.

---

## 0. TL;DR for the implementer

1. Implement a **tree-PLRU cache-set simulator** with the exact bit conventions in §3. It MUST reproduce the test vectors in §3.5.
2. Generate datasets of `(access history → evicted block)` samples following §4 and §5.
3. Train a **GRU classifier** (§6) that reads a tokenized history and outputs a probability for each of the W possible victims.
4. Evaluate with the metrics and acceptance criteria in §7. Target: **≥ 99.9% victim accuracy at W = 16 on in-distribution test data**, with accuracy on **longer-than-training histories** also reported.
5. Run the robustness and interpretability experiments in §8.
6. Follow the repo layout, CLI, config and reproducibility rules in §9 and §10.

Python 3.10+, PyTorch 2.x, NumPy, pandas, pyarrow, PyYAML, pytest, matplotlib. No other dependencies without a reason.

---

## 1. Background and Goal

When a cache set with **W ways** is full and a new block is requested, the **replacement policy** chooses which resident block to evict (the **victim**). The overall project wants to identify the undocumented policies of real Intel caches by training a neural model on observed evictions.

**The simulation phase answers one question: can a GRU learn a known, deterministic replacement policy exactly from (history, victim) examples?** We use **tree-PLRU** because:

- it has hidden state (W−1 tree bits), so the victim cannot be read off simple recency the way it can for LRU;
- Intel uses it in real L1 caches (uops.info reports tree-PLRU for Skylake L1D);
- it disagrees with LRU on about 77% of random histories at W = 16, which gives a strong negative control.

Simulation provides **clean, unlimited, perfectly labeled data**. If the model can't learn PLRU here, it won't learn anything from noisy hardware.

---

## 2. Threat Model (as it applies to simulation)

### 2.1 Security motivation

Cache side-channel attacks (Prime+Probe, Evict+Reload, Reload+Refresh) need the attacker to **force or predict evictions** in a specific cache set. Knowing the replacement policy lets an attacker build minimal eviction sets, order accesses so a target is evicted deterministically, and even leak information through replacement state itself. Assuming the wrong policy makes attacks unreliable and **makes security evaluations inaccurate**. A model that learns the policy from observed behavior removes the need for vendor documentation.

### 2.2 Adversary being modeled

| Property | Assumption |
|---|---|
| Position | Unprivileged local user sharing a cache with other processes |
| Observations | For any access sequence it chooses to run on its own blocks, it can observe **which of its blocks got evicted** (via timing or hit/miss) |
| Knowledge | Cache geometry (W, number of sets). **Not** the replacement policy |
| Goal | Learn a function f(history) → victim that matches the hardware |

### 2.3 What the simulation abstracts away

The simulator plays the role of an **ideal oracle** for that adversary:

- **Single cache set:** we assume the attacker already has W+1 addresses that map to the same set. Building eviction sets is out of scope.
- **Known starting state:** each experiment starts from a fixed reset state, then a fill.
- **Perfect observations:** no timing noise, no prefetchers, no other processes. §8.1 adds synthetic noise *on purpose* to see how robust the model is before we move to hardware.
- **No upper-level caches:** every access in the history reaches the simulated set.

### 2.4 Out of scope

Mounting any attack; attacking other processes; address mapping or slice hashing; hardware measurement code.

---

## 3. Cache Model and Policy Definitions

### 3.1 Terminology and naming conventions (MUST)

- **W:** associativity (number of ways). Supported values: **4, 8, 16** for PLRU, which requires W to be a power of two. Code SHOULD support any W ≥ 2 for non-tree policies.
- **Way:** a physical slot index `0 … W−1`.
- **Block:** an abstract memory block, identified by an integer ID.
- **b0 … b(W−1):** the W *original* blocks, named in **fill order**.
- **bW:** the *new* block whose access triggers the eviction.
- **Fill rule:** when a set has empty (invalid) ways, a miss fills the **lowest-index empty way** ("linear insertion order"). As a result, after filling from empty, block **b_i is in way i**, so *victim block ID equals victim way index*.
- **History:** a sequence `h1 … hL` of block IDs, each in `0 … W−1` (phase 1: all hits).
- **Label:** the block ID evicted by the trigger access to `bW`, in `0 … W−1`.

### 3.2 Common simulator interface (MUST)

```python
class ReplacementPolicy(Protocol):
    W: int
    def reset(self) -> None: ...                    # empty set, policy state set to its reset value
    def access(self, block: int) -> int | None: ... # returns evicted block ID, or None on a hit or on a fill into an empty way
    def resident(self) -> list[int | None]: ...     # block ID in each way (None = empty)
    def state(self) -> np.ndarray: ...              # policy-internal state snapshot (for probing; see §8.2)
    def clone(self) -> "ReplacementPolicy": ...     # deep copy, used for diagnostic search
```

Shared helper:

```python
def run_experiment(policy: ReplacementPolicy, history: list[int]) -> tuple[int, np.ndarray]:
    """reset → fill b0..b(W-1) → apply history → snapshot state → access bW.
    Returns (victim_block_id, state_snapshot_before_trigger)."""
```

`run_experiment` MUST assert that every history access is a hit in phase 1 (all IDs < W), and that the trigger evicts exactly one block.

### 3.3 Tree-PLRU (the target policy) — exact definition

**State:** `bits`, an array of **W−1** bits for a complete binary tree stored in **heap order**: node 0 is the root, and node n has left child `2n+1` and right child `2n+2`. The leaves, from left to right, are ways `0 … W−1`. Each internal node covers a contiguous range of ways `[lo, hi)`, split at `mid = (lo+hi)//2`.

**Bit meaning (MUST):** `bit = 0` means "the victim search goes **left**", and `bit = 1` means "the victim search goes **right**".

**Reset value:** all bits 0, all ways empty.

**Touch(way w)**, called on **every** access to way w (hit, or fill/replacement into w): walk from the root toward w, and set each node's bit to point **away** from w:
- if w is in the left half (`w < mid`): set bit = 1;
- if w is in the right half (`w >= mid`): set bit = 0.

**Victim selection** (when the set is full and a miss happens): start at the root, go left if the bit is 0 and right if it is 1, until you reach a leaf. That leaf's way is evicted, the new block is placed there, and then **Touch** is applied to that way.

**Access(block):**
1. If block is resident in way w: Touch(w); return None (hit).
2. Else if any way is empty: w = lowest empty index; place block; Touch(w); return None.
3. Else: w = victim(); evicted = block in w; place block; Touch(w); return evicted.

Reference implementation (verified to produce §3.5):

```python
class TreePLRU:
    def __init__(self, W):
        assert W >= 2 and (W & (W - 1)) == 0, "tree-PLRU needs power-of-two W"
        self.W = W
        self.reset()
    def reset(self):
        self.bits = [0] * (self.W - 1)
        self.ways = [None] * self.W
    def _touch(self, w):
        node, lo, hi = 0, 0, self.W
        while hi - lo > 1:
            mid = (lo + hi) // 2
            right = w >= mid
            self.bits[node] = 0 if right else 1
            node = 2 * node + (2 if right else 1)
            lo, hi = (mid, hi) if right else (lo, mid)
    def victim_way(self):
        node, lo, hi = 0, 0, self.W
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.bits[node]: node, lo = 2 * node + 2, mid
            else:               node, hi = 2 * node + 1, mid
        return lo
    def access(self, block):
        if block in self.ways:
            self._touch(self.ways.index(block)); return None
        w = self.ways.index(None) if None in self.ways else self.victim_way()
        evicted, self.ways[w] = self.ways[w], block
        self._touch(w)
        return evicted
```

### 3.4 Comparison policies (for controls and diagnostics only; NOT training targets)

These are needed for negative controls (§7.3) and diagnostic histories (§4.3). All of them fill empty ways linearly, as in §3.1.

- **LRU:** evict the block whose most recent access (fill counts as an access) is oldest.
- **FIFO:** evict the block that was inserted earliest. Hits do not change state. *Note:* with hits-only histories after a fill, FIFO **always evicts b0**. Mention this in reports rather than treating it as a finding.
- **SRRIP-HP** (Jaleel et al., ISCA 2010), M = 2 bits: each way has an RRPV in 0…3. Insertion sets RRPV = 2. A hit sets RRPV = 0. Victim: the lowest-index way with RRPV = 3; if there is none, increment every RRPV by 1 and repeat. Fills into empty ways also set RRPV = 2.
- **(Optional) BRRIP:** like SRRIP but inserts with RRPV = 3, except with probability 1/32 it inserts with RRPV = 2. This policy is **stochastic**, so seed its RNG explicitly.

### 3.5 Required test vectors (MUST pass in `tests/test_plru.py`)

All use `run_experiment` (reset, linear fill of b0…b(W−1), history, trigger bW). "bits" means the state immediately before the trigger.

| W | History | PLRU victim | Bits before trigger | LRU victim |
|---|---|---|---|---|
| 4 | `[]` | 0 | `[0,0,0]` | 0 |
| 4 | `[0]` | 2 | `[1,1,0]` | 1 |
| 4 | `[0,1,2,3]` | 0 | `[0,0,0]` | 0 |
| 4 | `[3,2,1,0]` | 3 | `[1,1,1]` | 3 |
| 4 | `[1,1,1]` | 2 | `[1,0,0]` | 0 |
| 4 | `[2,0,3]` | 1 | `[0,1,0]` | 1 |
| 8 | `[]` | 0 | all 0 | 0 |
| 16 | `[]` | 0 | all 0 | 0 |
| 16 | `[0,5,9]` | 2 | — | 1 |
| 16 | `[15,14,13,12,11,10,9,8]` | 0 | — | 0 |
| 16 | `[3,3,3,7,0,12]` | 4 | — | 1 |

### 3.6 Independent oracle (MUST implement as a test)

With hits-only histories after a full fill, the PLRU victim depends **only on the order of each block's most recent access**. Each tree node's bit points away from whichever half was accessed most recently. That gives a second, independent implementation:

```python
def plru_victim_from_recency(W, history):
    last = {b: i for i, b in enumerate(range(W))}          # fill counts as access at time i
    for t, b in enumerate(history): last[b] = W + t
    lo, hi = 0, W
    while hi - lo > 1:
        mid = (lo + hi) // 2
        left_recent  = max(last[b] for b in range(lo, mid))
        right_recent = max(last[b] for b in range(mid, hi))
        if left_recent > right_recent: lo = mid             # left half used more recently → victim on the right
        else:                          hi = mid
    return lo
```

`tests/test_plru.py` MUST check that `TreePLRU` and this oracle agree on **≥ 10,000 random histories** (W ∈ {4, 8, 16}, L ∈ [0, 64]). We have checked that this holds for 20,000 histories at W = 16.

**Implication for the model:** the task doesn't need the full history, only the recency order. Long histories are not harder in principle. That makes the length-generalization test (§7.2) a clean measure of whether the GRU learned the rule rather than memorizing patterns at particular lengths.

---

## 4. Dataset Generation

### 4.1 Sample definition

One sample is:
- `history`: list of int in [0, W), length L
- `victim_counts`: length-16 int vector (padded to `W_MAX = 16`). For deterministic simulation it is **one-hot** at the victim. Under synthetic noise (§8.1) it holds counts over R repeated trials.
- metadata (see §5)

### 4.2 Random histories (default)

- Draw L, then draw each element independently and uniformly from [0, W).
- **Length distributions (SHOULD):**
  - `train` / `val` / `test`: L uniform in **[0, 32]**
  - `test_long`: L uniform in **[33, 128]**
  - `test_xlong`: L uniform in **[129, 512]** (optional, for the length-generalization curve)
- **Duplicates:** for small L the space of possible histories is tiny (W^L), so the same history will be drawn many times. Assign the split **deterministically from a hash of the history** (§4.5) so that a history never appears in two splits. Within a split, duplicates are allowed for L ≤ 3 and SHOULD be removed for larger L.

### 4.3 Diagnostic histories

These are histories where the candidate policies **disagree**. They are the most useful for telling policies apart and will later be used first on hardware.

- Generate random candidates. Keep one only if `PLRU(h) != LRU(h)`. For a stronger version, keep it only if PLRU, LRU and SRRIP all give different victims.
- Tag them `gen_type = "diagnostic"`. The datasets SHOULD include a `test_diag` split of 20k diagnostic histories (L in [0, 32]).
- Implement a `generate_diagnostic(policies, n, W, L_range, rng, require="all_pairwise" | "any")` function. It SHOULD give up with a warning after `max_tries`.

### 4.4 Known class imbalance (MUST be aware of)

Measured over 20k random histories at W = 16, L uniform in [0, 32]:
- the victim distribution is **skewed** toward ways 0, 8, 4, 12 (way 0 ≈ 15%, odd ways ≈ 3.5% each). Short histories leave the tree close to its reset state, which points left. For L = 0 the victim is always 0.
- at L = 64 the distribution is about uniform (≈ 6.25% each).

Consequences:
- Report **per-victim accuracy** and **accuracy for each length bucket**, not just overall accuracy.
- A majority-class baseline scores ≈ 15%. Report it.
- Do **not** reweight classes by default. The skew is a real property of the policy. An optional `--balance-lengths` flag MAY oversample longer histories.

### 4.5 Splits (MUST)

```python
history_id = sha1(f"W{W}:" + ",".join(map(str, history))).hexdigest()[:16]
u = int(history_id, 16) / 16**16            # deterministic value in [0, 1)
split = "train" if u < 0.90 else ("val" if u < 0.95 else "test")
```

`test_long`, `test_xlong` and `test_diag` are generated **separately** with their own seeds. Any history in them whose `history_id` is also in train MUST be dropped.

**Default sizes:**

| Split | Size (W=16) | L range |
|---|---|---|
| train | 1,000,000 | 0–32 |
| val | 50,000 | 0–32 |
| test | 50,000 | 0–32 |
| test_long | 50,000 | 33–128 |
| test_xlong | 20,000 | 129–512 |
| test_diag | 20,000 | 0–32 |

For W = 4 and W = 8, use 200k train and the same eval sizes. Generating 1M samples MUST take under about 2 minutes on a laptop CPU. Use multiprocessing or a vectorized NumPy implementation if the pure-Python simulator is too slow, but keep the reference class in §3.3 as the source of truth and test the fast version against it.

---

## 5. On-Disk Format

This uses the same schema as the hardware phase, with hardware-only fields set to null.

```
data/
  processed/sim_plru_W16/
      samples.parquet
      meta.json            # generator config, seed, code version, counts per split
  histories/
      histories.parquet    # history_id, W, history, hist_len, gen_type
```

**`samples.parquet` columns:**

| Column | Type | Sim value |
|---|---|---|
| `sample_id` | str | `f"{target}_{index:08d}"` |
| `target` | str | `"sim_plru_W16"` |
| `source` | str | `"sim"` |
| `policy` | str | `"plru"` |
| `W` | int8 | 4, 8 or 16 |
| `cache_level`, `set_index`, `slice`, `set_role`, `forced_mode`, `run_id` | null | (hardware fields) |
| `history_id` | str | §4.5 |
| `history` | list<int8> | |
| `hist_len` | int16 | |
| `gen_type` | str | `random` / `diagnostic` |
| `victim_counts` | list<int32>[16] | one-hot (or counts under noise) |
| `n_valid_repeats` | int16 | 1 (or R under noise) |
| `label_hard` | int8 | argmax of `victim_counts` |
| `label_entropy` | float32 | 0.0 for clean data |
| `sim_victims` | struct{plru, lru, fifo, srrip: int8} | every comparison policy's victim for this history |
| `plru_bits` | list<int8>[15] | tree state before the trigger (padded with −1 for W<16) |
| `split` | str | |

---

## 6. Model

### 6.1 Tokenization (MUST)

Fixed vocabulary:

| ID | Token |
|---|---|
| 0 | `PAD` |
| 1 | `FILL` |
| 2 | `SEP` |
| 3 | `NEW` |
| 4 – 35 | `b0 … b31` (block tokens; b16–b31 reserved for phase-2 miss histories) |
| 36 – 47 | reserved (target / role tokens for later phases) |

`VOCAB_SIZE = 48`. Block b_i → token `4 + i`.

**Sequence for one sample:**

```
FILL  b0 b1 … b(W-1)  SEP  h1 h2 … hL  NEW
```

- Length = `W + L + 3`.
- The fill prefix is the same in every simulated sample. It is kept so the format matches the hardware phase.
- Batches are **right-padded** with PAD. Store `new_pos` (the index of `NEW`) for each sample.

### 6.2 Architecture (MUST, with SHOULD defaults)

```
Embedding(VOCAB_SIZE, d_emb=32, padding_idx=0)
→ GRU(input=32, hidden=128, num_layers=2, batch_first=True, dropout=0.0)
→ take hidden output at position new_pos           (torch.gather; NOT the last time step, because of padding)
→ LayerNorm(128) → Linear(128, W_MAX=16)
→ mask logits[:, W:] = -inf
```

- The output layer has `W_MAX = 16` logits, with ways ≥ W masked out. One codebase serves W = 4, 8 and 16 (and 11 later for the LLC).
- About 160k parameters. Do NOT make it larger unless it fails to reach the target accuracy (§7.4).
- The GRU MAY instead use `pack_padded_sequence`. If it does, the gather MUST still select the `NEW` position.
- Dropout 0 for clean data. You MAY use 0.1 for the noise experiments.

### 6.3 Loss

Soft-label cross-entropy:

```python
p_target = victim_counts / victim_counts.sum(-1, keepdim=True)   # one-hot for clean data
loss = -(p_target * log_softmax(masked_logits)).sum(-1).mean()
```

Make sure masked positions (−inf logits with 0 target) don't produce NaN. Use `torch.where` or replace −inf with a large negative number such as −1e9.

### 6.4 Training (SHOULD defaults)

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 1e-3, betas (0.9, 0.999), weight decay 0.01 |
| Schedule | Linear warmup 500 steps → cosine decay to 1e-5 |
| Batch size | 512 |
| Gradient clipping | max-norm 1.0 |
| Epochs | Until val accuracy ≥ 99.9% for 2 consecutive evals, or 20 epochs max |
| Eval frequency | Every 2,000 steps on val; save best checkpoint by val accuracy |
| Curriculum (on by default) | Max history length starts at 8 and increases by 8 every epoch until it reaches 32 |
| Device | `cuda` if available, otherwise `cpu`. MUST run end to end on CPU |
| Mixed precision | Off by default (the model is too small to benefit) |

**Order:** W = 4, then W = 8, then W = 16. Each W is a separate training run.

---

## 7. Evaluation

### 7.1 Metrics (MUST compute for every eval split)

- **Victim accuracy:** `argmax(masked_logits) == label_hard`
- **Accuracy per victim class:** a vector of length W
- **Accuracy per length bucket:** buckets `[0], [1–4], [5–8], [9–16], [17–32], [33–64], [65–128], [129–256], [257–512]`
- **NLL** (mean cross-entropy)
- **Expected calibration error** (15 bins)
- **Confidence:** mean max-probability on correct and on incorrect predictions
- **Training cost:** wall-clock time and the number of training samples seen when val accuracy first reaches 99% and 99.9%

### 7.2 Length generalization (RQ2)

Plot accuracy against history length over 0–512, combining test, test_long and test_xlong. Mark the training maximum length (32) with a vertical line. This is a headline figure.

### 7.3 Controls and baselines (MUST)

| Control | Expected result |
|---|---|
| Majority class (always predict way 0) | ≈ 15% on test |
| Uniform random | ≈ 1/W |
| Rule-based oracle (§3.6) | 100% (sanity check of the labels) |
| **Model vs. LRU simulator** (agreement between model and LRU's victim on test) | ≈ 23% (the measured PLRU/LRU agreement at W=16). If the model's agreement with LRU is much higher than this, something is wrong |
| **Shuffled labels** (train on randomly permuted labels) | Val accuracy close to chance level. Confirms there's no leakage |
| **Policy matching table** | For the PLRU-trained model, % agreement with each of {PLRU, LRU, FIFO, SRRIP} on `test` and `test_diag`. PLRU must clearly come out on top. This is a dry run of the procedure used on hardware |

### 7.4 Acceptance criteria

| Criterion | Threshold |
|---|---|
| W = 4 test accuracy | ≥ 99.9% |
| W = 8 test accuracy | ≥ 99.9% |
| **W = 16 test accuracy** | **≥ 99.9%** |
| W = 16 `test_diag` accuracy | ≥ 99.5% |
| W = 16 `test_long` accuracy | Report it. Target ≥ 99%. Not a hard requirement, but if it's lower, analyze where it fails |
| Policy-matching table | PLRU agreement ≥ 99.5%; every other policy ≤ 50% on `test_diag` |
| Reproducibility | Same seed, same data → test accuracy within ±0.05% |

**If W = 16 doesn't reach 99.9%**, change one thing at a time, in this order: train longer; use hidden size 256; use 3 layers; use lr 3e-4; train on 3M samples. Record each attempt.

---

## 8. Additional Experiments

### 8.1 Synthetic measurement noise (prepares for hardware)

Simulate the imperfect observations an attacker gets on real hardware:

- **Label-flip noise ε:** for each of R repeats, record the true victim with probability 1−ε. Otherwise record a uniformly random *other* way. `victim_counts` holds the R results.
- Grid: ε ∈ {0, 0.01, 0.05, 0.10, 0.20}, R ∈ {1, 5, 10}.
- Train with **soft labels** (counts/R), and also with **hard majority-vote labels**. Always evaluate against the **clean** labels.
- Report clean-label test accuracy against ε for each R and label type. Expectation: soft labels with R ≥ 5 stay close to 99%+ up to ε = 0.1.
- (Optional) **Reset-state noise:** with probability δ, start the experiment with randomized PLRU bits instead of all zeros. This models an imperfect reset sequence on hardware. Note that the fill touches every way, which overwrites many of the bits, so measure how much δ actually matters.

### 8.2 Interpretability: does the GRU contain the PLRU state?

- Take the GRU hidden state (last layer) at `NEW` for every test sample.
- Train **15 independent logistic-regression probes** (one per tree bit) on 80% of test samples, and evaluate on the other 20%. The probe targets are the `plru_bits` column.
- Report probe accuracy per bit and per tree depth (root, depth 1, depth 2, depth 3).
- Control: run the same probes on hidden states from a **randomly initialized**, untrained GRU.
- If the probes reach about 100% while the control stays near the majority-class rate, the model has learned the policy's state. That is a strong result for the report.

### 8.3 Sample efficiency (supports the hardware phase)

Train W = 16 models on {1k, 3k, 10k, 30k, 100k, 300k, 1M} training samples (same eval sets). Plot test accuracy against training-set size. This tells us roughly how many hardware samples we'll need.

### 8.4 (Optional) Multi-policy pretraining

Build `sim_mix_W16` with an equal mix of PLRU, LRU and SRRIP samples, adding a policy token (IDs 36–39) after `FILL`. Train one model on it. This tests whether a single GRU can represent several policies, as preparation for pretraining before fine-tuning on hardware.

---

## 9. Repository Layout

```
cache-policy-learning/
  pyproject.toml / requirements.txt
  configs/
    sim_plru_w4.yaml  sim_plru_w8.yaml  sim_plru_w16.yaml
    noise_grid.yaml
  src/cpl/
    __init__.py
    policies/
      base.py          # Protocol + run_experiment
      plru.py          # TreePLRU (reference) + fast vectorized version (optional)
      lru.py fifo.py srrip.py brrip.py
      oracle.py        # plru_victim_from_recency
    data/
      histories.py     # random + diagnostic generators, history_id, split assignment
      generate.py      # CLI: build samples.parquet + meta.json
      noise.py         # §8.1
      dataset.py       # torch Dataset + collate (tokenize, pad, new_pos)
      tokens.py        # vocab constants (§6.1)
    model/
      gru.py           # VictimGRU
      losses.py        # soft-label CE with masking
    train.py           # CLI training loop
    evaluate.py        # CLI: metrics, per-bucket tables, plots, policy-matching table
    probe.py           # §8.2
    utils/ seed.py logging.py
  tests/
    test_plru.py       # §3.5 vectors + oracle agreement
    test_policies.py   # LRU/FIFO/SRRIP unit tests
    test_histories.py  # split determinism, no overlap between splits
    test_tokens.py     # sequence layout, new_pos correctness
    test_model.py      # forward shapes, masking (prob of ways >= W is exactly 0), loss finite
    test_overfit.py    # model reaches 100% on 256 samples within 300 steps (smoke test)
  data/  runs/  figures/          # gitignored outputs
```

### 9.1 CLI (SHOULD)

```bash
python -m cpl.data.generate --config configs/sim_plru_w16.yaml          # writes data/processed/sim_plru_W16/
python -m cpl.train         --config configs/sim_plru_w16.yaml          # writes runs/<run_name>/
python -m cpl.evaluate      --run runs/<run_name> --splits test test_long test_xlong test_diag
python -m cpl.probe         --run runs/<run_name>
pytest -q
```

### 9.2 Example config (`configs/sim_plru_w16.yaml`)

```yaml
seed: 1234
target: sim_plru_W16
policy: plru
W: 16
data:
  out_dir: data/processed/sim_plru_W16
  sizes: {train: 1000000, val: 50000, test: 50000, test_long: 50000, test_xlong: 20000, test_diag: 20000}
  len_ranges: {train: [0, 32], test_long: [33, 128], test_xlong: [129, 512], test_diag: [0, 32]}
  diag_require: any            # any | all_pairwise
  compare_policies: [lru, fifo, srrip]
model:
  d_emb: 32
  hidden: 128
  layers: 2
  dropout: 0.0
train:
  batch_size: 512
  lr: 1.0e-3
  weight_decay: 0.01
  warmup_steps: 500
  max_epochs: 20
  grad_clip: 1.0
  eval_every: 2000
  curriculum: {start_len: 8, step: 8, max_len: 32}
  early_stop_acc: 0.999
noise: null                    # or {eps: 0.05, repeats: 5, label: soft}
```

### 9.3 Run outputs

`runs/<run_name>/` MUST contain: `config.yaml` (resolved), `metrics.jsonl` (one line per eval), `best.pt`, `final_eval.json`, `per_bucket.csv`, `per_victim.csv`, `policy_match.csv`, and `figures/*.png`.

---

## 10. Engineering Requirements

- **Reproducibility:** one `seed` drives Python `random`, NumPy, and torch (CPU and CUDA). Set `torch.backends.cudnn.deterministic = True`. Use separate seeded `np.random.Generator`s for each split (derived from the master seed plus the split name) so that changing one split's size doesn't change the others.
- **Correctness first:** a vectorized or numba simulator MAY be added for speed, but it MUST be tested against the reference `TreePLRU` on ≥ 100k random histories.
- **Leakage checks:** `generate.py` MUST assert that `history_id` sets don't overlap across splits, and MUST log the size of every split and the victim distribution of each.
- **Type hints and docstrings** on all public functions. Keep the core in pure functions.
- **Logging:** print a concise progress line every eval (step, loss, val accuracy, current max curriculum length). No heavyweight experiment-tracking dependency is required.
- **Performance targets:** data generation for W=16 at default sizes in ≤ 2 min; training to acceptance in ≤ 30 min on a laptop CPU (≤ 10 min on any GPU).
- **No hardware code** in this phase. Leave the `run_id`, `set_index` and similar columns null.

---

## 11. Deliverables Checklist

- [ ] `TreePLRU` + comparison policies; all §3.5 vectors and the §3.6 oracle test pass
- [ ] Datasets for W = 4, 8, 16, with `meta.json` and split-overlap checks
- [ ] Trained GRU for each W, meeting §7.4
- [ ] Accuracy-vs-length figure (0–512) for W = 16
- [ ] Per-victim and per-bucket accuracy tables
- [ ] Policy-matching table (PLRU vs LRU/FIFO/SRRIP) on `test` and `test_diag`
- [ ] Control results (majority class, shuffled labels, LRU agreement)
- [ ] Noise-robustness grid (§8.1)
- [ ] PLRU-bit probe results (§8.2)
- [ ] Sample-efficiency curve (§8.3)
- [ ] Short write-up of findings to feed into the final report

---

## 12. References

- A. Abel, J. Reineke. *nanoBench: A Low-Overhead Tool for Running Microbenchmarks on x86 Systems.* ISPASS 2020. Cache results: https://uops.info/cache.html
- P. Vila, P. Ganty, M. Guarnieri, B. Köpf. *CacheQuery: Learning Replacement Policies from Hardware Caches.* PLDI 2020.
- A. Jaleel, K. Theobald, S. Steely Jr., J. Emer. *High Performance Cache Replacement Using Re-Reference Interval Prediction (RRIP).* ISCA 2010.
- S. Briongos et al. *RELOAD+REFRESH: Abusing Cache Replacement Policies to Perform Stealthy Cache Attacks.* USENIX Security 2020.
- G. Delétang et al. *Neural Networks and the Chomsky Hierarchy.* ICLR 2023.
