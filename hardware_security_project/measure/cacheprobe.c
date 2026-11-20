/*
 * cacheprobe.c  --  L2 eviction-set finder + eviction-labeling engine
 * ------------------------------------------------------------------------
 * Part of the "Learning and Identifying Cache Replacement Policies" project.
 * This is the HARDWARE-measurement counterpart of the NumPy simulator in
 * src/cpl/policies/fast.py: instead of computing the victim from a known
 * policy, it OBSERVES the victim on a real CPU so the GRU can be trained on
 * (access history -> evicted way) samples measured from the machine itself.
 *
 * Scope / assumptions (match report section "Reverse-Engineering Setting"):
 *   - Runs with ordinary user privileges on a machine we own.
 *   - Targets one physical cache SET of a W-way cache (default: the core-
 *     private L2). It never touches another process's memory; every address
 *     comes from one buffer we allocate ourselves.
 *   - Noise-tolerant by repetition: each history is measured R times and the
 *     per-trial victims are aggregated into victim_counts (spec section 4.1,
 *     8.1), exactly the field the trainer consumes.
 *
 * Method (standard, e.g. Vila et al., "Theory and Practice of Finding
 * Eviction Sets", S&P 2019):
 *   1. Allocate a large buffer and treat one address-per-page offset as a
 *      candidate. Candidates sharing a page offset share the low set-index
 *      bits; the finder discovers which ones also collide in the higher bits
 *      purely by timing, so no knowledge of the physical mapping is needed.
 *   2. Reduce a big candidate set to a minimal set that evicts a target
 *      (group-elimination). Those addresses are mutually congruent: they all
 *      map to the same cache set and are our "blocks" b0..b(W-1).
 *   3. For each history: fill the set with b0..b(W-1), replay the accesses,
 *      access one more congruent "new" block to force an eviction, then probe
 *      b0..b(W-1) and report the way that became slow (the victim).
 *
 * Build:
 *   Linux   : gcc -O2 -march=native -o cacheprobe     cacheprobe.c
 *   Windows : gcc -O2 -march=native -o cacheprobe.exe cacheprobe.c   (MSYS2 gcc)
 * The Python driver (cpl.data.hw_measure) builds it automatically if missing.
 * On Linux, run the driver under sudo for SCHED_FIFO + huge pages (best timing).
 *
 * Protocol (kept dead simple so the Python side owns all the bookkeeping):
 *   argv: --core C --ways W --pool P --repeats R --buf-mb MB
 *         --in hist.txt --out res.txt [--threshold T] [--l1-fill N] [--verbose]
 *   hist.txt : line 1 "N" (#histories); then N lines "L i0 i1 ... i(L-1)"
 *              where each ik is a block index in [0, P). Indices >= W are
 *              congruent "miss" blocks (used by RRIP-style histories).
 *   res.txt  : N lines, each R integers = victim way per trial (-1 = no
 *              eviction detected that trial).
 *   Calibration / eviction-set stats are printed to stderr.
 */

#ifdef _WIN32
#include <windows.h>
#else
#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#endif
#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----- tunables (overridable via argv) ----------------------------------- */
static size_t PAGE       = 4096;   /* candidates one-per-page => shared offset */
static int    ACCESS_REPS = 4;     /* traversals per congruence test           */
static int    CAL_SAMPLES = 2048;  /* samples per calibration bucket           */
static int    VOTES       = 5;     /* majority-vote repeats per eviction test  */
static int    REDUCE_TRIES = 12;   /* reshuffle+retry attempts to reach W      */

/* ----- timing primitives ------------------------------------------------- */
static inline void mem_load(volatile char *p) { (void)*p; }

static inline uint64_t time_load(volatile char *p) {
    unsigned aux;
    _mm_mfence(); _mm_lfence();
    uint64_t t0 = __rdtscp(&aux); _mm_lfence();
    (void)*p;
    _mm_lfence();
    uint64_t t1 = __rdtscp(&aux); _mm_lfence();
    return t1 - t0;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
static uint64_t pct_u64(uint64_t *v, int n, double p) { /* v must be sorted */
    int i = (int)(p * (n - 1) + 0.5);
    if (i < 0) i = 0;
    if (i >= n) i = n - 1;
    return v[i];
}

/* ----- globals ----------------------------------------------------------- */
static char   *g_buf   = NULL;
static size_t  g_bufsz = 0;
static uint64_t g_thr  = 0;   /* slow/fast threshold in cycles */
static int      g_verbose = 0;
static double   g_min_detect = 0.60;  /* required victim-detection rate         */
static int      g_force = 0;          /* accept best-effort pool below threshold */
static int      g_sim = 0;            /* self-test: software tree-PLRU, no measure*/
static int      g_miss = 0;           /* miss mode: report the trigger's evicted BLOCK id */
static int      g_flush = 1;          /* flush the set at each trial start        */
static int      g_l1_want = 0;        /* desired size of the L1-flush set          */

/* L1-flush set: addresses at the TARGET's page offset (so they land in the same
 * L1 set as every block) but NOT congruent in L2 (different L2 set). Accessing
 * them between the trigger and the probe evicts the blocks from the small L1
 * without disturbing their L2 state, so the probe reads true L2 residency.
 * Built in find_congruent_pool. Used only if --l1-fill > 0. */
static char   **g_l1fill   = NULL;
static int      g_l1fill_n = 0;

/* ----- affinity / priority ----------------------------------------------- */
static void pin_and_boost(int core) {
#ifdef _WIN32
    DWORD_PTR mask = (DWORD_PTR)1 << core;
    if (!SetThreadAffinityMask(GetCurrentThread(), mask))
        fprintf(stderr, "[warn] SetThreadAffinityMask(core=%d) failed\n", core);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    Sleep(1); /* let the scheduler settle on the pinned core */
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        fprintf(stderr, "[warn] sched_setaffinity(core=%d) failed\n", core);
    struct sched_param sp;
    sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
        setpriority(PRIO_PROCESS, 0, -20);      /* best effort without root */
        fprintf(stderr, "[warn] SCHED_FIFO denied (run with sudo for the most "
                        "stable timing); falling back to nice -20\n");
    }
    usleep(1000); /* let the scheduler settle on the pinned core */
#endif
}

/* ----- buffer ------------------------------------------------------------ */
static int alloc_buffer(size_t bytes) {
#ifdef _WIN32
    g_buf = (char *)VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!g_buf) { fprintf(stderr, "[err] VirtualAlloc(%zu) failed\n", bytes); return 0; }
#else
    /* Prefer 2 MiB huge pages: contiguous physical memory means candidates at
     * the same page offset also share the higher set-index bits, so a full
     * eviction set is much easier to find. Fall back to 4 KiB pages. */
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p == MAP_FAILED) {
        p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { fprintf(stderr, "[err] mmap(%zu) failed\n", bytes); return 0; }
        fprintf(stderr, "[mem] 4 KiB pages (no huge pages; "
                        "`sudo sysctl vm.nr_hugepages=256` makes the finder more reliable)\n");
    } else {
        fprintf(stderr, "[mem] 2 MiB huge pages\n");
    }
    g_buf = (char *)p;
#endif
    g_bufsz = bytes;
    for (size_t i = 0; i < bytes; i += PAGE) g_buf[i] = (char)i; /* fault in */
    return 1;
}

/* Candidate address for page index i at a fixed within-page offset. */
static char *candidate(size_t i, size_t offset) { return g_buf + i * PAGE + offset; }

/* ----- congruence tests -------------------------------------------------- */
/* One prime/probe: bring `target` in, traverse `set` (forward then backward to
 * blunt the stride prefetcher), then time `target`. Slow => the set evicted it. */
static int evicts_once(char **set, int n, char *target) {
    mem_load(target);
    _mm_mfence();
    for (int r = 0; r < ACCESS_REPS; r++) {
        if (r & 1) for (int i = n - 1; i >= 0; i--) mem_load(set[i]);
        else       for (int i = 0; i < n; i++)      mem_load(set[i]);
    }
    _mm_mfence();
    return time_load(target) > g_thr;
}

/* Majority vote over VOTES primes: robust on a noisy machine. */
static int evicts(char **set, int n, char *target) {
    int hits = 0;
    for (int v = 0; v < VOTES; v++) hits += evicts_once(set, n, target);
    return 2 * hits > VOTES;
}

/* ----- calibration ------------------------------------------------------- */
/* Measure resident-hit vs flushed-miss latency and place the threshold just
 * above the hit distribution so an L2->L3 eviction still reads as "slow". */
static void calibrate(char **pool, int pool_n) {
    uint64_t *hit  = malloc(sizeof(uint64_t) * CAL_SAMPLES);
    uint64_t *miss = malloc(sizeof(uint64_t) * CAL_SAMPLES);
    char *p = pool_n ? pool[0] : g_buf;

    for (int i = 0; i < CAL_SAMPLES; i++) { mem_load(p); hit[i] = time_load(p); }
    for (int i = 0; i < CAL_SAMPLES; i++) {
        _mm_clflush(p); _mm_mfence();
        miss[i] = time_load(p);
    }
    qsort(hit,  CAL_SAMPLES, sizeof(uint64_t), cmp_u64);
    qsort(miss, CAL_SAMPLES, sizeof(uint64_t), cmp_u64);
    uint64_t hit_p50  = pct_u64(hit,  CAL_SAMPLES, 0.50);
    uint64_t hit_p90  = pct_u64(hit,  CAL_SAMPLES, 0.90);
    uint64_t miss_p10 = pct_u64(miss, CAL_SAMPLES, 0.10);
    uint64_t miss_p50 = pct_u64(miss, CAL_SAMPLES, 0.50);

    if (g_thr == 0) { /* auto unless --threshold given */
        uint64_t mid = (hit_p90 + miss_p10) / 2;
        uint64_t floor = hit_p50 + 25;          /* catch L2->L3 (~+30-50cyc) */
        g_thr = mid > floor ? mid : floor;
    }
    fprintf(stderr,
        "[calib] hit p50=%llu p90=%llu | miss p10=%llu p50=%llu | threshold=%llu\n",
        (unsigned long long)hit_p50, (unsigned long long)hit_p90,
        (unsigned long long)miss_p10, (unsigned long long)miss_p50,
        (unsigned long long)g_thr);
    if (miss_p10 <= hit_p90 + 10)
        fprintf(stderr, "[warn] hit/miss distributions overlap; expect noisy "
                        "labels. Try pinning to an idle core or raising --repeats.\n");
    free(hit); free(miss);
}

/* ----- eviction-set finder ---------------------------------------------- */
/* Group-elimination reduction (Vila et al. baseline). Reduce candidate set C
 * (in place) to a minimal subset that still evicts `target`; returns its size.
 * On a noisy round (no single group is safely removable) it stops early, so
 * the caller checks for == W and retries with fresh candidates if not. */
static int reduce_once(char **C, int n, char *target, int W) {
    while (n > W) {
        int group = n / (W + 1); if (group < 1) group = 1;
        int removed = 0;
        for (int g = 0; g <= W && g * group < n; g++) {
            int lo = g * group, hi = lo + group; if (hi > n) hi = n;
            int m = 0;
            static char *tmp[1 << 16];
            for (int i = 0; i < n; i++) if (i < lo || i >= hi) tmp[m++] = C[i];
            if (evicts(tmp, m, target)) {      /* subset still evicts: drop group */
                memcpy(C, tmp, sizeof(char *) * m);
                n = m; removed = 1; break;
            }
        }
        if (!removed) break;                   /* stuck: let caller retry */
    }
    return n;
}

static void shuffle(int *a, int n) {
    for (int i = n - 1; i > 0; i--) { int j = rand() % (i + 1); int t = a[i]; a[i] = a[j]; a[j] = t; }
}

/* forward decl: one measured fill+trigger+probe trial */
static int run_trial(char **blk, int W, int *hist, int L, char *trigger);

/* Functionally validate a candidate pool: the thing we actually need is that
 * filling W blocks and accessing the trigger evicts exactly one *detectable*
 * block most of the time. Returns the fraction of trials with a clean victim. */
static double validate_pool(char **pool, int W, char *trigger) {
    int detected = 0, N = 128;
    for (int t = 0; t < N; t++) detected += (run_trial(pool, W, NULL, 0, trigger) >= 0);
    return (double)detected / N;
}

/* Build a pool of `need` mutually-congruent addresses (blocks + trigger).
 * Finds a minimal W-way eviction set for `target`, verifies it, then extends
 * it with further addresses confirmed congruent to that set. */
static int find_congruent_pool(char **out, int need, int W, size_t offset) {
    size_t npages = g_bufsz / PAGE;
    int cand_n = (int)(npages > 4096 ? 4096 : npages);  /* candidates for finder */
    char *target = candidate(0, offset);                /* page 0 reserved       */
    int *idx = malloc(sizeof(int) * cand_n);
    char **C = malloc(sizeof(char *) * cand_n);

    /* An evicting set reduced from thousands of candidates is "mostly
     * congruent"; with a non-LRU insertion policy its minimal size can exceed
     * W, so we accept any reliably-evicting set in [W, es_cap] and then confirm
     * the pool works by measuring real evictions. */
    int es_cap = 4 * W + 8;
    int k = 0;
    double best_det = -1.0;
    static char *best_pool[1 << 10];
    int best_k = 0;
    for (int attempt = 0; attempt < REDUCE_TRIES; attempt++) {
        for (int i = 0; i < cand_n; i++) idx[i] = i + 1; /* candidate page ids   */
        shuffle(idx, cand_n);
        for (int i = 0; i < cand_n; i++) C[i] = candidate((size_t)idx[i], offset);
        if (!evicts(C, cand_n, target)) {
            fprintf(stderr, "[err] candidate set does not evict target; raise --buf-mb\n");
            free(idx); free(C); return 0;
        }
        int es_n = reduce_once(C, cand_n, target, W);
        if (es_n > es_cap || !evicts(C, es_n, target)) {
            fprintf(stderr, "[evset] attempt %d: size=%d not usable; retrying\n", attempt + 1, es_n);
            continue;
        }
        /* build pool = target + eviction set, extended with more congruent pages */
        k = 0;
        out[k++] = target;
        for (int i = 0; i < es_n && k < need; i++) out[k++] = C[i];
        for (int i = 0; i < cand_n && k < need; i++) {
            char *c = candidate((size_t)(i + 1), offset);
            int dup = 0;
            for (int j = 0; j < k; j++) if (out[j] == c) { dup = 1; break; }
            if (!dup && evicts(C, es_n, c)) out[k++] = c;
        }
        if (k < need) {
            fprintf(stderr, "[evset] attempt %d: only %d/%d congruent; retrying\n", attempt + 1, k, need);
            continue;
        }
        if (g_l1_want > 0) {   /* same L1 set as the blocks, but a different L2 set */
            free(g_l1fill);
            g_l1fill = malloc(sizeof(char *) * g_l1_want);
            int m = 0;
            for (int i = 0; i < cand_n && m < g_l1_want; i++) {
                char *c = candidate((size_t)(i + 1), offset);
                int inpool = 0;
                for (int j = 0; j < k; j++) if (out[j] == c) { inpool = 1; break; }
                if (!inpool && !evicts(C, es_n, c)) g_l1fill[m++] = c;
            }
            g_l1fill_n = m;
            if (attempt == 0) fprintf(stderr, "[evset] L1-flush set: %d addresses\n", m);
        }
        double det = validate_pool(out, W, out[W]);     /* use out[W] as a trigger */
        fprintf(stderr, "[evset] attempt %d: es=%d pool=%d victim-detect=%.2f\n",
                attempt + 1, es_n, k, det);
        if (det > best_det && k <= (1 << 10)) {          /* remember best so far */
            best_det = det; best_k = k;
            for (int i = 0; i < k; i++) best_pool[i] = out[i];
        }
        if (det >= g_min_detect) {
            fprintf(stderr, "[evset] usable eviction pool found (detect=%.2f)\n", det);
            free(idx); free(C);
            return k;
        }
    }
    if (g_force && best_k >= need) {
        for (int i = 0; i < best_k; i++) out[i] = best_pool[i];
        fprintf(stderr, "[evset] --force: using best-effort pool (detect=%.2f); "
                        "expect noisy labels\n", best_det);
        free(idx); free(C);
        return best_k;
    }
    fprintf(stderr, "[err] best victim-detect=%.2f < --min-detect=%.2f after %d tries. "
                    "Check --ways matches the real associativity, pin to an idle core, "
                    "raise --buf-mb, or pass --force to proceed anyway.\n",
                    best_det, g_min_detect, REDUCE_TRIES);
    free(idx); free(C);
    return 0;
}

/* ----- software tree-PLRU (self-test oracle) ----------------------------- */
/* Mirrors cpl.policies.fast.simulate_batch exactly: linear fill b0..b(W-1)
 * (block id == way index), then the history (hits), then report the way the
 * tree would evict. Used only in --sim mode to validate the whole dataset
 * pipeline without touching hardware; a --sim dataset must satisfy
 * agreement_with_label[plru] == 1.0. */
static int plru_victim(int W, const int *hist, int L) {
    int k = 0; while ((1 << k) < W) k++;      /* W = 2^k */
    int bits[64] = {0};                        /* W-1 internal nodes */
    for (int step = 0; step < W + L; step++) {
        int w = step < W ? step : hist[step - W];
        for (int d = 0; d < k; d++) {
            int node = ((1 << d) - 1) + (w >> (k - d));
            int go_right = (w >> (k - 1 - d)) & 1;
            bits[node] = 1 - go_right;
        }
    }
    int node = 0;
    for (int d = 0; d < k; d++) node = 2 * node + 1 + bits[node];
    return node - (W - 1);
}

/* ----- one measured trial ------------------------------------------------ */
/* One trial, mirroring the simulator's reset -> fill -> history -> trigger:
 *   - flush the set first (g_flush) so each trial starts from an EMPTY set and
 *     the fill deterministically loads b0..b(W-1); this is what makes repeats
 *     of the same history agree. Safe on parts with no L3 (e.g. N3450): a flush
 *     goes to DRAM and the refill is clean.
 *   - after the trigger, drain L1 (g_l1fill) so the probe reads L2, not L1.
 *   - the victim is the FIRST way over threshold: every way probed before it
 *     was a hit (resident), so returning immediately avoids the reload of the
 *     victim cascading into later ways.
 * Returns the way index, or -1 if nothing was evicted. */
static int run_trial(char **blk, int W, int *hist, int L, char *trigger) {
    if (g_sim) return plru_victim(W, hist, L);             /* self-test oracle */
    if (g_flush) {
        for (int i = 0; i < W; i++) _mm_clflush(blk[i]);
        _mm_clflush(trigger);
        _mm_mfence();
    }
    for (int i = 0; i < W; i++) mem_load(blk[i]);          /* fill the set    */
    _mm_mfence();
    for (int i = 0; i < L; i++) mem_load(blk[hist[i]]);    /* replay history  */
    _mm_mfence();
    mem_load(trigger);                                     /* force eviction  */
    _mm_mfence();
    for (int i = 0; i < g_l1fill_n; i++) mem_load(g_l1fill[i]); /* drain L1    */
    _mm_mfence();
    for (int i = 0; i < W; i++)                            /* first miss = victim */
        if (time_load(blk[i]) > g_thr) return i;
    return -1;
}

/* ----- miss-mode trial (distinguishes LRU/SRRIP etc.) -------------------- */
/* With misses in the history the resident set is no longer {b0..b(W-1)}, so a
 * single probe can't isolate the trigger's victim (several blocks may already
 * be gone). Instead we measure the trigger's evicted BLOCK directly: replay the
 * sequence and probe one block in isolation (no probe-cascade), once WITHOUT the
 * trigger and once WITH it. The victim is the block resident before the trigger
 * but gone after. `blk` has `pool` blocks (b0..b(pool-1)); fill is still b0..b(W-1). */
static int seq_resident(char **blk, int W, int pool, int *hist, int L,
                        char *trigger, int use_trigger, char *probe) {
    for (int i = 0; i < pool; i++) _mm_clflush(blk[i]);
    _mm_clflush(trigger);
    _mm_mfence();
    for (int i = 0; i < W; i++) mem_load(blk[i]);          /* fill b0..b(W-1)  */
    _mm_mfence();
    for (int i = 0; i < L; i++) mem_load(blk[hist[i]]);    /* replay (may miss)*/
    _mm_mfence();
    if (use_trigger) { mem_load(trigger); _mm_mfence(); }
    for (int i = 0; i < g_l1fill_n; i++) mem_load(g_l1fill[i]);
    _mm_mfence();
    return time_load(probe) <= g_thr;                      /* 1 = resident (fast) */
}

static int run_miss_trial(char **blk, int W, int pool, int *hist, int L, char *trigger) {
    for (int c = 0; c < pool; c++) {
        if (!seq_resident(blk, W, pool, hist, L, trigger, 0, blk[c])) continue; /* not resident pre-trigger */
        if (!seq_resident(blk, W, pool, hist, L, trigger, 1, blk[c])) return c; /* evicted by the trigger   */
    }
    return -1;
}

/* ----- main -------------------------------------------------------------- */
int main(int argc, char **argv) {
    int core = 0, W = 16, pool = 16, R = 16;
    size_t buf_mb = 256;
    const char *in_path = NULL, *out_path = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--core")      && i+1 < argc) core    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ways")      && i+1 < argc) W       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pool")      && i+1 < argc) pool    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeats")   && i+1 < argc) R       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--buf-mb")    && i+1 < argc) buf_mb  = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) g_thr   = (uint64_t)strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--l1-fill")   && i+1 < argc) g_l1_want = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-detect")&& i+1 < argc) g_min_detect = atof(argv[++i]);
        else if (!strcmp(argv[i], "--force"))                   g_force = 1;
        else if (!strcmp(argv[i], "--sim"))                     g_sim = 1;
        else if (!strcmp(argv[i], "--miss"))                    g_miss = 1;
        else if (!strcmp(argv[i], "--no-flush"))                g_flush = 0;
        else if (!strcmp(argv[i], "--in")        && i+1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out")       && i+1 < argc) out_path= argv[++i];
        else if (!strcmp(argv[i], "--verbose"))                 g_verbose = 1;
        else { fprintf(stderr, "[err] bad arg: %s\n", argv[i]); return 2; }
    }
    if (!in_path || !out_path) { fprintf(stderr, "[err] need --in and --out\n"); return 2; }
    if (pool < W) pool = W;

    srand(0xC0FFEE);
    pin_and_boost(core);
    if (!alloc_buffer(buf_mb << 20)) return 1;

    /* within-page offset for the target set: cache-line aligned, mid-page */
    size_t offset = ((PAGE / 2) / 64) * 64;

    int need = pool + 1;                 /* blocks[0..pool-1] + one trigger */
    char **poolv = malloc(sizeof(char *) * need);
    if (!g_sim) {
        calibrate(poolv, 0);             /* initial threshold from raw buffer */
        if (find_congruent_pool(poolv, need, W, offset) < need) return 1;
        calibrate(poolv, need);          /* refine threshold on real pool */
    } else {
        for (int i = 0; i < need; i++) poolv[i] = candidate((size_t)i, offset);
        fprintf(stderr, "[sim] software tree-PLRU oracle (no hardware measurement)\n");
    }
    char *trigger = poolv[pool];         /* a congruent block outside [0,pool) */
    /* g_l1fill was built inside find_congruent_pool (same L1 set, other L2 set). */

    FILE *fin = fopen(in_path, "r");
    if (!fin) { fprintf(stderr, "[err] cannot open %s\n", in_path); return 1; }
    FILE *fout = fopen(out_path, "w");
    if (!fout) { fprintf(stderr, "[err] cannot open %s\n", out_path); return 1; }

    int N = 0;
    if (fscanf(fin, "%d", &N) != 1) { fprintf(stderr, "[err] bad header\n"); return 1; }
    int *hist = malloc(sizeof(int) * 4096);

    for (int s = 0; s < N; s++) {
        int L = 0;
        if (fscanf(fin, "%d", &L) != 1) { fprintf(stderr, "[err] bad L at %d\n", s); return 1; }
        if (L > 4096) L = 4096;
        for (int i = 0; i < L; i++) {
            if (fscanf(fin, "%d", &hist[i]) != 1) { fprintf(stderr, "[err] bad hist\n"); return 1; }
            if (hist[i] < 0 || hist[i] >= pool) hist[i] = 0;
        }
        for (int r = 0; r < R; r++) {
            int v = g_miss ? run_miss_trial(poolv, W, pool, hist, L, trigger)
                           : run_trial(poolv, W, hist, L, trigger);
            fprintf(fout, "%d%c", v, r + 1 < R ? ' ' : '\n');
        }
        if (g_verbose && (s % 2000 == 0))
            fprintf(stderr, "[run] %d/%d histories\n", s, N);
    }

    fclose(fin); fclose(fout);
    fprintf(stderr, "[done] wrote %d histories x %d repeats to %s\n", N, R, out_path);
#ifdef _WIN32
    VirtualFree(g_buf, 0, MEM_RELEASE);
#else
    munmap(g_buf, g_bufsz);
#endif
    return 0;
}
