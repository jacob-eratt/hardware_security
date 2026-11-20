/*
 * l3probe.c -- L3 (LLC) eviction-set finder + per-access hit/miss engine
 * ------------------------------------------------------------------------
 * Target: Intel Core i7-13700H (Raptor Lake-H). Companion of cacheprobe.c (L2).
 * Only touches memory from one buffer this process allocates.
 *
 * Runs as a SERVER: Python (cpl.data.l3_pipeline) writes one command per line on
 * stdin and gets one JSON line back on stdout. Eviction sets are only valid for
 * the lifetime of this allocation, so every phase (calibration, eviction sets,
 * checks, collection) happens inside a single engine process. Bulk data moves
 * through files (uint16 little-endian raw TSC cycle counts, clipped at 65535).
 *
 * Address layout (2 MiB pages): within a page, bits 0..20 of the physical
 * address equal the virtual offset. With 64 B lines and 4096 sets per L3 slice
 * [VERIFY: 24 MiB / 12 ways / 64 B = 32768 sets; 8 slices assumed], the
 * per-slice set index is bits 6..17 and the P-core L2 set index (1.25 MiB,
 * 10-way, 2048 sets) is bits 6..16. Both are known from the offset. The slice
 * hash uses higher physical bits and is NOT modelled: same-index candidates are
 * congruent with probability ~1/slices, and group testing finds which ones.
 *
 *   candidate(page p, h in 0..7, set s) = buf + p*2MiB + (h<<18) + (s<<6)
 *   L2 pusher for set s                  = same with set s ^ 0x800 (bit 17 flipped):
 *                                          same L2 set, different L3 set.
 *
 * Build:  gcc -O2 -march=native -o l3probe.exe l3probe.c       (MSYS2 / Linux)
 *
 * Commands (each reply is one JSON object):
 *   info
 *   setthr <l2l3> <l3dram>            thresholds in TSC cycles (<= l3dram => hit)
 *   setpush <passes>                  pusher sweeps after every access (default 1)
 *   calib <n> <out>                   uint16[4][n]: L1, L2, L3, DRAM-resident loads
 *   pushval <npush> <trials> <loads> <out>
 *                                     uint16[trials]: reload latency of X after
 *                                     <loads> loads of X + one push; random sets
 *   evset <set> <pool> <nseq> <nreset> <npush> <seed>
 *                                     find a 12-line eviction set by group testing,
 *                                     extend it, store as a slot
 *   seqs <slot> <reset_id> <R> <in> <out>
 *                                     in: "N\n" then "L i0 .. i(L-1)" per line
 *                                     (ids index the slot's sequence lines);
 *                                     out: uint16 cycles, order [seq][repeat][t]
 *   cond <mode> <passes>              conditioning traffic (set-dueling tests):
 *                                     1 = cyclic thrash over 1.5x L3 (favours
 *                                     thrash-resistant insertion), 2 = short-
 *                                     reuse stream (favours MRU insertion)
 *   quit
 */
#ifdef _WIN32
#include <windows.h>
#else
#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#include <cpuid.h>
#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HUGE_SZ   (2u << 20)
#define MAX_SLOTS 128
#define MAX_LINES 64
#define MAX_PUSH  64
#define MAX_POOL  4096

/* ----- timing ------------------------------------------------------------ */
static inline void ld(volatile char *p) { (void)*p; }

static inline uint32_t tload(volatile char *p) {
    unsigned aux;
    _mm_mfence(); _mm_lfence();
    uint64_t t0 = __rdtscp(&aux); _mm_lfence();
    (void)*p;
    _mm_lfence();
    uint64_t t1 = __rdtscp(&aux); _mm_lfence();
    uint64_t d = t1 - t0;
    return d > 65535 ? 65535 : (uint32_t)d;
}

/* xorshift so runs are reproducible from a seed (rand() is 15-bit on Windows) */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng;
}
static void shuffle_ptr(char **a, int n) {
    for (int i = n - 1; i > 0; i--) { int j = (int)(rnd() % (uint64_t)(i + 1)); char *t = a[i]; a[i] = a[j]; a[j] = t; }
}

/* ----- globals ----------------------------------------------------------- */
static char    *g_buf = NULL;          /* first page (logging only)              */
static char   **g_pages = NULL;        /* independently allocated 2 MiB pages     */
static size_t   g_bufsz = 0, g_npages = 0;
static int      g_large = 0;
static uint32_t g_thr_l2l3 = 0, g_thr_l3dram = 0;
static int      g_push_passes = 1;
static int      g_core = 2;
static int      g_prefetch_state = -1;   /* -1 unknown/unavailable, 0 disabled ok */
static char     g_prefetch_msg[256] = "not attempted";

/* geometry from CPUID leaf 4 (filled in main) */
static int g_l3_ways = 12, g_l3_sets = 32768, g_l2_ways = 10, g_l2_sets = 2048;

typedef struct {
    int set;                       /* per-slice set index (bits 6..17)         */
    int nseq, nreset, npush;
    char *seq[MAX_LINES];          /* lines used in sequences (seq[0] = target) */
    char *rst[MAX_LINES];          /* extra congruent lines used by resets      */
    char *push[MAX_PUSH];          /* L2 pushers                                */
    int rot;                       /* pusher rotation                           */
} Slot;
static Slot g_slots[MAX_SLOTS];
static int  g_nslots = 0;

static inline char *cand(size_t page, int h, int set) {
    return g_pages[page] + ((size_t)h << 18) + ((size_t)set << 6);
}

/* ----- platform ---------------------------------------------------------- */
static void pin_and_boost(int core) {
#ifdef _WIN32
    if (!SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core))
        fprintf(stderr, "[warn] SetThreadAffinityMask(%d) failed\n", core);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    Sleep(2);
#else
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s);
    if (sched_setaffinity(0, sizeof s, &s)) fprintf(stderr, "[warn] sched_setaffinity failed\n");
    struct sched_param sp = { .sched_priority = sched_get_priority_max(SCHED_FIFO) };
    if (sched_setscheduler(0, SCHED_FIFO, &sp)) setpriority(PRIO_PROCESS, 0, -20);
    usleep(2000);
#endif
}

#ifdef _WIN32
static int enable_lock_memory_privilege(void) {
    HANDLE tok; TOKEN_PRIVILEGES tp;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return 0;
    tp.PrivilegeCount = 1; tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    int ok = LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid)
          && AdjustTokenPrivileges(tok, FALSE, &tp, sizeof tp, NULL, NULL)
          && GetLastError() == ERROR_SUCCESS;
    CloseHandle(tok);
    return ok;
}
#endif

/* Allocate up to `mb` MiB as individual 2 MiB pages. One VirtualAlloc per page
 * succeeds even when Windows cannot find many free large pages in one request
 * (ERROR_NO_SYSTEM_RESOURCES 1450 on a fragmented machine). */
static void *alloc_one_huge(void) {
#ifdef _WIN32
    return VirtualAlloc(NULL, HUGE_SZ, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
#else
    void *p = mmap(NULL, HUGE_SZ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    return p == MAP_FAILED ? NULL : p;
#endif
}
static int alloc_buffer(size_t mb) {
    size_t want = (mb << 20) / HUGE_SZ;
#ifdef _WIN32
    if (GetLargePageMinimum() != HUGE_SZ || !enable_lock_memory_privilege()) {
        fprintf(stderr, "[err] SeLockMemoryPrivilege unavailable (run measure/grant_lock_pages.ps1)
");
        return 0;
    }
#endif
    g_pages = malloc(sizeof(char *) * want);
    for (g_npages = 0; g_npages < want; g_npages++) {
        char *p = alloc_one_huge();
        if (!p) break;
        g_pages[g_npages] = p;
    }
    if (g_npages < 8) {   /* without 2 MiB pages bits 12..17 are unknown: refuse */
        fprintf(stderr, "[err] only %zu x 2 MiB pages; need >= 8
", g_npages);
        return 0;
    }
    g_large = 1; g_buf = g_pages[0]; g_bufsz = g_npages * HUGE_SZ;
    for (size_t k = 0; k < g_npages; k++)
        for (size_t i = 0; i < HUGE_SZ; i += 4096) g_pages[k][i] = (char)i;
    fprintf(stderr, "[mem] %zu x 2 MiB pages (%zu requested)
", g_npages, want);
    return 1;
}
static inline char *line_addr(size_t i) {      /* i-th 64 B line across all pages */
    return g_pages[i >> 15] + ((i & 32767) << 6);
}

/* Linux only: physical address via pagemap (for logging / sanity). */
static uint64_t phys_addr(void *v) {
#ifdef _WIN32
    (void)v; return 0;
#else
    int fd = open("/proc/self/pagemap", O_RDONLY); if (fd < 0) return 0;
    uint64_t e = 0, va = (uint64_t)v;
    if (pread(fd, &e, 8, (off_t)(va / 4096) * 8) != 8) e = 0;
    close(fd);
    if (!(e >> 63) || !(e & ((1ull << 55) - 1))) return 0;
    return ((e & ((1ull << 55) - 1)) << 12) | (va & 4095);
#endif
}

/* Prefetcher control (MSR 0x1A4). Bits [VERIFY for Golden Cove/Raptor Cove]:
 * 0 L2 HW prefetcher, 1 L2 adjacent-line, 2 DCU streamer, 3 DCU IP.
 * Needs ring 0: Linux /dev/cpu/N/msr (root + msr module); impossible from
 * Windows user mode without a kernel driver. */
#ifndef _WIN32
static uint64_t g_msr_orig = 0;
#endif
static void prefetchers_disable(void) {
#ifdef _WIN32
    snprintf(g_prefetch_msg, sizeof g_prefetch_msg,
             "unavailable: Windows user mode has no MSR access; prefetchers left ENABLED");
#else
    char path[64]; snprintf(path, sizeof path, "/dev/cpu/%d/msr", g_core);
    int fd = open(path, O_RDWR);
    if (fd < 0) { snprintf(g_prefetch_msg, sizeof g_prefetch_msg, "open %s failed (need root + msr module)", path); return; }
    uint64_t v;
    if (pread(fd, &v, 8, 0x1A4) != 8) { snprintf(g_prefetch_msg, sizeof g_prefetch_msg, "rdmsr 0x1A4 failed"); close(fd); return; }
    g_msr_orig = v;
    uint64_t nv = v | 0xF;
    if (pwrite(fd, &nv, 8, 0x1A4) != 8) { snprintf(g_prefetch_msg, sizeof g_prefetch_msg, "wrmsr 0x1A4 rejected"); close(fd); return; }
    uint64_t rb = 0; pread(fd, &rb, 8, 0x1A4); close(fd);
    g_prefetch_state = ((rb & 0xF) == 0xF) ? 0 : -1;
    snprintf(g_prefetch_msg, sizeof g_prefetch_msg, "MSR 0x1A4 0x%llx -> 0x%llx (readback 0x%llx)",
             (unsigned long long)v, (unsigned long long)nv, (unsigned long long)rb);
#endif
}
static void prefetchers_restore(void) {
#ifndef _WIN32
    if (g_prefetch_state != 0) return;
    char path[64]; snprintf(path, sizeof path, "/dev/cpu/%d/msr", g_core);
    int fd = open(path, O_RDWR); if (fd < 0) return;
    pwrite(fd, &g_msr_orig, 8, 0x1A4); close(fd);
#endif
}

/* ----- CPUID geometry ---------------------------------------------------- */
static void cpuid_geometry(char *json, size_t cap) {
    unsigned a, b, c, d;
    size_t n = 0;
    n += snprintf(json + n, cap - n, "\"caches\":[");
    for (unsigned i = 0, first = 1; i < 16; i++) {
        __cpuid_count(4, i, a, b, c, d);
        unsigned type = a & 0x1F; if (!type) break;
        unsigned level = (a >> 5) & 7;
        unsigned ways = ((b >> 22) & 0x3FF) + 1, parts = ((b >> 12) & 0x3FF) + 1;
        unsigned line = (b & 0xFFF) + 1, sets = c + 1;
        unsigned sharing = ((a >> 14) & 0xFFF) + 1;
        unsigned incl = (d >> 1) & 1, cmplx = (d >> 2) & 1;
        unsigned long long size = (unsigned long long)ways * parts * line * sets;
        n += snprintf(json + n, cap - n,
            "%s{\"level\":%u,\"type\":%u,\"ways\":%u,\"sets\":%u,\"line\":%u,\"size\":%llu,"
            "\"inclusive\":%u,\"complex_index\":%u,\"max_sharing_threads\":%u}",
            first ? "" : ",", level, type, ways, sets, line, size, incl, cmplx, sharing);
        first = 0;
        if (level == 3) { g_l3_ways = ways; g_l3_sets = sets; }
        if (level == 2) { g_l2_ways = ways; g_l2_sets = sets; }
    }
    n += snprintf(json + n, cap - n, "]");
    __cpuid_count(0x1A, 0, a, b, c, d);
    unsigned ctype = a >> 24;
    n += snprintf(json + n, cap - n, ",\"core_type\":\"%s\"",
                  ctype == 0x40 ? "P-core" : ctype == 0x20 ? "E-core" : "unknown");
    __cpuid(1, a, b, c, d);
    n += snprintf(json + n, cap - n, ",\"cpuid_1_eax\":\"0x%x\"", a);
    unsigned brand[12];
    __cpuid(0x80000002, brand[0], brand[1], brand[2], brand[3]);
    __cpuid(0x80000003, brand[4], brand[5], brand[6], brand[7]);
    __cpuid(0x80000004, brand[8], brand[9], brand[10], brand[11]);
    char bs[49]; memcpy(bs, brand, 48); bs[48] = 0;
    char *p = bs; while (*p == ' ') p++;
    n += snprintf(json + n, cap - n, ",\"brand\":\"%s\"", p);
}

static void siblings_json(char *json, size_t cap) {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    char *buf = malloc(len);
    unsigned long long sib = 0; int eff = -1;
    if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore, (void *)buf, &len)) {
        for (DWORD off = 0; off < len;) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e = (void *)(buf + off);
            KAFFINITY m = e->Processor.GroupMask[0].Mask;
            if (m & ((KAFFINITY)1 << g_core)) { sib = (unsigned long long)m; eff = e->Processor.EfficiencyClass; }
            off += e->Size;
        }
    }
    free(buf);
    snprintf(json, cap, "\"core_logical_mask\":\"0x%llx\",\"efficiency_class\":%d", sib, eff);
#else
    char path[128], s[64] = "?";
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", g_core);
    FILE *f = fopen(path, "r"); if (f) { if (fgets(s, sizeof s, f)) s[strcspn(s, "\n")] = 0; fclose(f); }
    snprintf(json, cap, "\"thread_siblings\":\"%s\"", s);
#endif
}

/* ----- pushing a line out of L1/L2 --------------------------------------- */
static inline void push_slot(Slot *S) {
    for (int p = 0; p < g_push_passes; p++)
        for (int i = 0; i < S->npush; i++) ld(S->push[(S->rot + i) % S->npush]);
    S->rot = (S->rot + 1) % S->npush;
    _mm_mfence();
}

static void make_pushers(char **out, int n, int set) {
    int ps = set ^ 0x800;
    for (int i = 0; i < n; i++) out[i] = cand(rnd() % g_npages, (int)(rnd() % 8), ps);
}

/* ----- calibration -------------------------------------------------------- */
static void cmd_calib(int n, const char *path) {
    uint16_t *v = malloc(sizeof(uint16_t) * 4 * n);
    char *push[20], *l1ev[16];
    for (int k = 0; k < n; k++) {
        int set = (int)(rnd() % 2048);            /* bit 17 = 0 */
        char *x = cand(rnd() % g_npages, (int)(rnd() % 8), set);
        make_pushers(push, 20, set);
        for (int i = 0; i < 16; i++) l1ev[i] = cand(rnd() % g_npages, (int)(rnd() % 8), set ^ ((i + 1) << 6));
        /* L1: load then time */
        ld(x); _mm_mfence(); v[k] = (uint16_t)tload(x);
        /* L2: load, evict from L1 with lines in the same L1 set but other L2 sets */
        ld(x); _mm_mfence();
        for (int r = 0; r < 2; r++) for (int i = 0; i < 16; i++) ld(l1ev[i]);
        _mm_mfence(); v[n + k] = (uint16_t)tload(x);
        /* L3: load, push out of L1+L2 */
        ld(x); _mm_mfence();
        for (int r = 0; r < 2; r++) for (int i = 0; i < 20; i++) ld(push[i]);
        _mm_mfence(); v[2 * n + k] = (uint16_t)tload(x);
        /* DRAM: flush */
        _mm_clflush(x); _mm_mfence(); v[3 * n + k] = (uint16_t)tload(x);
    }
    FILE *f = fopen(path, "wb"); fwrite(v, 2, 4 * (size_t)n, f); fclose(f);
    free(v);
    printf("{\"ok\":1,\"n\":%d}\n", n);
}

/* Reload of X after <loads> loads + one push: L2 (push failed), L3, or DRAM
 * (the line was not kept in L3 when it left L2). */
static void cmd_pushval(int npush, int trials, int loads, const char *path) {
    uint16_t *v = malloc(sizeof(uint16_t) * trials);
    Slot S; memset(&S, 0, sizeof S);
    int c2 = 0, c3 = 0, cd = 0;
    for (int k = 0; k < trials; k++) {
        if (k % 64 == 0) { int set = (int)(rnd() % 2048); S.npush = npush; make_pushers(S.push, npush, set);
                           S.seq[0] = cand(rnd() % g_npages, (int)(rnd() % 8), set); }
        char *x = S.seq[0];
        _mm_clflush(x); _mm_mfence();
        for (int i = 0; i < loads; i++) { ld(x); _mm_mfence(); }
        push_slot(&S);
        uint32_t t = tload(x); v[k] = (uint16_t)t;
        if (g_thr_l2l3 && t <= g_thr_l2l3) c2++; else if (t <= g_thr_l3dram) c3++; else cd++;
    }
    FILE *f = fopen(path, "wb"); fwrite(v, 2, (size_t)trials, f); fclose(f);
    free(v);
    printf("{\"ok\":1,\"l2\":%d,\"l3\":%d,\"dram\":%d,\"trials\":%d}\n", c2, c3, cd, trials);
}

/* ----- eviction-set finding (Vila et al. group testing) ------------------ */
static Slot *g_cur;      /* slot whose pushers are used during the search */
static int   g_votes = 5, g_passes = 2;

/* load X, push it to L3, traverse set (fwd/back), time X. 1 = X evicted (DRAM). */
static int evicts_once(char **set, int n, char *x) {
    ld(x); _mm_mfence();
    push_slot(g_cur);
    for (int r = 0; r < g_passes; r++) {
        if (r & 1) for (int i = n - 1; i >= 0; i--) ld(set[i]);
        else       for (int i = 0; i < n; i++)      ld(set[i]);
        _mm_mfence();
    }
    push_slot(g_cur);  /* in case the traversal left X in L2 (it shares the L2 set) */
    return tload(x) > g_thr_l3dram;
}
static int evicts(char **set, int n, char *x) {
    int e = 0;
    for (int v = 0; v < g_votes; v++) e += evicts_once(set, n, x);
    return 2 * e > g_votes;
}
static double evict_rate(char **set, int n, char *x, int trials) {
    int e = 0;
    for (int t = 0; t < trials; t++) e += evicts_once(set, n, x);
    return (double)e / trials;
}

static int reduce(char **C, int n, char *x, int W) {
    static char *tmp[MAX_POOL];
    while (n > W) {
        int G = W + 1, removed = 0;
        for (int g = 0; g < G; g++) {
            int lo = (int)((long)n * g / G), hi = (int)((long)n * (g + 1) / G);
            if (hi <= lo) continue;
            int m = 0;
            for (int i = 0; i < n; i++) if (i < lo || i >= hi) tmp[m++] = C[i];
            if (m >= W && evicts(tmp, m, x)) { memcpy(C, tmp, sizeof(char *) * m); n = m; removed = 1; break; }
        }
        if (!removed) return -n;   /* stuck (noise); caller retries */
    }
    return n;
}

static void cmd_evset(int set, int pool, int nseq, int nreset, int npush, uint64_t seed) {
    if (g_nslots >= MAX_SLOTS || nseq + nreset > MAX_LINES || npush > MAX_PUSH || pool > MAX_POOL) {
        printf("{\"ok\":0,\"err\":\"limits\"}\n"); return;
    }
    clock_t t0 = clock();
    g_rng = seed * 0x9E3779B97F4A7C15ull + 1;
    Slot *S = &g_slots[g_nslots]; memset(S, 0, sizeof *S);
    S->set = set; S->nseq = nseq; S->nreset = nreset; S->npush = npush;
    make_pushers(S->push, npush, set);
    g_cur = S;
    int W = g_l3_ways;
    /* every same-index candidate in the buffer */
    int total = (int)g_npages * 8;
    char **all = malloc(sizeof(char *) * total);
    for (int i = 0; i < total; i++) all[i] = cand((size_t)(i / 8), i % 8, set);
    static char *C[MAX_POOL];
    int attempts = 0, es = 0; char *x = NULL;
    double r12 = 0, r11max = 0, pool_rate = 0;
    for (attempts = 1; attempts <= 10; attempts++) {
        shuffle_ptr(all, total);
        x = all[0];
        int n = pool < total - 1 ? pool : total - 1;
        for (int i = 0; i < n; i++) C[i] = all[1 + i];
        pool_rate = evict_rate(C, n, x, 10);
        if (pool_rate < 0.6) { fprintf(stderr, "[evset %d] attempt %d: pool evicts X only %.2f\n", set, attempts, pool_rate); continue; }
        es = reduce(C, n, x, W);
        if (es != W) { fprintf(stderr, "[evset %d] attempt %d: reduction stuck at %d\n", set, attempts, -es); continue; }
        r12 = evict_rate(C, W, x, 50);
        r11max = 0;
        for (int drop = 0; drop < W; drop++) {
            char *sub[64]; int m = 0;
            for (int i = 0; i < W; i++) if (i != drop) sub[m++] = C[i];
            double r = evict_rate(sub, m, x, 20); if (r > r11max) r11max = r;
        }
        if (r12 >= 0.9 && r11max <= 0.2) break;
        fprintf(stderr, "[evset %d] attempt %d: verify failed (12:%.2f 11max:%.2f)\n", set, attempts, r12, r11max);
    }
    if (attempts > 10) { free(all); printf("{\"ok\":0,\"set\":%d,\"err\":\"not found\"}\n", set); return; }
    /* congruent lines: X, the eviction set, then any other candidate the set evicts */
    char *cong[MAX_LINES]; int k = 0;
    cong[k++] = x;
    for (int i = 0; i < W && k < nseq + nreset; i++) cong[k++] = C[i];
    int tested = 0;
    for (int i = 1; i < total && k < nseq + nreset; i++) {
        char *c = all[i]; int dup = 0;
        for (int j = 0; j < k; j++) if (cong[j] == c) { dup = 1; break; }
        if (dup) continue;
        tested++;
        if (evicts(C, W, c)) cong[k++] = c;
    }
    free(all);
    if (k < nseq + nreset) { printf("{\"ok\":0,\"set\":%d,\"err\":\"only %d congruent\"}\n", set, k); return; }
    /* sequence lines: shuffled so line ids carry no finder-order meaning */
    shuffle_ptr(cong, k);
    for (int i = 0; i < nseq; i++) S->seq[i] = cong[i];
    for (int i = 0; i < nreset; i++) S->rst[i] = cong[nseq + i];
    /* re-verify: every extended line is evicted by 12 of the others */
    int bad = 0;
    for (int i = 0; i < k; i++) {
        char *others[MAX_LINES]; int m = 0;
        for (int j = 0; j < k && m < W; j++) if (j != i) others[m++] = cong[j];
        if (evict_rate(others, m, cong[i], 10) < 0.7) bad++;
    }
    int slot = g_nslots++;
    uint64_t pa = phys_addr(S->seq[0]);
    printf("{\"ok\":1,\"slot\":%d,\"set\":%d,\"attempts\":%d,\"pool_evict\":%.3f,\"es12_evict\":%.3f,"
           "\"es11_evict_max\":%.3f,\"congruent\":%d,\"tested\":%d,\"ext_verify_bad\":%d,"
           "\"seq0_offset\":%llu,\"seq0_phys\":%llu,\"ms\":%.0f}\n",
           slot, set, attempts, pool_rate, r12, r11max, k, tested, bad,
           (unsigned long long)((uintptr_t)S->seq[0] & (HUGE_SZ - 1)), (unsigned long long)pa,
           1000.0 * (clock() - t0) / CLOCKS_PER_SEC);
}

/* ----- resets ------------------------------------------------------------- */
static void flush_slot(Slot *S, int with_reset) {
    for (int i = 0; i < S->nseq; i++) _mm_clflush(S->seq[i]);
    if (with_reset) for (int i = 0; i < S->nreset; i++) _mm_clflush(S->rst[i]);
    _mm_mfence();
}
static void sweep_reset(Slot *S, int passes) {
    for (int p = 0; p < passes; p++)
        for (int i = 0; i < S->nreset; i++) { ld(S->rst[i]); _mm_mfence(); push_slot(S); }
}
/*  0 flush      : clflush sequence + reset lines
 *  1 flush+fill : 0, then load each of 12 reset lines once (pushed to L3)
 *  2 flush+fill2: 0, then all reset lines x2
 *  3 thrash     : no clflush; all reset lines x4 (natural eviction)
 *  4 flush+thrash: flush sequence lines only; all reset lines x4 */
static void do_reset(Slot *S, int id) {
    switch (id) {
    case 0: flush_slot(S, 1); break;
    case 1: flush_slot(S, 1);
            for (int i = 0; i < S->nreset && i < g_l3_ways; i++) { ld(S->rst[i]); _mm_mfence(); push_slot(S); }
            break;
    case 2: flush_slot(S, 1); sweep_reset(S, 2); break;
    case 3: sweep_reset(S, 4); break;
    case 4: flush_slot(S, 0); sweep_reset(S, 4); break;
    }
}

static void cmd_seqs(int slot, int reset_id, int R, const char *in, const char *out) {
    if (slot < 0 || slot >= g_nslots) { printf("{\"ok\":0,\"err\":\"bad slot\"}\n"); return; }
    Slot *S = &g_slots[slot];
    FILE *fi = fopen(in, "r"), *fo = fopen(out, "wb");
    if (!fi || !fo) { printf("{\"ok\":0,\"err\":\"io\"}\n"); if (fi) fclose(fi); if (fo) fclose(fo); return; }
    int N = 0; if (fscanf(fi, "%d", &N) != 1) N = 0;
    static int ids[4096]; static uint16_t cyc[4096];
    long long acc = 0, hits = 0;
    for (int s = 0; s < N; s++) {
        int L = 0; if (fscanf(fi, "%d", &L) != 1) break;
        if (L > 4096) L = 4096;
        for (int t = 0; t < L; t++) { if (fscanf(fi, "%d", &ids[t]) != 1) ids[t] = 0;
                                      if (ids[t] < 0 || ids[t] >= S->nseq) ids[t] = 0; }
        for (int r = 0; r < R; r++) {
            do_reset(S, reset_id);
            for (int t = 0; t < L; t++) {
                char *a = S->seq[ids[t]];
                uint32_t c = tload(a);
                push_slot(S);
                cyc[t] = (uint16_t)c;
                hits += c <= g_thr_l3dram;
            }
            fwrite(cyc, 2, (size_t)L, fo);
            acc += L;
        }
    }
    fclose(fi); fclose(fo);
    printf("{\"ok\":1,\"seqs\":%d,\"accesses\":%lld,\"hit_rate\":%.4f}\n", N, acc, acc ? (double)hits / acc : 0.0);
}

/* ----- conditioning traffic for set-dueling tests ------------------------- */
/* Lines are visited in a multiplicative-hash order (odd multiplier mod 2^k is a
 * bijection) so the stream/stride prefetchers see no pattern. */
static void cmd_cond(int mode, int passes) {
    size_t lines = g_bufsz / 64, region;
    unsigned long long cnt = 0;
    if (mode == 1) {  /* cyclic thrash over ~1.5x L3: LRU-like insertion -> 0 hits */
        region = 1; while (region * 64 < (size_t)(1.5 * g_l3_sets * g_l3_ways * 64)) region <<= 1;
        if (region > lines) region = lines;
        for (int p = 0; p < passes; p++)
            for (size_t i = 0; i < region; i++) { ld(line_addr((i * 0x9E3779B1ull) & (region - 1))); cnt++; }
    } else if (mode == 2) {  /* each line re-touched after ~4 other lines per set */
        region = 1; while (region * 2 <= lines) region <<= 1;
        size_t D = (size_t)4 * g_l3_sets;
        for (int p = 0; p < passes; p++)
            for (size_t i = D; i < region; i++) {
                ld(line_addr((i * 0x9E3779B1ull) & (region - 1)));
                ld(line_addr(((i - D) * 0x9E3779B1ull) & (region - 1)));
                cnt += 2;
            }
    }
    _mm_mfence();
    printf("{\"ok\":1,\"mode\":%d,\"loads\":%llu}\n", mode, cnt);
}

/* ----- main --------------------------------------------------------------- */
int main(int argc, char **argv) {
    size_t buf_mb = 256;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--core") && i + 1 < argc) g_core = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--buf-mb") && i + 1 < argc) buf_mb = (size_t)atoi(argv[++i]);
        else { fprintf(stderr, "usage: l3probe [--core C] [--buf-mb MB]\n"); return 2; }
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    pin_and_boost(g_core);
    char geo[4096]; cpuid_geometry(geo, sizeof geo);
    if (!alloc_buffer(buf_mb)) { printf("{\"ok\":0,\"err\":\"no large pages\"}\n"); return 1; }
    prefetchers_disable();
    fprintf(stderr, "[prefetch] %s\n", g_prefetch_msg);
    printf("{\"ok\":1,\"ready\":1}\n");

    char line[1024];
    while (fgets(line, sizeof line, stdin)) {
        char cmd[32] = {0}, a[512] = {0}, b[512] = {0};
        sscanf(line, "%31s", cmd);
        if (!strcmp(cmd, "quit")) break;
        else if (!strcmp(cmd, "info")) {
            char sib[256]; siblings_json(sib, sizeof sib);
            printf("{\"ok\":1,%s,%s,\"core\":%d,\"large_pages\":%d,\"buf_mb\":%zu,"
                   "\"prefetchers_disabled\":%d,\"prefetch_msg\":\"%s\",\"phys_addr_available\":%d,"
                   "\"thr_l2l3\":%u,\"thr_l3dram\":%u,\"push_passes\":%d}\n",
                   geo, sib, g_core, g_large, g_bufsz >> 20, g_prefetch_state == 0, g_prefetch_msg,
                   phys_addr(g_buf) != 0, g_thr_l2l3, g_thr_l3dram, g_push_passes);
        } else if (!strcmp(cmd, "setthr")) {
            unsigned x, y; if (sscanf(line, "%*s %u %u", &x, &y) == 2) { g_thr_l2l3 = x; g_thr_l3dram = y; }
            printf("{\"ok\":1,\"thr_l2l3\":%u,\"thr_l3dram\":%u}\n", g_thr_l2l3, g_thr_l3dram);
        } else if (!strcmp(cmd, "setpush")) {
            int p; if (sscanf(line, "%*s %d", &p) == 1 && p > 0) g_push_passes = p;
            printf("{\"ok\":1,\"push_passes\":%d}\n", g_push_passes);
        } else if (!strcmp(cmd, "setvote")) {
            int v, p; if (sscanf(line, "%*s %d %d", &v, &p) == 2) { g_votes = v; g_passes = p; }
            printf("{\"ok\":1,\"votes\":%d,\"passes\":%d}\n", g_votes, g_passes);
        } else if (!strcmp(cmd, "calib")) {
            int n; if (sscanf(line, "%*s %d %511s", &n, a) == 2) cmd_calib(n, a); else printf("{\"ok\":0}\n");
        } else if (!strcmp(cmd, "pushval")) {
            int np, tr, lo; if (sscanf(line, "%*s %d %d %d %511s", &np, &tr, &lo, a) == 4) cmd_pushval(np, tr, lo, a);
            else printf("{\"ok\":0}\n");
        } else if (!strcmp(cmd, "evset")) {
            int set, pool, ns, nr, np; unsigned long long seed;
            if (sscanf(line, "%*s %d %d %d %d %d %llu", &set, &pool, &ns, &nr, &np, &seed) == 6)
                cmd_evset(set, pool, ns, nr, np, seed);
            else printf("{\"ok\":0}\n");
        } else if (!strcmp(cmd, "seqs")) {
            int slot, rid, R; if (sscanf(line, "%*s %d %d %d %511s %511s", &slot, &rid, &R, a, b) == 5) cmd_seqs(slot, rid, R, a, b);
            else printf("{\"ok\":0}\n");
        } else if (!strcmp(cmd, "cond")) {
            int m, p; if (sscanf(line, "%*s %d %d", &m, &p) == 2) cmd_cond(m, p); else printf("{\"ok\":0}\n");
        } else printf("{\"ok\":0,\"err\":\"unknown command\"}\n");
        fflush(stdout);
    }
    prefetchers_restore();
    for (size_t k = 0; k < g_npages; k++) {
#ifdef _WIN32
        VirtualFree(g_pages[k], 0, MEM_RELEASE);
#else
        munmap(g_pages[k], HUGE_SZ);
#endif
    }
    return 0;
}
