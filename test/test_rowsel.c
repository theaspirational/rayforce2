/*
 *   Copyright (c) 2025-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "munit.h"
#include <rayforce.h>
#include "mem/heap.h"   /* BSIZEOF for block-size assertion */
#include "ops/ops.h"
#include "ops/rowsel.h"
#include <string.h>
#include <stdint.h>

/* ──────────────────────────────────────────────────────────────────
 * Helpers
 * ────────────────────────────────────────────────────────────────── */

/* Build a RAY_BOOL vec from a literal byte array. */
static ray_t* make_pred(const uint8_t* bytes, int64_t n) {
    ray_t* v = ray_vec_new(RAY_BOOL, n);
    if (!v || RAY_IS_ERR(v)) return NULL;
    v->len = n;
    memcpy(ray_data(v), bytes, (size_t)n);
    return v;
}

/* Naive popcount over the raw bool vec — used to cross-check the
 * per-segment popcounts the producer encodes via seg_offsets. */
static int64_t naive_popcount(const uint8_t* p, int64_t n) {
    int64_t c = 0;
    for (int64_t i = 0; i < n; i++) c += p[i] != 0;
    return c;
}

/* Walk a rowsel block and reconstruct the global row indices it
 * encodes (ALL segments expand to dense ranges, MIX uses idx[],
 * NONE skipped).  Used by tests to compare against an oracle. */
static int64_t reconstruct(ray_t* block, int64_t* out) {
    if (!block) return -1;
    ray_rowsel_t*   m       = ray_rowsel_meta(block);
    const uint8_t*  flags   = ray_rowsel_flags(block);
    const uint32_t* offsets = ray_rowsel_offsets(block);
    const uint16_t* idx     = ray_rowsel_idx(block);
    int64_t out_n = 0;
    for (uint32_t s = 0; s < m->n_segs; s++) {
        int64_t base = (int64_t)s * RAY_MORSEL_ELEMS;
        int64_t end  = base + RAY_MORSEL_ELEMS;
        if (end > m->nrows) end = m->nrows;
        if (flags[s] == RAY_SEL_NONE) continue;
        if (flags[s] == RAY_SEL_ALL) {
            for (int64_t r = base; r < end; r++) out[out_n++] = r;
            continue;
        }
        const uint16_t* slice = idx + offsets[s];
        uint32_t n = offsets[s + 1] - offsets[s];
        for (uint32_t i = 0; i < n; i++) out[out_n++] = base + slice[i];
    }
    return out_n;
}

/* ──────────────────────────────────────────────────────────────────
 * Tests
 * ────────────────────────────────────────────────────────────────── */

/* Empty input pred — returns an empty selection (n_segs == 0). */
static MunitResult test_rowsel_empty(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    ray_t* pred = make_pred((const uint8_t*)"", 0);
    munit_assert_ptr_not_null(pred);
    ray_t* sel = ray_rowsel_from_pred(pred);
    munit_assert_ptr_not_null(sel);
    ray_rowsel_t* m = ray_rowsel_meta(sel);
    munit_assert_int(m->total_pass, ==, 0);
    munit_assert_int(m->nrows, ==, 0);
    munit_assert_int(m->n_segs, ==, 0);
    ray_rowsel_release(sel);
    ray_release(pred);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* All-true pred — convention: returns NULL meaning "all rows pass". */
static MunitResult test_rowsel_all_pass(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    uint8_t bytes[100];
    memset(bytes, 1, sizeof(bytes));
    ray_t* pred = make_pred(bytes, 100);
    ray_t* sel = ray_rowsel_from_pred(pred);
    munit_assert_null(sel);  /* all-pass → NULL */
    ray_release(pred);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* All-false pred — empty selection, all flags NONE. */
static MunitResult test_rowsel_none_pass(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    uint8_t bytes[100] = {0};
    ray_t* pred = make_pred(bytes, 100);
    ray_t* sel = ray_rowsel_from_pred(pred);
    munit_assert_ptr_not_null(sel);
    ray_rowsel_t* m = ray_rowsel_meta(sel);
    munit_assert_int(m->total_pass, ==, 0);
    munit_assert_int(m->nrows, ==, 100);
    munit_assert_int(m->n_segs, ==, 1);
    munit_assert_int(ray_rowsel_flags(sel)[0], ==, RAY_SEL_NONE);
    ray_rowsel_release(sel);
    ray_release(pred);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* Single morsel, mixed pred — verify reconstruction. */
static MunitResult test_rowsel_single_morsel_mixed(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    /* 10 rows: positions 1, 3, 4, 7 set */
    uint8_t bytes[10] = {0,1,0,1,1,0,0,1,0,0};
    ray_t* pred = make_pred(bytes, 10);
    ray_t* sel = ray_rowsel_from_pred(pred);
    munit_assert_ptr_not_null(sel);
    ray_rowsel_t* m = ray_rowsel_meta(sel);
    munit_assert_int(m->total_pass, ==, 4);
    munit_assert_int(m->n_segs, ==, 1);
    munit_assert_int(ray_rowsel_flags(sel)[0], ==, RAY_SEL_MIX);
    int64_t out[10];
    int64_t n = reconstruct(sel, out);
    munit_assert_int(n, ==, 4);
    munit_assert_int(out[0], ==, 1);
    munit_assert_int(out[1], ==, 3);
    munit_assert_int(out[2], ==, 4);
    munit_assert_int(out[3], ==, 7);
    ray_rowsel_release(sel);
    ray_release(pred);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* Multi-morsel pred with one ALL segment, one NONE segment, one MIX
 * segment.  Forces the producer to dispatch all three flag paths. */
static MunitResult test_rowsel_multi_morsel(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    int64_t nrows = 3 * RAY_MORSEL_ELEMS;
    uint8_t* bytes = (uint8_t*)ray_data(ray_alloc((size_t)nrows));
    munit_assert_ptr_not_null(bytes);
    /* Seg 0: all true (ALL), Seg 1: all false (NONE),
     * Seg 2: every other row (MIX, 512 passing). */
    for (int64_t i = 0; i < RAY_MORSEL_ELEMS; i++) bytes[i] = 1;
    for (int64_t i = RAY_MORSEL_ELEMS; i < 2 * RAY_MORSEL_ELEMS; i++) bytes[i] = 0;
    for (int64_t i = 0; i < RAY_MORSEL_ELEMS; i++)
        bytes[2 * RAY_MORSEL_ELEMS + i] = (uint8_t)(i & 1);
    ray_t* pred = make_pred(bytes, nrows);
    ray_t* sel = ray_rowsel_from_pred(pred);
    munit_assert_ptr_not_null(sel);

    const uint8_t* flags = ray_rowsel_flags(sel);
    munit_assert_int(flags[0], ==, RAY_SEL_ALL);
    munit_assert_int(flags[1], ==, RAY_SEL_NONE);
    munit_assert_int(flags[2], ==, RAY_SEL_MIX);

    ray_rowsel_t* m = ray_rowsel_meta(sel);
    munit_assert_int(m->n_segs, ==, 3);
    munit_assert_int(m->total_pass, ==, RAY_MORSEL_ELEMS + RAY_MORSEL_ELEMS / 2);

    /* Reconstruct and compare to oracle. */
    int64_t* oracle = (int64_t*)ray_data(ray_alloc((size_t)m->total_pass * sizeof(int64_t)));
    int64_t  oracle_n = 0;
    for (int64_t i = 0; i < nrows; i++) if (bytes[i]) oracle[oracle_n++] = i;
    munit_assert_int(oracle_n, ==, m->total_pass);

    int64_t* recon = (int64_t*)ray_data(ray_alloc((size_t)m->total_pass * sizeof(int64_t)));
    int64_t  recon_n = reconstruct(sel, recon);
    munit_assert_int(recon_n, ==, m->total_pass);
    for (int64_t i = 0; i < recon_n; i++)
        munit_assert_int(recon[i], ==, oracle[i]);

    ray_rowsel_release(sel);
    ray_release(pred);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* Last morsel is partial (nrows not a multiple of RAY_MORSEL_ELEMS).
 * The "ALL" determination uses the morsel's actual length, not 1024. */
static MunitResult test_rowsel_partial_last_morsel(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    int64_t nrows = RAY_MORSEL_ELEMS + 5;  /* one full morsel + 5 rows */
    uint8_t* bytes = (uint8_t*)ray_data(ray_alloc((size_t)nrows));
    /* Seg 0: alternating (MIX, 512 passing).
     * Seg 1: 5 rows, all true → ALL. */
    for (int64_t i = 0; i < RAY_MORSEL_ELEMS; i++) bytes[i] = (uint8_t)(i & 1);
    for (int64_t i = 0; i < 5; i++) bytes[RAY_MORSEL_ELEMS + i] = 1;
    ray_t* pred = make_pred(bytes, nrows);
    ray_t* sel = ray_rowsel_from_pred(pred);
    munit_assert_ptr_not_null(sel);

    const uint8_t* flags = ray_rowsel_flags(sel);
    munit_assert_int(flags[0], ==, RAY_SEL_MIX);
    munit_assert_int(flags[1], ==, RAY_SEL_ALL);
    munit_assert_int(ray_rowsel_meta(sel)->total_pass, ==, RAY_MORSEL_ELEMS / 2 + 5);
    munit_assert_int(ray_rowsel_meta(sel)->n_segs, ==, 2);
    ray_rowsel_release(sel);
    ray_release(pred);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* Producer over a range that crosses the parallel threshold so the
 * pool dispatch fires.  Cross-checks reconstruction against the
 * naive oracle. */
static MunitResult test_rowsel_parallel(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    int64_t nrows = RAY_PARALLEL_THRESHOLD * 2 + 173;  /* parallel path + odd remainder */
    uint8_t* bytes = (uint8_t*)ray_data(ray_alloc((size_t)nrows));
    for (int64_t i = 0; i < nrows; i++) bytes[i] = (i % 7 == 0);
    ray_t* pred = make_pred(bytes, nrows);
    ray_t* sel = ray_rowsel_from_pred(pred);
    munit_assert_ptr_not_null(sel);

    int64_t expected_pass = naive_popcount(bytes, nrows);
    munit_assert_int(ray_rowsel_meta(sel)->total_pass, ==, expected_pass);

    int64_t* recon = (int64_t*)ray_data(ray_alloc((size_t)expected_pass * sizeof(int64_t)));
    int64_t  recon_n = reconstruct(sel, recon);
    munit_assert_int(recon_n, ==, expected_pass);
    int64_t check = 0;
    for (int64_t i = 0; i < nrows; i++) {
        if (bytes[i]) {
            munit_assert_int(recon[check], ==, i);
            check++;
        }
    }
    munit_assert_int(check, ==, expected_pass);
    ray_rowsel_release(sel);
    ray_release(pred);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* Many ALL segments + a few MIX — verifies that idx[] is sized for
 * MIX-contributed entries only, not total_pass.  Without this fix the
 * producer over-allocates idx[] to ~total_pass uint16s, which on a
 * 10M-row 99%-selective filter wastes ~20 MB.
 *
 * Test shape: 4 morsels, segments 0..2 are all-true (ALL, contribute
 * 0 to idx[]), segment 3 has 7 mixed bits.  total_pass should be
 * 3*1024 + 7 = 3079 but the underlying allocation should size idx[]
 * for only 7 entries. */
static MunitResult test_rowsel_all_segments_compact(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    int64_t nrows = 4 * RAY_MORSEL_ELEMS;
    ray_t* buf = ray_alloc((size_t)nrows);
    uint8_t* bytes = (uint8_t*)ray_data(buf);
    memset(bytes, 1, 3 * RAY_MORSEL_ELEMS);
    /* Segment 3: only 7 set bits, in a partial pattern. */
    memset(bytes + 3 * RAY_MORSEL_ELEMS, 0, RAY_MORSEL_ELEMS);
    int positions[] = {2, 5, 100, 333, 700, 900, 1023};
    for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); i++)
        bytes[3 * RAY_MORSEL_ELEMS + positions[i]] = 1;

    ray_t* pred = make_pred(bytes, nrows);
    ray_t* sel = ray_rowsel_from_pred(pred);
    munit_assert_ptr_not_null(sel);

    ray_rowsel_t* m = ray_rowsel_meta(sel);
    munit_assert_int(m->total_pass, ==, 3 * RAY_MORSEL_ELEMS + 7);
    munit_assert_int(m->n_segs, ==, 4);

    const uint8_t* flags = ray_rowsel_flags(sel);
    munit_assert_int(flags[0], ==, RAY_SEL_ALL);
    munit_assert_int(flags[1], ==, RAY_SEL_ALL);
    munit_assert_int(flags[2], ==, RAY_SEL_ALL);
    munit_assert_int(flags[3], ==, RAY_SEL_MIX);

    /* seg_offsets[n_segs] must equal 7 — the actual idx[]
     * occupancy.  This part is correct in BOTH the buggy and
     * fixed code (offsets are computed by walking popcounts) so
     * it's a sanity check, not the regression assertion. */
    const uint32_t* offsets = ray_rowsel_offsets(sel);
    munit_assert_int(offsets[0], ==, 0);
    munit_assert_int(offsets[1], ==, 0);
    munit_assert_int(offsets[2], ==, 0);
    munit_assert_int(offsets[3], ==, 0);
    munit_assert_int(offsets[4], ==, 7);

    /* Real regression assertion: the underlying ray_alloc block
     * size must reflect idx_count=7, not idx_count=total_pass.
     *
     * Fixed payload   = sizeof(meta)24 + pad8(4)8 + (4+1)*4 + 7*2     = 66 B
     * Buggy payload   = sizeof(meta)24 + pad8(4)8 + (4+1)*4 + 3079*2  = 6210 B
     *
     * Buddy allocator rounds to the next power-of-two order, so
     * the fixed block is order 7 (128 B) and the buggy block is
     * order 13 (8 KB).  Assert the block's actual size is less
     * than the buggy expectation. */
    size_t fixed_payload = ray_rowsel_payload_bytes(nrows, 7);
    size_t buggy_payload = ray_rowsel_payload_bytes(nrows, m->total_pass);
    munit_assert_size(fixed_payload, <, buggy_payload);
    size_t actual_block = BSIZEOF(sel->order);
    munit_assert_size(actual_block, <, buggy_payload);

    /* Reconstruct: 3072 dense rows from segments 0..2, plus 7
     * indexed rows from segment 3. */
    int64_t* recon = (int64_t*)ray_data(ray_alloc((size_t)m->total_pass * sizeof(int64_t)));
    int64_t recon_n = reconstruct(sel, recon);
    munit_assert_int(recon_n, ==, m->total_pass);
    /* Spot-check the segment-3 indices land in the right spots. */
    int64_t base = 3 * RAY_MORSEL_ELEMS;
    munit_assert_int(recon[m->total_pass - 7], ==, base + 2);
    munit_assert_int(recon[m->total_pass - 1], ==, base + 1023);

    ray_rowsel_release(sel);
    ray_release(pred);
    ray_release(buf);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* Refine: existing rowsel ANDed with a second pred shrinks correctly. */
static MunitResult test_rowsel_refine(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    int64_t nrows = 100;
    uint8_t a[100], b[100];
    for (int64_t i = 0; i < nrows; i++) {
        a[i] = (i % 2 == 0);   /* even */
        b[i] = (i % 3 == 0);   /* mult-of-3 */
    }
    ray_t* pa = make_pred(a, nrows);
    ray_t* pb = make_pred(b, nrows);

    ray_t* s1 = ray_rowsel_from_pred(pa);
    munit_assert_ptr_not_null(s1);
    ray_t* s2 = ray_rowsel_refine(s1, pb);
    munit_assert_ptr_not_null(s2);

    /* Expected survivors: even AND multiple of 3 → 0, 6, 12, …, 96 → 17 rows. */
    int64_t expect = 0;
    for (int64_t i = 0; i < nrows; i++) if (a[i] && b[i]) expect++;
    munit_assert_int(ray_rowsel_meta(s2)->total_pass, ==, expect);

    int64_t recon[100];
    int64_t recon_n = reconstruct(s2, recon);
    munit_assert_int(recon_n, ==, expect);
    int64_t check = 0;
    for (int64_t i = 0; i < nrows; i++) {
        if (a[i] && b[i]) {
            munit_assert_int(recon[check], ==, i);
            check++;
        }
    }
    ray_rowsel_release(s2);
    ray_rowsel_release(s1);
    ray_release(pa);
    ray_release(pb);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* to_indices: flattening must produce sorted global row indices
 * matching the oracle. */
static MunitResult test_rowsel_to_indices(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    int64_t nrows = 2 * RAY_MORSEL_ELEMS + 13;
    ray_t* buf = ray_alloc((size_t)nrows);
    uint8_t* bytes = (uint8_t*)ray_data(buf);
    for (int64_t i = 0; i < nrows; i++) bytes[i] = (i % 5 == 0);
    ray_t* pred = make_pred(bytes, nrows);
    ray_t* sel = ray_rowsel_from_pred(pred);
    munit_assert_ptr_not_null(sel);

    ray_t* idx_block = ray_rowsel_to_indices(sel);
    munit_assert_ptr_not_null(idx_block);
    int64_t* idx = (int64_t*)ray_data(idx_block);

    int64_t oracle_n = ray_rowsel_meta(sel)->total_pass;
    int64_t k = 0;
    for (int64_t i = 0; i < nrows; i++) {
        if (bytes[i]) {
            munit_assert_int(idx[k], ==, i);
            k++;
        }
    }
    munit_assert_int(k, ==, oracle_n);

    ray_release(idx_block);
    ray_rowsel_release(sel);
    ray_release(pred);
    ray_release(buf);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* Refine on a NULL existing — should behave like from_pred(pred). */
static MunitResult test_rowsel_refine_null_existing(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_heap_init();
    uint8_t bytes[10] = {1,0,1,0,1,0,1,0,1,0};
    ray_t* pred = make_pred(bytes, 10);
    ray_t* sel = ray_rowsel_refine(NULL, pred);
    munit_assert_ptr_not_null(sel);
    munit_assert_int(ray_rowsel_meta(sel)->total_pass, ==, 5);
    ray_rowsel_release(sel);
    ray_release(pred);
    ray_heap_destroy();
    return MUNIT_OK;
}

/* ──────────────────────────────────────────────────────────────────
 * Suite registration
 * ────────────────────────────────────────────────────────────────── */

static MunitTest rowsel_tests[] = {
    { "/empty",                  test_rowsel_empty,                  NULL, NULL, 0, NULL },
    { "/all_pass",               test_rowsel_all_pass,               NULL, NULL, 0, NULL },
    { "/none_pass",              test_rowsel_none_pass,              NULL, NULL, 0, NULL },
    { "/single_morsel_mixed",    test_rowsel_single_morsel_mixed,    NULL, NULL, 0, NULL },
    { "/multi_morsel",           test_rowsel_multi_morsel,           NULL, NULL, 0, NULL },
    { "/partial_last_morsel",    test_rowsel_partial_last_morsel,    NULL, NULL, 0, NULL },
    { "/parallel",               test_rowsel_parallel,               NULL, NULL, 0, NULL },
    { "/all_segments_compact",   test_rowsel_all_segments_compact,   NULL, NULL, 0, NULL },
    { "/to_indices",             test_rowsel_to_indices,             NULL, NULL, 0, NULL },
    { "/refine",                 test_rowsel_refine,                 NULL, NULL, 0, NULL },
    { "/refine_null_existing",   test_rowsel_refine_null_existing,   NULL, NULL, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL }
};

MunitSuite test_rowsel_suite = {
    "/rowsel", rowsel_tests, NULL, 1, 0
};
