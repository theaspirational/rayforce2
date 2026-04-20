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
#include "mem/heap.h"
#include "mem/arena.h"
#include "table/sym.h"
#include "store/col.h"
#include <string.h>
#include <stdio.h>

/* ---- Setup / Teardown -------------------------------------------------- */

static void* sym_setup(const void* params, void* user_data) {
    (void)params; (void)user_data;
    ray_heap_init();
    (void)ray_sym_init();
    return NULL;
}

static void sym_teardown(void* fixture) {
    (void)fixture;
    ray_sym_destroy();
    ray_heap_destroy();
}

/* ---- sym_init_destroy -------------------------------------------------- */

static MunitResult test_sym_init_destroy(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* After init, count should be 0 */
    munit_assert_uint(ray_sym_count(), ==, 0);

    return MUNIT_OK;
}

/* ---- sym_intern_basic -------------------------------------------------- */

static MunitResult test_sym_intern_basic(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id = ray_sym_intern("hello", 5);
    munit_assert_int(id, >=, 0);
    munit_assert_uint(ray_sym_count(), ==, 1);

    return MUNIT_OK;
}

/* ---- sym_intern_duplicate ---------------------------------------------- */

static MunitResult test_sym_intern_duplicate(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id1 = ray_sym_intern("hello", 5);
    int64_t id2 = ray_sym_intern("hello", 5);
    munit_assert_int(id1, ==, id2);
    munit_assert_uint(ray_sym_count(), ==, 1);

    return MUNIT_OK;
}

/* ---- sym_find_existing ------------------------------------------------- */

static MunitResult test_sym_find_existing(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id = ray_sym_intern("world", 5);
    int64_t found = ray_sym_find("world", 5);
    munit_assert_int(found, ==, id);

    return MUNIT_OK;
}

/* ---- sym_find_missing -------------------------------------------------- */

static MunitResult test_sym_find_missing(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t found = ray_sym_find("nonexistent", 11);
    munit_assert_int(found, ==, -1);

    return MUNIT_OK;
}

/* ---- sym_str_roundtrip ------------------------------------------------- */

static MunitResult test_sym_str_roundtrip(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id = ray_sym_intern("roundtrip", 9);
    munit_assert_int(id, >=, 0);

    ray_t* s = ray_sym_str(id);
    munit_assert_ptr_not_null(s);
    munit_assert_size(ray_str_len(s), ==, 9);
    munit_assert_memory_equal(9, ray_str_ptr(s), "roundtrip");

    return MUNIT_OK;
}

/* ---- sym_count --------------------------------------------------------- */

static MunitResult test_sym_count(const void* params, void* fixture) {
    (void)params; (void)fixture;

    munit_assert_uint(ray_sym_count(), ==, 0);

    ray_sym_intern("a", 1);
    munit_assert_uint(ray_sym_count(), ==, 1);

    ray_sym_intern("b", 1);
    munit_assert_uint(ray_sym_count(), ==, 2);

    ray_sym_intern("c", 1);
    munit_assert_uint(ray_sym_count(), ==, 3);

    /* Duplicate should not increase count */
    ray_sym_intern("a", 1);
    munit_assert_uint(ray_sym_count(), ==, 3);

    return MUNIT_OK;
}

/* ---- sym_many ---------------------------------------------------------- */

static MunitResult test_sym_many(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* Intern 1000 unique symbols */
    int64_t ids[1000];
    char buf[32];
    for (int i = 0; i < 1000; i++) {
        int len = snprintf(buf, sizeof(buf), "sym_%d", i);
        ids[i] = ray_sym_intern(buf, (size_t)len);
        munit_assert_int(ids[i], >=, 0);
    }

    munit_assert_uint(ray_sym_count(), ==, 1000);

    /* Verify all are distinct IDs */
    for (int i = 0; i < 1000; i++) {
        for (int j = i + 1; j < 1000; j++) {
            munit_assert_int(ids[i], !=, ids[j]);
        }
    }

    /* Verify all are retrievable with correct strings */
    for (int i = 0; i < 1000; i++) {
        int len = snprintf(buf, sizeof(buf), "sym_%d", i);
        ray_t* s = ray_sym_str(ids[i]);
        munit_assert_ptr_not_null(s);
        munit_assert_size(ray_str_len(s), ==, (size_t)len);
        munit_assert_memory_equal((size_t)len, ray_str_ptr(s), buf);
    }

    /* Re-interning should return same IDs */
    for (int i = 0; i < 1000; i++) {
        int len = snprintf(buf, sizeof(buf), "sym_%d", i);
        int64_t id2 = ray_sym_intern(buf, (size_t)len);
        munit_assert_int(id2, ==, ids[i]);
    }

    munit_assert_uint(ray_sym_count(), ==, 1000);

    return MUNIT_OK;
}

/* ---- sym_bulk: intern 100K symbols and verify none are lost ------------ */

static MunitResult test_sym_bulk(const void* params, void* fixture) {
    (void)params; (void)fixture;

    #define BULK_N 100000

    /* Pre-reserve capacity (tests ray_sym_ensure_cap) */
    bool cap_ok = ray_sym_ensure_cap(BULK_N);
    munit_assert_true(cap_ok);

    /* Intern 100K unique symbols */
    char buf[32];
    for (int i = 0; i < BULK_N; i++) {
        int len = snprintf(buf, sizeof(buf), "bulk_%06d", i);
        int64_t id = ray_sym_intern(buf, (size_t)len);
        munit_assert_int(id, >=, 0);
    }

    munit_assert_uint(ray_sym_count(), ==, BULK_N);

    /* Verify every symbol is retrievable with correct string */
    for (int i = 0; i < BULK_N; i++) {
        int len = snprintf(buf, sizeof(buf), "bulk_%06d", i);
        int64_t id = ray_sym_find(buf, (size_t)len);
        munit_assert_int(id, >=, 0);
        ray_t* s = ray_sym_str(id);
        munit_assert_ptr_not_null(s);
        munit_assert_size(ray_str_len(s), ==, (size_t)len);
        munit_assert_memory_equal((size_t)len, ray_str_ptr(s), buf);
    }

    /* Re-interning must return same IDs (idempotent) */
    for (int i = 0; i < BULK_N; i++) {
        int len = snprintf(buf, sizeof(buf), "bulk_%06d", i);
        int64_t id1 = ray_sym_find(buf, (size_t)len);
        int64_t id2 = ray_sym_intern(buf, (size_t)len);
        munit_assert_int(id1, ==, id2);
    }

    munit_assert_uint(ray_sym_count(), ==, BULK_N);

    #undef BULK_N
    return MUNIT_OK;
}

/* ---- sym_save_load_roundtrip ------------------------------------------- */

static MunitResult test_sym_save_load_roundtrip(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_roundtrip.sym";

    /* Intern some symbols */
    int64_t id_hello = ray_sym_intern("hello", 5);
    int64_t id_world = ray_sym_intern("world", 5);
    int64_t id_foo   = ray_sym_intern("foo", 3);
    munit_assert_int(id_hello, >=, 0);
    munit_assert_int(id_world, >=, 0);
    munit_assert_int(id_foo, >=, 0);
    munit_assert_uint(ray_sym_count(), ==, 3);

    /* Save */
    ray_err_t err = ray_sym_save(sym_path);
    munit_assert_int(err, ==, RAY_OK);

    /* Destroy and re-init sym table */
    ray_sym_destroy();
    (void)ray_sym_init();
    munit_assert_uint(ray_sym_count(), ==, 0);

    /* Load */
    err = ray_sym_load(sym_path);
    munit_assert_int(err, ==, RAY_OK);
    munit_assert_uint(ray_sym_count(), ==, 3);

    /* Verify all strings match */
    ray_t* s0 = ray_sym_str(id_hello);
    munit_assert_ptr_not_null(s0);
    munit_assert_size(ray_str_len(s0), ==, 5);
    munit_assert_memory_equal(5, ray_str_ptr(s0), "hello");

    ray_t* s1 = ray_sym_str(id_world);
    munit_assert_ptr_not_null(s1);
    munit_assert_size(ray_str_len(s1), ==, 5);
    munit_assert_memory_equal(5, ray_str_ptr(s1), "world");

    ray_t* s2 = ray_sym_str(id_foo);
    munit_assert_ptr_not_null(s2);
    munit_assert_size(ray_str_len(s2), ==, 3);
    munit_assert_memory_equal(3, ray_str_ptr(s2), "foo");

    /* Cleanup */
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    return MUNIT_OK;
}

/* ---- sym_save_append_only --------------------------------------------- */

static MunitResult test_sym_save_append_only(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_append.sym";

    /* Intern initial batch */
    int64_t id_a = ray_sym_intern("alpha", 5);
    int64_t id_b = ray_sym_intern("beta", 4);
    munit_assert_int(id_a, >=, 0);
    munit_assert_int(id_b, >=, 0);

    /* First save */
    ray_err_t err = ray_sym_save(sym_path);
    munit_assert_int(err, ==, RAY_OK);

    /* Second save with no changes -> should be no-op */
    err = ray_sym_save(sym_path);
    munit_assert_int(err, ==, RAY_OK);

    /* Intern more symbols */
    int64_t id_c = ray_sym_intern("gamma", 5);
    int64_t id_d = ray_sym_intern("delta", 5);
    munit_assert_int(id_c, >=, 0);
    munit_assert_int(id_d, >=, 0);

    /* Save again (append-only: new entries added) */
    err = ray_sym_save(sym_path);
    munit_assert_int(err, ==, RAY_OK);

    /* Destroy and reload */
    ray_sym_destroy();
    (void)ray_sym_init();
    err = ray_sym_load(sym_path);
    munit_assert_int(err, ==, RAY_OK);
    munit_assert_uint(ray_sym_count(), ==, 4);

    /* Verify old IDs are stable */
    ray_t* sa = ray_sym_str(id_a);
    munit_assert_ptr_not_null(sa);
    munit_assert_memory_equal(5, ray_str_ptr(sa), "alpha");

    ray_t* sb = ray_sym_str(id_b);
    munit_assert_ptr_not_null(sb);
    munit_assert_memory_equal(4, ray_str_ptr(sb), "beta");

    /* Verify new IDs are present */
    ray_t* sc = ray_sym_str(id_c);
    munit_assert_ptr_not_null(sc);
    munit_assert_memory_equal(5, ray_str_ptr(sc), "gamma");

    ray_t* sd = ray_sym_str(id_d);
    munit_assert_ptr_not_null(sd);
    munit_assert_memory_equal(5, ray_str_ptr(sd), "delta");

    /* Cleanup */
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    return MUNIT_OK;
}

/* ---- sym_load_corrupt -------------------------------------------------- */

static MunitResult test_sym_load_corrupt(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_corrupt.sym";

    /* Write garbage data */
    FILE* f = fopen(sym_path, "wb");
    munit_assert_ptr_not_null(f);
    const char garbage[] = "this is not a valid sym file";
    fwrite(garbage, 1, sizeof(garbage), f);
    fclose(f);

    /* Load should fail with corrupt */
    ray_err_t err = ray_sym_load(sym_path);
    munit_assert_int(err, !=, RAY_OK);

    /* Count unchanged */
    munit_assert_uint(ray_sym_count(), ==, 0);

    /* Cleanup */
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    return MUNIT_OK;
}

/* ---- sym_load_truncated ------------------------------------------------ */

static MunitResult test_sym_load_truncated(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_trunc.sym";

    /* Intern and save valid sym file */
    ray_sym_intern("abc", 3);
    ray_sym_intern("def", 3);
    ray_err_t err = ray_sym_save(sym_path);
    munit_assert_int(err, ==, RAY_OK);

    /* Truncate the file to 2 bytes */
    FILE* f = fopen(sym_path, "wb");
    munit_assert_ptr_not_null(f);
    fwrite("AB", 1, 2, f);
    fclose(f);

    /* Destroy and re-init */
    ray_sym_destroy();
    (void)ray_sym_init();

    /* Load should fail */
    err = ray_sym_load(sym_path);
    munit_assert_int(err, !=, RAY_OK);
    munit_assert_uint(ray_sym_count(), ==, 0);

    /* Cleanup */
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    return MUNIT_OK;
}

/* ---- sym_dotted -------------------------------------------------------- */

static MunitResult test_sym_dotted_detect(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t flat_id = ray_sym_intern("foo", 3);
    munit_assert_int(flat_id, >=, 0);
    munit_assert_false(ray_sym_is_dotted(flat_id));

    int64_t dotted_id = ray_sym_intern("foo.bar", 7);
    munit_assert_int(dotted_id, >=, 0);
    munit_assert_true(ray_sym_is_dotted(dotted_id));

    /* foo.bar is a distinct sym_id from foo */
    munit_assert_int(dotted_id, !=, flat_id);

    return MUNIT_OK;
}

static MunitResult test_sym_dotted_segments(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id = ray_sym_intern("math.pi", 7);
    munit_assert_int(id, >=, 0);
    munit_assert_true(ray_sym_is_dotted(id));

    const int64_t* segs = NULL;
    int n = ray_sym_segs(id, &segs);
    munit_assert_int(n, ==, 2);
    munit_assert_ptr_not_null(segs);

    /* Each segment must itself be interned and resolvable */
    int64_t math_id = ray_sym_find("math", 4);
    int64_t pi_id   = ray_sym_find("pi",   2);
    munit_assert_int(math_id, ==, segs[0]);
    munit_assert_int(pi_id,   ==, segs[1]);

    /* Segments themselves are plain names (not dotted) */
    munit_assert_false(ray_sym_is_dotted(segs[0]));
    munit_assert_false(ray_sym_is_dotted(segs[1]));

    return MUNIT_OK;
}

static MunitResult test_sym_dotted_triple(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id = ray_sym_intern("a.b.c", 5);
    munit_assert_int(id, >=, 0);
    munit_assert_true(ray_sym_is_dotted(id));

    const int64_t* segs = NULL;
    int n = ray_sym_segs(id, &segs);
    munit_assert_int(n, ==, 3);
    munit_assert_int(ray_sym_find("a", 1), ==, segs[0]);
    munit_assert_int(ray_sym_find("b", 1), ==, segs[1]);
    munit_assert_int(ray_sym_find("c", 1), ==, segs[2]);

    return MUNIT_OK;
}

static MunitResult test_sym_dotted_idempotent(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id1 = ray_sym_intern("lib.util", 8);
    int64_t id2 = ray_sym_intern("lib.util", 8);
    munit_assert_int(id1, ==, id2);
    munit_assert_true(ray_sym_is_dotted(id1));

    const int64_t* segs1 = NULL;
    const int64_t* segs2 = NULL;
    int n1 = ray_sym_segs(id1, &segs1);
    int n2 = ray_sym_segs(id2, &segs2);
    munit_assert_int(n1, ==, n2);
    /* Same symbol so the cached segment pointer should be identical too */
    munit_assert_ptr_equal((void*)segs1, (void*)segs2);

    return MUNIT_OK;
}

static MunitResult test_sym_dotted_rejects_empty_segment(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* Leading / trailing dot, or double-dot: the name still interns (a plain
     * symbol with '.' in it), but is_dotted stays false so env-lookup
     * doesn't try to split it. */
    int64_t id1 = ray_sym_intern(".foo", 4);
    int64_t id2 = ray_sym_intern("foo.", 4);
    int64_t id3 = ray_sym_intern("a..b", 4);
    munit_assert_int(id1, >=, 0);
    munit_assert_int(id2, >=, 0);
    munit_assert_int(id3, >=, 0);
    munit_assert_false(ray_sym_is_dotted(id1));
    munit_assert_false(ray_sym_is_dotted(id2));
    munit_assert_false(ray_sym_is_dotted(id3));

    return MUNIT_OK;
}

/* ---- sym_bytes_upper_covers_sym_str_arena ------------------------------ */

/* Directly validates ray_sym_bytes_upper against the actual arena
 * consumption of sym_str_arena's allocation pattern for every length
 * in a wide range.  The previous formula returned `chars_block` for
 * len>=7 but sym_str_arena's ray_arena_alloc actually charges
 * 32 + chars_block — an off-by-32 under-reservation that could still
 * cause the atomic-intern commit phase to trip a new-chunk allocation
 * and fail.  A regression would cause this test to fail for the first
 * long length (len=7) where actual > predicted.
 *
 * The test mirrors sym_str_arena's allocation pattern from sym.c:
 * short names (len<7) call ray_arena_alloc(_, 0); long names call
 * ray_arena_alloc(_, chars_block) where chars_block = ALIGN(32+len+1). */
static MunitResult test_sym_bytes_upper_covers_arena(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* Local arena so we're isolated from g_sym's state. */
    ray_arena_t* a = ray_arena_new(1024 * 1024);
    munit_assert_ptr_not_null(a);

    for (size_t len = 0; len <= 128; len++) {
        size_t before = ray_arena_total_used(a);
        ray_t* v;
        if (len < 7) {
            v = ray_arena_alloc(a, 0);
        } else {
            size_t chars_block = ((size_t)32 + len + 1 + 31) & ~(size_t)31;
            v = ray_arena_alloc(a, chars_block);
        }
        munit_assert_ptr_not_null(v);
        size_t after = ray_arena_total_used(a);

        size_t actual    = after - before;
        size_t predicted = ray_sym_bytes_upper(len);
        /* Invariant: predicted must be a safe upper bound. */
        if (actual > predicted) {
            fprintf(stderr,
                "ray_sym_bytes_upper(%zu) = %zu but arena consumed %zu bytes\n",
                len, predicted, actual);
            return MUNIT_FAIL;
        }
    }

    ray_arena_destroy(a);
    return MUNIT_OK;
}

/* ---- sym_intern_long_segments ------------------------------------------ */

/* Exercises the long-string arena path (segment name length >= 7) for
 * every commit inside a dotted intern.  sym_str_arena takes a different
 * allocation branch for names >= 7 bytes (fused CHAR+STR block) and the
 * pre-reservation formula has to account for the ARENA_ALIGN_UP(32 +
 * nbytes) overhead that ray_arena_alloc charges for each block —
 * previously under-reserved by 32 bytes per long string, so phase C
 * could trigger a new-chunk allocation and fail mid-commit.  Many deep
 * paths with long segments stress that repeatedly. */
static MunitResult test_sym_intern_long_segments(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* 5-segment path, every segment >=7 bytes: forces the long-string
     * path for main + all segments in one atomic commit. */
    const char* name =
        "configuration.database.primary.connection.timeout_seconds";
    size_t len = strlen(name);
    int64_t id = ray_sym_intern(name, len);
    munit_assert_int(id, >=, 0);
    munit_assert_true(ray_sym_is_dotted(id));

    const int64_t* segs = NULL;
    int n = ray_sym_segs(id, &segs);
    munit_assert_int(n, ==, 5);
    munit_assert_int(ray_sym_find("configuration",    13), ==, segs[0]);
    munit_assert_int(ray_sym_find("database",          8), ==, segs[1]);
    munit_assert_int(ray_sym_find("primary",           7), ==, segs[2]);
    munit_assert_int(ray_sym_find("connection",       10), ==, segs[3]);
    munit_assert_int(ray_sym_find("timeout_seconds",  15), ==, segs[4]);

    /* Repeat with a second long path that overlaps the first on one
     * segment to confirm re-use works under the atomic commit. */
    int64_t id2 = ray_sym_intern("configuration.logging.destination", 33);
    munit_assert_int(id2, >=, 0);
    munit_assert_true(ray_sym_is_dotted(id2));
    segs = NULL;
    n = ray_sym_segs(id2, &segs);
    munit_assert_int(n, ==, 3);
    munit_assert_int(segs[0], ==, ray_sym_find("configuration",  13));

    return MUNIT_OK;
}

/* ---- sym_intern_atomic_no_orphans -------------------------------------- */

/* Interning a dotted name must commit the main sym BEFORE (or at least
 * in lockstep with) its segment syms — we must never observe a state
 * where the segments exist but the main name does not.  Previously,
 * prep sub-interned segments eagerly and only then attempted the main
 * commit; a failure between those steps would leave the segments as
 * orphans on the sym table.  The three-phase intern closes that gap:
 * all commits happen under a reservation that guarantees success. */
static MunitResult test_sym_intern_atomic_no_orphans(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* Fresh table: no segments or main name yet. */
    munit_assert_int(ray_sym_find("cfg",      3), ==, -1);
    munit_assert_int(ray_sym_find("host",     4), ==, -1);
    munit_assert_int(ray_sym_find("cfg.host", 8), ==, -1);

    /* Intern the dotted name.  After success, main + both segments exist
     * AND are linked through the segments cache. */
    int64_t id = ray_sym_intern("cfg.host", 8);
    munit_assert_int(id, >=, 0);

    int64_t host_id = ray_sym_find("host", 4);
    int64_t cfg_id  = ray_sym_find("cfg",  3);
    munit_assert_int(host_id, >=, 0);
    munit_assert_int(cfg_id,  >=, 0);

    munit_assert_true(ray_sym_is_dotted(id));
    const int64_t* segs = NULL;
    int n = ray_sym_segs(id, &segs);
    munit_assert_int(n, ==, 2);
    munit_assert_int(segs[0], ==, cfg_id);
    munit_assert_int(segs[1], ==, host_id);

    /* Segments are marked as plain (scanned bit set, dotted bit clear). */
    munit_assert_false(ray_sym_is_dotted(cfg_id));
    munit_assert_false(ray_sym_is_dotted(host_id));

    /* Re-interning reuses the same ids — no duplicate commits. */
    munit_assert_int(ray_sym_intern("cfg.host", 8), ==, id);
    munit_assert_int(ray_sym_intern("cfg",      3), ==, cfg_id);
    munit_assert_int(ray_sym_intern("host",     4), ==, host_id);

    return MUNIT_OK;
}

/* ---- sym_intern_atomic_cache ------------------------------------------- */

/* Invariant: if ray_sym_intern returns a valid id, the dotted/scanned
 * metadata for that id is already consistent with the name's structure.
 * If we ever regress to committing the sym first and caching after (a la
 * the previous "sym_intern_nolock_noseg then sym_cache_segments" split),
 * env_set/env_get would silently take the flat path for a dotted name
 * during the window before caching ran.  This test locks that shut. */
static MunitResult test_sym_intern_atomic_cache(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* Dotted name: is_dotted + segments must be populated on return. */
    int64_t id = ray_sym_intern("ns.leaf", 7);
    munit_assert_int(id, >=, 0);
    munit_assert_true(ray_sym_is_dotted(id));
    const int64_t* segs = NULL;
    int n = ray_sym_segs(id, &segs);
    munit_assert_int(n, ==, 2);
    munit_assert_int(ray_sym_find("ns",   2), ==, segs[0]);
    munit_assert_int(ray_sym_find("leaf", 4), ==, segs[1]);

    /* Plain name: is_dotted is false immediately, segments empty. */
    int64_t pid = ray_sym_intern("plain_thing", 11);
    munit_assert_int(pid, >=, 0);
    munit_assert_false(ray_sym_is_dotted(pid));
    segs = NULL;
    munit_assert_int(ray_sym_segs(pid, &segs), ==, 0);

    /* Deep path: all segments interned, all visible. */
    int64_t d = ray_sym_intern("a.b.c.d.e", 9);
    munit_assert_int(d, >=, 0);
    munit_assert_true(ray_sym_is_dotted(d));
    segs = NULL;
    munit_assert_int(ray_sym_segs(d, &segs), ==, 5);

    return MUNIT_OK;
}

/* ---- sym_dotted_reintern_retries_cache --------------------------------- */

/* If the very first intern of a dotted name fails to cache its segments
 * (e.g. transient arena OOM), the next intern of the same name must
 * retry the cache — otherwise a momentary OOM would permanently strip
 * the dotted semantics from that sym.  We simulate "first intern didn't
 * cache" with ray_sym_intern_no_split, then verify the normal intern
 * path picks up the missing cache. */
static MunitResult test_sym_dotted_reintern_retries_cache(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* Seed the table with a dotted sym via the no-split path — this is
     * exactly the in-memory state after a transient cache failure:
     * the sym exists but its dotted bit is clear. */
    int64_t id = ray_sym_intern_no_split("math.pi", 7);
    munit_assert_int(id, >=, 0);
    munit_assert_false(ray_sym_is_dotted(id));

    /* Re-intern through the normal path.  Must return the same id AND
     * retroactively populate the segment cache. */
    int64_t id2 = ray_sym_intern("math.pi", 7);
    munit_assert_int(id2, ==, id);
    munit_assert_true(ray_sym_is_dotted(id));

    const int64_t* segs = NULL;
    int n = ray_sym_segs(id, &segs);
    munit_assert_int(n, ==, 2);
    munit_assert_int(ray_sym_find("math", 4), ==, segs[0]);
    munit_assert_int(ray_sym_find("pi",   2), ==, segs[1]);

    return MUNIT_OK;
}

/* ---- sym_rebuild_segments_contract ------------------------------------- */

/* Direct-call contract for ray_sym_rebuild_segments: returns RAY_OK on
 * success, is idempotent, and populates the dotted cache for entries
 * created via the no-split variant (which is what the persistence paths
 * use internally). */
static MunitResult test_sym_rebuild_segments_contract(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* Empty table: rebuild is a no-op but still OK. */
    munit_assert_int(ray_sym_rebuild_segments(), ==, RAY_OK);

    /* Insert a dotted name via the no-split path: segments are NOT cached
     * until rebuild runs. */
    int64_t id = ray_sym_intern_no_split("alpha.beta", 10);
    munit_assert_int(id, >=, 0);
    munit_assert_false(ray_sym_is_dotted(id));   /* not yet cached */

    /* Rebuild: cache populated, succeeds. */
    munit_assert_int(ray_sym_rebuild_segments(), ==, RAY_OK);
    munit_assert_true(ray_sym_is_dotted(id));

    /* Idempotent: second call is still OK, doesn't re-cache. */
    const int64_t* segs1 = NULL;
    int n1 = ray_sym_segs(id, &segs1);
    munit_assert_int(ray_sym_rebuild_segments(), ==, RAY_OK);
    const int64_t* segs2 = NULL;
    int n2 = ray_sym_segs(id, &segs2);
    munit_assert_int(n1, ==, n2);
    munit_assert_ptr_equal((void*)segs1, (void*)segs2);  /* same cached array */

    return MUNIT_OK;
}

/* ---- sym_load_legacy_dotted -------------------------------------------- */

/* Pre-feature sym files can contain dotted names like "user.name" (a CSV
 * column, a sym literal, etc.) *without* accompanying segment entries
 * because the old interner didn't split on '.'.  Loading must still
 * succeed: segment sub-interning during load would otherwise shift
 * subsequent disk positions and trip the id==pos check. */
static MunitResult test_sym_load_legacy_dotted(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_legacy_dotted.sym";
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    /* Build a file-on-disk that only contains ["alice", "user.name",
     * "charlie"] — no "user" or "name" entries following the dotted name.
     * Temporarily disable segment caching by interning the dotted name
     * with a fake name first, then renaming via direct RAY_LIST build. */
    ray_t* list = ray_list_new(3);
    munit_assert_ptr_not_null(list);
    ray_t* s0 = ray_str("alice", 5);
    ray_t* s1 = ray_str("user.name", 9);
    ray_t* s2 = ray_str("charlie", 7);
    list = ray_list_append(list, s0); ray_release(s0);
    list = ray_list_append(list, s1); ray_release(s1);
    list = ray_list_append(list, s2); ray_release(s2);
    munit_assert_ptr_not_null(list);
    munit_assert_false(RAY_IS_ERR(list));
    ray_err_t err = ray_col_save(list, sym_path);
    munit_assert_int(err, ==, RAY_OK);
    ray_release(list);

    /* Load must succeed even though the file has a dotted name but no
     * segment entries.  The load should re-intern the three disk entries
     * at ids 0,1,2; then separately cache segment info for "user.name",
     * placing "user" and "name" at whatever transient ids follow. */
    err = ray_sym_load(sym_path);
    munit_assert_int(err, ==, RAY_OK);

    munit_assert_int(ray_sym_find("alice",     5), ==, 0);
    munit_assert_int(ray_sym_find("user.name", 9), ==, 1);
    munit_assert_int(ray_sym_find("charlie",   7), ==, 2);
    munit_assert_true(ray_sym_is_dotted(1));

    const int64_t* segs = NULL;
    int n = ray_sym_segs(1, &segs);
    munit_assert_int(n, ==, 2);
    munit_assert_int(ray_sym_find("user", 4), ==, segs[0]);
    munit_assert_int(ray_sym_find("name", 4), ==, segs[1]);

    remove(sym_path);
    remove(lk_path);
    return MUNIT_OK;
}

/* ---- sym_save_load_dotted ---------------------------------------------- */

/* Persistence must be stable across dotted names.  This exercises the
 * interaction between segment sub-interning and the disk-position ==
 * sym_id invariant that ray_sym_load/ray_sym_save rely on. */
static MunitResult test_sym_save_load_dotted(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_dotted.sym";
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    /* Intern a mix of plain and dotted names.  The dotted name triggers
     * segment sub-interning, appending "alpha" + "beta" at further ids. */
    int64_t id_plain      = ray_sym_intern("plain_one", 9);
    int64_t id_dotted     = ray_sym_intern("alpha.beta", 10);
    int64_t id_after      = ray_sym_intern("plain_two", 9);
    munit_assert_int(id_plain, >=, 0);
    munit_assert_int(id_dotted, >=, 0);
    munit_assert_int(id_after, >=, 0);
    munit_assert_true(ray_sym_is_dotted(id_dotted));

    /* Persist */
    ray_err_t err = ray_sym_save(sym_path);
    munit_assert_int(err, ==, RAY_OK);

    /* Tear down and reload from disk */
    ray_sym_destroy();
    (void)ray_sym_init();
    munit_assert_uint(ray_sym_count(), ==, 0);

    err = ray_sym_load(sym_path);
    munit_assert_int(err, ==, RAY_OK);

    /* Every pre-save sym_id must still resolve to its original string.
     * If load's segment sub-intern shifted ids, this is where it breaks. */
    ray_t* sp = ray_sym_str(id_plain);
    munit_assert_ptr_not_null(sp);
    munit_assert_size(ray_str_len(sp), ==, 9);
    munit_assert_memory_equal(9, ray_str_ptr(sp), "plain_one");

    ray_t* sd = ray_sym_str(id_dotted);
    munit_assert_ptr_not_null(sd);
    munit_assert_size(ray_str_len(sd), ==, 10);
    munit_assert_memory_equal(10, ray_str_ptr(sd), "alpha.beta");

    ray_t* sa = ray_sym_str(id_after);
    munit_assert_ptr_not_null(sa);
    munit_assert_size(ray_str_len(sa), ==, 9);
    munit_assert_memory_equal(9, ray_str_ptr(sa), "plain_two");

    /* And the dotted metadata must be reconstructed — otherwise env lookup
     * of a dotted name restored from disk would silently fall back to a
     * flat scan and fail. */
    munit_assert_true(ray_sym_is_dotted(id_dotted));
    const int64_t* segs = NULL;
    int n = ray_sym_segs(id_dotted, &segs);
    munit_assert_int(n, ==, 2);
    munit_assert_ptr_not_null(segs);
    munit_assert_int(ray_sym_find("alpha", 5), ==, segs[0]);
    munit_assert_int(ray_sym_find("beta", 4),  ==, segs[1]);

    /* Cleanup */
    remove(sym_path);
    remove(lk_path);
    return MUNIT_OK;
}

/* ---- sym_load_missing_file --------------------------------------------- */

static MunitResult test_sym_load_missing(const void* params, void* fixture) {
    (void)params; (void)fixture;

    ray_err_t err = ray_sym_load("/tmp/nonexistent_sym_file_xyz.sym");
    munit_assert_int(err, !=, RAY_OK);
    munit_assert_uint(ray_sym_count(), ==, 0);

    return MUNIT_OK;
}

/* ---- Suite definition -------------------------------------------------- */

static MunitTest sym_tests[] = {
    { "/init_destroy",    test_sym_init_destroy,    sym_setup, sym_teardown, 0, NULL },
    { "/intern_basic",    test_sym_intern_basic,    sym_setup, sym_teardown, 0, NULL },
    { "/intern_duplicate", test_sym_intern_duplicate, sym_setup, sym_teardown, 0, NULL },
    { "/find_existing",   test_sym_find_existing,   sym_setup, sym_teardown, 0, NULL },
    { "/find_missing",    test_sym_find_missing,    sym_setup, sym_teardown, 0, NULL },
    { "/str_roundtrip",   test_sym_str_roundtrip,   sym_setup, sym_teardown, 0, NULL },
    { "/count",           test_sym_count,           sym_setup, sym_teardown, 0, NULL },
    { "/many",            test_sym_many,            sym_setup, sym_teardown, 0, NULL },
    { "/bulk",            test_sym_bulk,            sym_setup, sym_teardown, 0, NULL },
    { "/save_load_roundtrip", test_sym_save_load_roundtrip, sym_setup, sym_teardown, 0, NULL },
    { "/save_append_only",    test_sym_save_append_only,    sym_setup, sym_teardown, 0, NULL },
    { "/load_corrupt",        test_sym_load_corrupt,        sym_setup, sym_teardown, 0, NULL },
    { "/load_truncated",      test_sym_load_truncated,      sym_setup, sym_teardown, 0, NULL },
    { "/load_missing",        test_sym_load_missing,        sym_setup, sym_teardown, 0, NULL },
    { "/dotted_detect",       test_sym_dotted_detect,       sym_setup, sym_teardown, 0, NULL },
    { "/dotted_segments",     test_sym_dotted_segments,     sym_setup, sym_teardown, 0, NULL },
    { "/dotted_triple",       test_sym_dotted_triple,       sym_setup, sym_teardown, 0, NULL },
    { "/dotted_idempotent",   test_sym_dotted_idempotent,   sym_setup, sym_teardown, 0, NULL },
    { "/dotted_empty_segment", test_sym_dotted_rejects_empty_segment, sym_setup, sym_teardown, 0, NULL },
    { "/save_load_dotted",    test_sym_save_load_dotted,     sym_setup, sym_teardown, 0, NULL },
    { "/load_legacy_dotted",  test_sym_load_legacy_dotted,   sym_setup, sym_teardown, 0, NULL },
    { "/rebuild_segments_contract", test_sym_rebuild_segments_contract, sym_setup, sym_teardown, 0, NULL },
    { "/dotted_reintern_retries_cache", test_sym_dotted_reintern_retries_cache, sym_setup, sym_teardown, 0, NULL },
    { "/intern_atomic_cache",  test_sym_intern_atomic_cache,  sym_setup, sym_teardown, 0, NULL },
    { "/intern_atomic_no_orphans", test_sym_intern_atomic_no_orphans, sym_setup, sym_teardown, 0, NULL },
    { "/intern_long_segments",     test_sym_intern_long_segments,     sym_setup, sym_teardown, 0, NULL },
    { "/bytes_upper_covers_arena", test_sym_bytes_upper_covers_arena, sym_setup, sym_teardown, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL },
};

MunitSuite test_sym_suite = {
    "/sym",
    sym_tests,
    NULL,
    0,
    0,
};
