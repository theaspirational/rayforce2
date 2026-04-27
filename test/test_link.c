/*
 *   Copyright (c) 2025-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#define _GNU_SOURCE

#include "test.h"
#include <rayforce.h>
#include "mem/heap.h"
#include "vec/vec.h"
#include "table/sym.h"
#include "table/table.h"
#include "lang/eval.h"
#include "lang/env.h"
#define _POSIX_C_SOURCE 200809L

#include "ops/linkop.h"
#include "ops/idxop.h"
#include "store/col.h"
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

/* Tests run inside a runtime so the env is alive (link target lookup
 * needs that).  Use the same setup/teardown shape as test_lang.c. */
struct ray_runtime_s;
typedef struct ray_runtime_s ray_runtime_t;
extern ray_runtime_t* ray_runtime_create(int argc, char** argv);
extern void           ray_runtime_destroy(ray_runtime_t* rt);
extern ray_runtime_t* __RUNTIME;

static void link_setup(void)    { ray_runtime_create(0, NULL); }
static void link_teardown(void) { ray_runtime_destroy(__RUNTIME); }

/* ─── Helpers ──────────────────────────────────────────────────────── */

static ray_t* make_i64_vec(const int64_t* xs, int64_t n) {
    ray_t* v = ray_vec_new(RAY_I64, n);
    for (int64_t i = 0; i < n; i++) v = ray_vec_append(v, &xs[i]);
    return v;
}

/* Build a tiny target table {id, age, name(SYM), city(STR)} for tests. */
static ray_t* build_target_table(const char* name) {
    int64_t ids[]  = { 100, 200, 300 };
    int64_t ages[] = {  18,  25,  42 };
    ray_t* idcol  = make_i64_vec(ids,  3);
    ray_t* agecol = make_i64_vec(ages, 3);

    /* SYM column: alice, bob, carol. */
    ray_t* namecol = ray_sym_vec_new(RAY_SYM_W64, 3);
    int64_t s_alice = ray_sym_intern("alice", 5);
    int64_t s_bob   = ray_sym_intern("bob",   3);
    int64_t s_carol = ray_sym_intern("carol", 5);
    namecol = ray_vec_append(namecol, &s_alice);
    namecol = ray_vec_append(namecol, &s_bob);
    namecol = ray_vec_append(namecol, &s_carol);

    /* STR column: NYC, LA, SF — short enough to be inline (<=12 chars). */
    ray_t* citycol = ray_vec_new(RAY_STR, 3);
    citycol = ray_str_vec_append(citycol, "NYC", 3);
    citycol = ray_str_vec_append(citycol, "LA",  2);
    citycol = ray_str_vec_append(citycol, "SF",  2);

    ray_t* tab = ray_table_new(4);
    tab = ray_table_add_col(tab, ray_sym_intern("id",   2), idcol);
    tab = ray_table_add_col(tab, ray_sym_intern("age",  3), agecol);
    tab = ray_table_add_col(tab, ray_sym_intern("name", 4), namecol);
    tab = ray_table_add_col(tab, ray_sym_intern("city", 4), citycol);
    ray_release(idcol);
    ray_release(agecol);
    ray_release(namecol);
    ray_release(citycol);
    (void)name;
    return tab;
}

/* ─── Phase 1: storage round-trip ──────────────────────────────────── */

static test_result_t test_link_attach_basic(void) {
    int64_t rids[] = { 0, 1, 2, 1, 0 };
    ray_t* v = make_i64_vec(rids, 5);
    TEST_ASSERT_FALSE(v->attrs & RAY_ATTR_HAS_LINK);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);  /* env now holds 'custs' -> table */
    ray_release(target);

    ray_t* w = v;
    ray_t* r = ray_link_attach(&w, custs_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(r));
    TEST_ASSERT_TRUE(w->attrs & RAY_ATTR_HAS_LINK);
    TEST_ASSERT_EQ_I(w->link_target, custs_sym);
    TEST_ASSERT_TRUE(ray_link_has(w));
    TEST_ASSERT_EQ_I(ray_link_target_id(w), custs_sym);

    /* Detach. */
    w = ray_link_detach(&w);
    TEST_ASSERT_FALSE(w->attrs & RAY_ATTR_HAS_LINK);
    TEST_ASSERT_EQ_I(w->link_target, 0);

    ray_release(w);
    PASS();
}

static test_result_t test_link_reject_wrong_type(void) {
    ray_t* v = ray_vec_new(RAY_F64, 3);
    double zeros[] = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < 3; i++) v = ray_vec_append(v, &zeros[i]);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    ray_t* r = ray_link_attach(&w, custs_sym);
    TEST_ASSERT_TRUE(RAY_IS_ERR(r));
    TEST_ASSERT_FALSE(w->attrs & RAY_ATTR_HAS_LINK);

    if (RAY_IS_ERR(r)) ray_error_free(r);
    ray_release(w);
    PASS();
}

static test_result_t test_link_reject_unknown_target(void) {
    int64_t rids[] = { 0, 1, 2 };
    ray_t* v = make_i64_vec(rids, 3);

    int64_t bogus = ray_sym_intern("nope_no_table_here", 18);
    ray_t* w = v;
    ray_t* r = ray_link_attach(&w, bogus);
    TEST_ASSERT_TRUE(RAY_IS_ERR(r));
    TEST_ASSERT_FALSE(w->attrs & RAY_ATTR_HAS_LINK);

    if (RAY_IS_ERR(r)) ray_error_free(r);
    ray_release(w);
    PASS();
}

static test_result_t test_link_with_inline_nulls_promotes(void) {
    int64_t rids[] = { 0, 1, 2, 1, 0 };
    ray_t* v = make_i64_vec(rids, 5);
    TEST_ASSERT_EQ_I(ray_vec_set_null_checked(v, 1, true), RAY_OK);
    TEST_ASSERT_TRUE(v->attrs & RAY_ATTR_HAS_NULLS);
    TEST_ASSERT_FALSE(v->attrs & RAY_ATTR_NULLMAP_EXT);  /* inline initially */

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    ray_t* r = ray_link_attach(&w, custs_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(r));
    /* Inline nulls must have been promoted to ext to free up bytes 8-15. */
    TEST_ASSERT_TRUE(w->attrs & RAY_ATTR_NULLMAP_EXT);
    TEST_ASSERT_TRUE(w->attrs & RAY_ATTR_HAS_LINK);
    /* Null bit at row 1 is still readable. */
    TEST_ASSERT_TRUE(ray_vec_is_null(w, 1));

    ray_release(w);
    PASS();
}

static test_result_t test_link_mutation_preserves_link(void) {
    int64_t rids[] = { 0, 1, 2 };
    ray_t* v = make_i64_vec(rids, 3);
    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));
    TEST_ASSERT_TRUE(w->attrs & RAY_ATTR_HAS_LINK);

    /* Mutate row 1. */
    int64_t new_rid = 0;
    w = ray_vec_set(w, 1, &new_rid);
    TEST_ASSERT_FALSE(RAY_IS_ERR(w));
    /* Link must survive the mutation. */
    TEST_ASSERT_TRUE(w->attrs & RAY_ATTR_HAS_LINK);
    TEST_ASSERT_EQ_I(w->link_target, custs_sym);

    ray_release(w);
    PASS();
}

/* ─── Phase 2: deref ──────────────────────────────────────────────── */

static test_result_t test_link_deref_basic(void) {
    int64_t rids[] = { 2, 0, 1, 2 };
    ray_t* v = make_i64_vec(rids, 4);
    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));

    /* Deref the 'age' field — expected: [42, 18, 25, 42]. */
    int64_t age_sym = ray_sym_intern("age", 3);
    ray_t* result = ray_link_deref(w, age_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    TEST_ASSERT_EQ_I(result->len, 4);
    TEST_ASSERT_EQ_I(result->type, RAY_I64);
    int64_t* d = (int64_t*)ray_data(result);
    TEST_ASSERT_EQ_I(d[0], 42);
    TEST_ASSERT_EQ_I(d[1], 18);
    TEST_ASSERT_EQ_I(d[2], 25);
    TEST_ASSERT_EQ_I(d[3], 42);

    ray_release(result);
    ray_release(w);
    PASS();
}

static test_result_t test_link_deref_null_propagation(void) {
    int64_t rids[] = { 0, 1, 2 };
    ray_t* v = make_i64_vec(rids, 3);
    /* Mark row 1 of the link column as null. */
    TEST_ASSERT_EQ_I(ray_vec_set_null_checked(v, 1, true), RAY_OK);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));

    int64_t age_sym = ray_sym_intern("age", 3);
    ray_t* result = ray_link_deref(w, age_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    TEST_ASSERT_EQ_I(result->len, 3);
    TEST_ASSERT_FALSE(ray_vec_is_null(result, 0));
    TEST_ASSERT_TRUE (ray_vec_is_null(result, 1));   /* link[1] null -> result null */
    TEST_ASSERT_FALSE(ray_vec_is_null(result, 2));
    int64_t* d = (int64_t*)ray_data(result);
    TEST_ASSERT_EQ_I(d[0], 18);
    TEST_ASSERT_EQ_I(d[2], 42);

    ray_release(result);
    ray_release(w);
    PASS();
}

static test_result_t test_link_deref_oob_yields_null(void) {
    int64_t rids[] = { 0, 99, 2 };  /* 99 is out-of-bounds */
    ray_t* v = make_i64_vec(rids, 3);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));

    int64_t age_sym = ray_sym_intern("age", 3);
    ray_t* result = ray_link_deref(w, age_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    TEST_ASSERT_FALSE(ray_vec_is_null(result, 0));
    TEST_ASSERT_TRUE (ray_vec_is_null(result, 1));   /* 99 OOB -> null */
    TEST_ASSERT_FALSE(ray_vec_is_null(result, 2));

    ray_release(result);
    ray_release(w);
    PASS();
}

/* ─── Phase 3: persistence round-trip ─────────────────────────────── */

static test_result_t test_link_persistence_roundtrip(void) {
    int64_t rids[] = { 0, 1, 2, 1, 0 };
    ray_t* v = make_i64_vec(rids, 5);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));

    /* Save to a temp path. */
    char path[] = "/tmp/link_persist_test_XXXXXX";
    int fd = mkstemp(path);
    TEST_ASSERT_TRUE(fd >= 0);
    close(fd);
    ray_err_t err = ray_col_save(w, path);
    TEST_ASSERT_EQ_I(err, RAY_OK);

    /* Sidecar must exist. */
    char link_path[256];
    snprintf(link_path, sizeof link_path, "%s.link", path);
    FILE* lf = fopen(link_path, "rb");
    TEST_ASSERT_NOT_NULL(lf);
    fclose(lf);

    /* Load back and verify HAS_LINK + target. */
    ray_t* loaded = ray_col_load(path);
    TEST_ASSERT_FALSE(RAY_IS_ERR(loaded));
    TEST_ASSERT_TRUE(loaded->attrs & RAY_ATTR_HAS_LINK);
    TEST_ASSERT_EQ_I(loaded->link_target, custs_sym);
    TEST_ASSERT_EQ_I(loaded->len, 5);

    /* Deref still works. */
    int64_t age_sym = ray_sym_intern("age", 3);
    ray_t* result = ray_link_deref(loaded, age_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    int64_t* d = (int64_t*)ray_data(result);
    TEST_ASSERT_EQ_I(d[0], 18);
    TEST_ASSERT_EQ_I(d[1], 25);
    TEST_ASSERT_EQ_I(d[2], 42);

    unlink(path);
    unlink(link_path);
    ray_release(result);
    ray_release(loaded);
    ray_release(w);
    PASS();
}

/* ─── Sidecar must be picked up by ray_col_mmap too (splay-mmap path) ── */

static test_result_t test_link_mmap_loads_sidecar(void) {
    int64_t rids[] = { 0, 1, 2 };
    ray_t* v = make_i64_vec(rids, 3);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));

    char path[] = "/tmp/link_mmap_test_XXXXXX";
    int fd = mkstemp(path);
    TEST_ASSERT_TRUE(fd >= 0);
    close(fd);
    TEST_ASSERT_EQ_I(ray_col_save(w, path), RAY_OK);

    /* Load via mmap path (splayed-table mmap mode uses this). */
    ray_t* mapped = ray_col_mmap(path);
    TEST_ASSERT_FALSE(RAY_IS_ERR(mapped));
    TEST_ASSERT_TRUE(mapped->attrs & RAY_ATTR_HAS_LINK);
    TEST_ASSERT_EQ_I(mapped->link_target, custs_sym);

    char link_path[256];
    snprintf(link_path, sizeof link_path, "%s.link", path);
    unlink(path);
    unlink(link_path);
    ray_release(mapped);
    ray_release(w);
    PASS();
}

/* ─── Deref must NOT leak the target table ref ────────────────────── */

static test_result_t test_link_deref_no_target_leak(void) {
    int64_t rids[] = { 2, 0, 1, 2 };
    ray_t* v = make_i64_vec(rids, 4);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    /* Snapshot target rc after env-set + our local hold. */
    uint32_t rc_before = ray_atomic_load(&target->rc);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));
    int64_t age_sym = ray_sym_intern("age", 3);

    /* Run many derefs; rc on target must not grow. */
    for (int i = 0; i < 100; i++) {
        ray_t* r = ray_link_deref(w, age_sym);
        TEST_ASSERT_FALSE(RAY_IS_ERR(r));
        ray_release(r);
    }
    uint32_t rc_after = ray_atomic_load(&target->rc);
    TEST_ASSERT_EQ_U(rc_before, rc_after);

    ray_release(target);
    ray_release(w);
    PASS();
}

/* ─── Deref through SYM and STR target columns ────────────────────── */

static test_result_t test_link_deref_sym_target(void) {
    int64_t rids[] = { 2, 0, 1, 2 };
    ray_t* v = make_i64_vec(rids, 4);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));

    int64_t name_sym = ray_sym_intern("name", 4);
    ray_t* result = ray_link_deref(w, name_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    TEST_ASSERT_EQ_I(result->type, RAY_SYM);
    TEST_ASSERT_EQ_I(result->len, 4);
    /* Width must match the target's. */
    TEST_ASSERT_EQ_U((unsigned)(result->attrs & RAY_SYM_W_MASK), RAY_SYM_W64);

    /* Element values are global sym IDs since the target uses the global
     * sym table (sym_dict NULL) — directly comparable. */
    int64_t s_alice = ray_sym_intern("alice", 5);
    int64_t s_bob   = ray_sym_intern("bob",   3);
    int64_t s_carol = ray_sym_intern("carol", 5);
    int64_t* sd = (int64_t*)ray_data(result);
    TEST_ASSERT_EQ_I(sd[0], s_carol);   /* rid 2 */
    TEST_ASSERT_EQ_I(sd[1], s_alice);   /* rid 0 */
    TEST_ASSERT_EQ_I(sd[2], s_bob);     /* rid 1 */
    TEST_ASSERT_EQ_I(sd[3], s_carol);   /* rid 2 */

    ray_release(result);
    ray_release(w);
    PASS();
}

static test_result_t test_link_deref_str_target(void) {
    int64_t rids[] = { 0, 2, 1 };
    ray_t* v = make_i64_vec(rids, 3);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));

    int64_t city_sym = ray_sym_intern("city", 4);
    ray_t* result = ray_link_deref(w, city_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    TEST_ASSERT_EQ_I(result->type, RAY_STR);
    TEST_ASSERT_EQ_I(result->len, 3);

    /* Verify each row reads back as the expected string via ray_str_vec_get,
     * which resolves inline + pool storage transparently. */
    size_t L = 0;
    const char* s0 = ray_str_vec_get(result, 0, &L);
    TEST_ASSERT_EQ_U(L, 3);
    TEST_ASSERT_MEM_EQ(3, s0, "NYC");
    const char* s1 = ray_str_vec_get(result, 1, &L);
    TEST_ASSERT_EQ_U(L, 2);
    TEST_ASSERT_MEM_EQ(2, s1, "SF");
    const char* s2 = ray_str_vec_get(result, 2, &L);
    TEST_ASSERT_EQ_U(L, 2);
    TEST_ASSERT_MEM_EQ(2, s2, "LA");

    ray_release(result);
    ray_release(w);
    PASS();
}

/* ─── Slice over a linked parent must inherit the link ───────────── */

static test_result_t test_link_slice_inherits(void) {
    /* Parent: 5 row-IDs into custs.  Slice rows [1..4) -> 3 rows. */
    int64_t rids[] = { 0, 2, 1, 0, 2 };
    ray_t* parent = make_i64_vec(rids, 5);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = parent;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));
    TEST_ASSERT_TRUE(w->attrs & RAY_ATTR_HAS_LINK);

    /* Slice rows 1..4 (rids 2, 1, 0). */
    ray_t* slice = ray_vec_slice(w, 1, 3);
    TEST_ASSERT_FALSE(RAY_IS_ERR(slice));
    TEST_ASSERT_TRUE(slice->attrs & RAY_ATTR_SLICE);
    /* Slice's own attrs do NOT carry HAS_LINK — the bit is inherited
     * transparently via slice_parent at deref time. */
    TEST_ASSERT_FALSE(slice->attrs & RAY_ATTR_HAS_LINK);

    /* But ray_link_has reaches through. */
    TEST_ASSERT_TRUE(ray_link_has(slice));
    TEST_ASSERT_EQ_I(ray_link_target_id(slice), custs_sym);

    /* (.col.target slice) builtin must also report the parent's target,
     * not garbage from reading the slice's bytes 8-15 as a sym ID. */
    ray_t* tgt_sym = ray_col_target_fn(slice);
    TEST_ASSERT_NOT_NULL(tgt_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(tgt_sym));
    TEST_ASSERT_EQ_I(tgt_sym->type, -RAY_SYM);
    TEST_ASSERT_EQ_I(tgt_sym->i64, custs_sym);
    ray_release(tgt_sym);

    /* Deref must work and produce the right values for rids [2,1,0]. */
    int64_t age_sym = ray_sym_intern("age", 3);
    ray_t* result = ray_link_deref(slice, age_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    TEST_ASSERT_EQ_I(result->len, 3);
    int64_t* d = (int64_t*)ray_data(result);
    /* target.age = [18, 25, 42]; rids[1..4] = [2, 1, 0]. */
    TEST_ASSERT_EQ_I(d[0], 42);
    TEST_ASSERT_EQ_I(d[1], 25);
    TEST_ASSERT_EQ_I(d[2], 18);

    ray_release(result);
    ray_release(slice);
    ray_release(w);
    PASS();
}

/* ─── Empty link column ──────────────────────────────────────────── */

static test_result_t test_link_deref_empty_link(void) {
    ray_t* v = ray_vec_new(RAY_I64, 0);  /* zero-length link */
    v->len = 0;

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));
    TEST_ASSERT_EQ_I(w->len, 0);

    int64_t age_sym = ray_sym_intern("age", 3);
    ray_t* result = ray_link_deref(w, age_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    TEST_ASSERT_EQ_I(result->len, 0);

    ray_release(result);
    ray_release(w);
    PASS();
}

/* ─── Probe miss (field not on target) returns NULL, not error ─────── */

static test_result_t test_link_deref_unknown_field(void) {
    int64_t rids[] = { 0, 1, 2 };
    ray_t* v = make_i64_vec(rids, 3);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));

    /* Field 'phone' doesn't exist on the target; deref must return NULL,
     * not a hard error.  Letting the dotted-walk's method-dispatch
     * fallback (env.c:208-247) try other resolutions. */
    int64_t phone_sym = ray_sym_intern("phone", 5);
    ray_t* result = ray_link_deref(w, phone_sym);
    TEST_ASSERT_NULL(result);

    ray_release(w);
    PASS();
}

/* ─── Save→load with no link must NOT spuriously attach one ───────── */

static test_result_t test_link_save_no_link_no_sidecar(void) {
    int64_t xs[] = { 1, 2, 3 };
    ray_t* v = make_i64_vec(xs, 3);
    /* Plain int column — no link attached. */
    TEST_ASSERT_FALSE(v->attrs & RAY_ATTR_HAS_LINK);

    char path[] = "/tmp/link_no_sidecar_XXXXXX";
    int fd = mkstemp(path);
    TEST_ASSERT_TRUE(fd >= 0);
    close(fd);
    TEST_ASSERT_EQ_I(ray_col_save(v, path), RAY_OK);

    /* No `.link` sidecar should exist. */
    char link_path[256];
    snprintf(link_path, sizeof link_path, "%s.link", path);
    FILE* lf = fopen(link_path, "rb");
    if (lf) { fclose(lf); FAILF("unexpected sidecar at %s", link_path); }

    ray_t* loaded = ray_col_load(path);
    TEST_ASSERT_FALSE(RAY_IS_ERR(loaded));
    TEST_ASSERT_FALSE(loaded->attrs & RAY_ATTR_HAS_LINK);

    unlink(path);
    ray_release(loaded);
    ray_release(v);
    PASS();
}

/* ─── Slice + narrow-width sym target ────────────────────────────── */

static test_result_t test_link_deref_sym_slice_w8(void) {
    /* Build a narrow W8 sym column with 4 rows, then place a slice of
     * the last 3 rows into the table.  Without slice-aware width
     * resolution, the deref reads slice attrs (which carry W_MASK = 0
     * = W8) AND ray_data() multiplies slice_offset by 8 (W64 default
     * in ray_type_sizes), giving a misaligned base.  This test catches
     * both errors. */
    int64_t s_a = ray_sym_intern("aa", 2);
    int64_t s_b = ray_sym_intern("bb", 2);
    int64_t s_c = ray_sym_intern("cc", 2);
    int64_t s_d = ray_sym_intern("dd", 2);

    /* W8 sym vec: ['aa, 'bb, 'cc, 'dd] but stored as 1-byte indices.
     * ray_sym_vec_new + ray_vec_append handles the width-narrow path. */
    ray_t* full = ray_sym_vec_new(RAY_SYM_W8, 4);
    full = ray_vec_append(full, &s_a);
    full = ray_vec_append(full, &s_b);
    full = ray_vec_append(full, &s_c);
    full = ray_vec_append(full, &s_d);
    TEST_ASSERT_EQ_U((unsigned)(full->attrs & RAY_SYM_W_MASK), RAY_SYM_W8);

    /* Slice to rows [1..4): ['bb, 'cc, 'dd]. */
    ray_t* slice = ray_vec_slice(full, 1, 3);
    TEST_ASSERT_FALSE(RAY_IS_ERR(slice));
    TEST_ASSERT_TRUE(slice->attrs & RAY_ATTR_SLICE);

    /* Build target table with the SLICE as the 'name' column. */
    int64_t ids[]  = { 100, 200, 300 };
    ray_t* idcol = make_i64_vec(ids, 3);
    ray_t* tab = ray_table_new(2);
    tab = ray_table_add_col(tab, ray_sym_intern("id",   2), idcol);
    tab = ray_table_add_col(tab, ray_sym_intern("name", 4), slice);
    ray_release(idcol);
    ray_release(slice);
    ray_release(full);

    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, tab);
    ray_release(tab);

    /* Link: rids point at the slice (not the full vec). */
    int64_t rids[] = { 0, 2, 1 };
    ray_t* v = make_i64_vec(rids, 3);
    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));

    int64_t name_sym = ray_sym_intern("name", 4);
    ray_t* result = ray_link_deref(w, name_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    TEST_ASSERT_EQ_I(result->type, RAY_SYM);
    /* Result width must mirror the slice's parent width. */
    TEST_ASSERT_EQ_U((unsigned)(result->attrs & RAY_SYM_W_MASK), RAY_SYM_W8);

    /* Read elements back through ray_vec_get, which is width-aware,
     * and intern-compare with the expected names. */
    uint8_t* d = (uint8_t*)ray_data(result);
    TEST_ASSERT_EQ_U(d[0], (uint8_t)s_b);  /* slice[0] = parent[1] = 'bb */
    TEST_ASSERT_EQ_U(d[1], (uint8_t)s_d);  /* slice[2] = parent[3] = 'dd */
    TEST_ASSERT_EQ_U(d[2], (uint8_t)s_c);  /* slice[1] = parent[2] = 'cc */

    ray_release(result);
    ray_release(w);
    PASS();
}

/* ─── Phase 4: coexistence with HAS_INDEX ─────────────────────────── */

static test_result_t test_link_coexists_with_index(void) {
    int64_t rids[] = { 0, 1, 2, 1, 0 };
    ray_t* v = make_i64_vec(rids, 5);

    ray_t* target = build_target_table("custs");
    int64_t custs_sym = ray_sym_intern("custs", 5);
    ray_env_set(custs_sym, target);
    ray_release(target);

    /* Attach link first, then index. */
    ray_t* w = v;
    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_link_attach(&w, custs_sym)));
    TEST_ASSERT_TRUE(w->attrs & RAY_ATTR_HAS_LINK);

    TEST_ASSERT_FALSE(RAY_IS_ERR(ray_index_attach_zone(&w)));
    TEST_ASSERT_TRUE(w->attrs & RAY_ATTR_HAS_INDEX);
    TEST_ASSERT_TRUE(w->attrs & RAY_ATTR_HAS_LINK);
    TEST_ASSERT_EQ_I(w->link_target, custs_sym);

    /* Drop the index — link must remain. */
    ray_t* x = w;
    ray_index_drop(&x);
    TEST_ASSERT_FALSE(x->attrs & RAY_ATTR_HAS_INDEX);
    TEST_ASSERT_TRUE (x->attrs & RAY_ATTR_HAS_LINK);
    TEST_ASSERT_EQ_I(x->link_target, custs_sym);

    /* Deref still works after index drop. */
    int64_t age_sym = ray_sym_intern("age", 3);
    ray_t* result = ray_link_deref(x, age_sym);
    TEST_ASSERT_FALSE(RAY_IS_ERR(result));
    TEST_ASSERT_EQ_I(result->len, 5);
    int64_t* d = (int64_t*)ray_data(result);
    TEST_ASSERT_EQ_I(d[0], 18);
    TEST_ASSERT_EQ_I(d[1], 25);
    TEST_ASSERT_EQ_I(d[2], 42);

    ray_release(result);
    ray_release(x);
    PASS();
}

const test_entry_t link_entries[] = {
    { "link/attach_basic",                   test_link_attach_basic,                  link_setup, link_teardown },
    { "link/reject_wrong_type",              test_link_reject_wrong_type,             link_setup, link_teardown },
    { "link/reject_unknown_target",          test_link_reject_unknown_target,         link_setup, link_teardown },
    { "link/with_inline_nulls_promotes",     test_link_with_inline_nulls_promotes,    link_setup, link_teardown },
    { "link/mutation_preserves_link",        test_link_mutation_preserves_link,       link_setup, link_teardown },
    { "link/deref_basic",                    test_link_deref_basic,                   link_setup, link_teardown },
    { "link/deref_null_propagation",         test_link_deref_null_propagation,        link_setup, link_teardown },
    { "link/deref_oob_yields_null",          test_link_deref_oob_yields_null,         link_setup, link_teardown },
    { "link/persistence_roundtrip",          test_link_persistence_roundtrip,         link_setup, link_teardown },
    { "link/coexists_with_index",            test_link_coexists_with_index,           link_setup, link_teardown },
    { "link/mmap_loads_sidecar",             test_link_mmap_loads_sidecar,            link_setup, link_teardown },
    { "link/deref_no_target_leak",           test_link_deref_no_target_leak,          link_setup, link_teardown },
    { "link/deref_sym_target",               test_link_deref_sym_target,              link_setup, link_teardown },
    { "link/deref_str_target",               test_link_deref_str_target,              link_setup, link_teardown },
    { "link/deref_sym_slice_w8",             test_link_deref_sym_slice_w8,            link_setup, link_teardown },
    { "link/deref_empty_link",               test_link_deref_empty_link,              link_setup, link_teardown },
    { "link/deref_unknown_field",            test_link_deref_unknown_field,           link_setup, link_teardown },
    { "link/save_no_link_no_sidecar",        test_link_save_no_link_no_sidecar,       link_setup, link_teardown },
    { "link/slice_inherits",                 test_link_slice_inherits,                link_setup, link_teardown },
    { NULL, NULL, NULL, NULL },
};
