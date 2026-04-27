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

/* Build a tiny target table {id, age} and bind it to `name_sym` in env. */
static ray_t* build_target_table(const char* name) {
    int64_t ids[]  = { 100, 200, 300 };
    int64_t ages[] = {  18,  25,  42 };
    ray_t* idcol  = make_i64_vec(ids,  3);
    ray_t* agecol = make_i64_vec(ages, 3);
    ray_t* tab = ray_table_new(2);
    int64_t id_sym  = ray_sym_intern("id",  2);
    int64_t age_sym = ray_sym_intern("age", 3);
    tab = ray_table_add_col(tab, id_sym,  idcol);
    tab = ray_table_add_col(tab, age_sym, agecol);
    ray_release(idcol);
    ray_release(agecol);
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
    { NULL, NULL, NULL, NULL },
};
