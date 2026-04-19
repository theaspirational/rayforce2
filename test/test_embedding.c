/*
 *   Copyright (c) 2025-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 *
 *   Tests for Rayfall embedding / vector-similarity / HNSW builtins.
 *   Names follow pgvector: cos-dist, l2-dist, inner-prod, plus norm, knn,
 *   hnsw-build, ann, hnsw-save, hnsw-load, hnsw-info, hnsw-free.
 */

#include "munit.h"
#include <rayforce.h>
#include "mem/heap.h"
#include "table/sym.h"
#include "lang/eval.h"
#include "lang/format.h"
#include "store/hnsw.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct ray_runtime_s;
typedef struct ray_runtime_s ray_runtime_t;
extern ray_runtime_t* ray_runtime_create(int argc, char** argv);
extern void           ray_runtime_destroy(ray_runtime_t* rt);
extern ray_runtime_t* __RUNTIME;

/* Small epsilon for F64 comparisons. */
#define EPS 1e-6

static void* emb_setup(const void* params, void* user_data) {
    (void)params; (void)user_data;
    ray_runtime_create(0, NULL);
    return NULL;
}

static void emb_teardown(void* fixture) {
    (void)fixture;
    ray_runtime_destroy(__RUNTIME);
}

/* Evaluate a Rayfall expression and return its F64 atom value. */
static double eval_f64(const char* expr) {
    ray_t* r = ray_eval_str(expr);
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_int(r->type, ==, -RAY_F64);
    double v = r->f64;
    ray_release(r);
    return v;
}

/* Evaluate a Rayfall expression and return its I64 atom value. */
static int64_t eval_i64(const char* expr) {
    ray_t* r = ray_eval_str(expr);
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_true(r->type == -RAY_I64 || r->type == -RAY_I32 || r->type == -RAY_I16);
    int64_t v;
    switch (r->type) {
        case -RAY_I64: v = r->i64; break;
        case -RAY_I32: v = r->i32; break;
        default:       v = r->i16; break;
    }
    ray_release(r);
    return v;
}

/* ============ Direct metric builtins — scalar results ============ */

static MunitResult test_cos_dist_scalar(const void* p, void* f) {
    (void)p; (void)f;
    /* Orthogonal vectors: cos(90°) = 0 → distance 1. */
    munit_assert_double_equal(eval_f64("(cos-dist [1.0 0.0] [0.0 1.0])"), 1.0, 6);
    /* Parallel identical: cos(0°) = 1 → distance 0. */
    munit_assert_double_equal(eval_f64("(cos-dist [1.0 1.0] [1.0 1.0])"), 0.0, 6);
    /* Opposite: cos(180°) = -1 → distance 2. */
    munit_assert_double_equal(eval_f64("(cos-dist [1.0 0.0] [-1.0 0.0])"), 2.0, 6);
    /* 45°: cos = √2/2 → distance 1 - √2/2 ≈ 0.2929. */
    double v = eval_f64("(cos-dist [1.0 0.0] [1.0 1.0])");
    munit_assert_double_equal(v, 1.0 - 1.0/sqrt(2.0), 6);
    return MUNIT_OK;
}

static MunitResult test_l2_dist_scalar(const void* p, void* f) {
    (void)p; (void)f;
    munit_assert_double_equal(eval_f64("(l2-dist [0.0 0.0] [3.0 4.0])"), 5.0, 6);
    munit_assert_double_equal(eval_f64("(l2-dist [1.0 1.0] [1.0 1.0])"), 0.0, 6);
    munit_assert_double_equal(eval_f64("(l2-dist [0.0 0.0 0.0] [1.0 2.0 2.0])"), 3.0, 6);
    return MUNIT_OK;
}

static MunitResult test_inner_prod_scalar(const void* p, void* f) {
    (void)p; (void)f;
    munit_assert_double_equal(eval_f64("(inner-prod [1.0 2.0 3.0] [4.0 5.0 6.0])"), 32.0, 6);
    munit_assert_double_equal(eval_f64("(inner-prod [1.0 0.0] [0.0 1.0])"), 0.0, 6);
    munit_assert_double_equal(eval_f64("(inner-prod [2.0 3.0] [-1.0 -1.0])"), -5.0, 6);
    return MUNIT_OK;
}

static MunitResult test_norm_scalar(const void* p, void* f) {
    (void)p; (void)f;
    munit_assert_double_equal(eval_f64("(norm [3.0 4.0])"), 5.0, 6);
    munit_assert_double_equal(eval_f64("(norm [0.0 0.0])"), 0.0, 6);
    munit_assert_double_equal(eval_f64("(norm [1.0 2.0 2.0])"), 3.0, 6);
    return MUNIT_OK;
}

/* ============ Errors on malformed inputs ============ */

static MunitResult test_metric_length_mismatch(const void* p, void* f) {
    (void)p; (void)f;
    ray_t* r = ray_eval_str("(cos-dist [1.0 2.0 3.0] [1.0 2.0])");
    munit_assert_ptr_not_null(r);
    munit_assert_true(RAY_IS_ERR(r));
    return MUNIT_OK;
}

static MunitResult test_metric_type_error(const void* p, void* f) {
    (void)p; (void)f;
    /* String isn't a numeric vec. */
    ray_t* r = ray_eval_str("(l2-dist \"hello\" [1.0 2.0])");
    munit_assert_ptr_not_null(r);
    munit_assert_true(RAY_IS_ERR(r));
    return MUNIT_OK;
}

/* ============ LIST-column metrics — vector results ============ */

static MunitResult test_cos_dist_list(const void* p, void* f) {
    (void)p; (void)f;
    /* Query [1,0,0] against rows [1,0,0], [0,1,0], [1,1,0]: distances 0, 1, 1-1/√2. */
    double d0 = eval_f64("(at (cos-dist (list [1.0 0.0 0.0] [0.0 1.0 0.0] [1.0 1.0 0.0]) [1.0 0.0 0.0]) 0)");
    double d1 = eval_f64("(at (cos-dist (list [1.0 0.0 0.0] [0.0 1.0 0.0] [1.0 1.0 0.0]) [1.0 0.0 0.0]) 1)");
    double d2 = eval_f64("(at (cos-dist (list [1.0 0.0 0.0] [0.0 1.0 0.0] [1.0 1.0 0.0]) [1.0 0.0 0.0]) 2)");
    munit_assert_double_equal(d0, 0.0, 6);
    munit_assert_double_equal(d1, 1.0, 6);
    munit_assert_double_equal(d2, 1.0 - 1.0/sqrt(2.0), 6);
    return MUNIT_OK;
}

static MunitResult test_l2_dist_list(const void* p, void* f) {
    (void)p; (void)f;
    double d0 = eval_f64("(at (l2-dist (list [0.0 0.0] [3.0 4.0] [6.0 8.0]) [0.0 0.0]) 0)");
    double d1 = eval_f64("(at (l2-dist (list [0.0 0.0] [3.0 4.0] [6.0 8.0]) [0.0 0.0]) 1)");
    double d2 = eval_f64("(at (l2-dist (list [0.0 0.0] [3.0 4.0] [6.0 8.0]) [0.0 0.0]) 2)");
    munit_assert_double_equal(d0, 0.0, 6);
    munit_assert_double_equal(d1, 5.0, 6);
    munit_assert_double_equal(d2, 10.0, 6);
    return MUNIT_OK;
}

static MunitResult test_inner_prod_list(const void* p, void* f) {
    (void)p; (void)f;
    double d0 = eval_f64("(at (inner-prod (list [1.0 0.0] [0.0 1.0] [1.0 1.0]) [1.0 2.0]) 0)");
    double d1 = eval_f64("(at (inner-prod (list [1.0 0.0] [0.0 1.0] [1.0 1.0]) [1.0 2.0]) 1)");
    double d2 = eval_f64("(at (inner-prod (list [1.0 0.0] [0.0 1.0] [1.0 1.0]) [1.0 2.0]) 2)");
    munit_assert_double_equal(d0, 1.0, 6);
    munit_assert_double_equal(d1, 2.0, 6);
    munit_assert_double_equal(d2, 3.0, 6);
    return MUNIT_OK;
}

static MunitResult test_norm_list(const void* p, void* f) {
    (void)p; (void)f;
    double d0 = eval_f64("(at (norm (list [3.0 4.0] [5.0 12.0] [8.0 15.0])) 0)");
    double d1 = eval_f64("(at (norm (list [3.0 4.0] [5.0 12.0] [8.0 15.0])) 1)");
    double d2 = eval_f64("(at (norm (list [3.0 4.0] [5.0 12.0] [8.0 15.0])) 2)");
    munit_assert_double_equal(d0, 5.0, 6);
    munit_assert_double_equal(d1, 13.0, 6);
    munit_assert_double_equal(d2, 17.0, 6);
    return MUNIT_OK;
}

/* ============ Brute-force knn — correctness per metric ============ */

/* Helper: common 5-row test column built in-script. */
#define FIVE_VECS "(list [1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0] [1.0 1.0 0.0] [1.0 0.0 1.0])"

static MunitResult test_knn_cosine(const void* p, void* f) {
    (void)p; (void)f;
    /* Top-3 nearest to [1,0,0] by cosine distance: row 0 (dist 0), then rows 3 and 4 (both 1-1/√2). */
    munit_assert_int(eval_i64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'cosine) '_rowid) 0)"), ==, 0);
    double d0 = eval_f64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'cosine) '_dist) 0)");
    munit_assert_double_equal(d0, 0.0, 6);
    double d1 = eval_f64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'cosine) '_dist) 1)");
    munit_assert_double_equal(d1, 1.0 - 1.0/sqrt(2.0), 6);
    double d2 = eval_f64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'cosine) '_dist) 2)");
    munit_assert_double_equal(d2, 1.0 - 1.0/sqrt(2.0), 6);
    return MUNIT_OK;
}

static MunitResult test_knn_l2(const void* p, void* f) {
    (void)p; (void)f;
    munit_assert_int(eval_i64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'l2) '_rowid) 0)"), ==, 0);
    double d0 = eval_f64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'l2) '_dist) 0)");
    double d1 = eval_f64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'l2) '_dist) 1)");
    double d2 = eval_f64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'l2) '_dist) 2)");
    munit_assert_double_equal(d0, 0.0, 6);
    munit_assert_double_equal(d1, 1.0, 6);
    munit_assert_double_equal(d2, 1.0, 6);
    return MUNIT_OK;
}

static MunitResult test_knn_ip(const void* p, void* f) {
    (void)p; (void)f;
    /* Inner-product metric sorts by -dot; all vecs that match with dot=1 tie at -1. */
    double d0 = eval_f64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'ip) '_dist) 0)");
    double d1 = eval_f64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'ip) '_dist) 1)");
    double d2 = eval_f64("(at (at (knn " FIVE_VECS " [1.0 0.0 0.0] 3 'ip) '_dist) 2)");
    munit_assert_double_equal(d0, -1.0, 6);
    munit_assert_double_equal(d1, -1.0, 6);
    munit_assert_double_equal(d2, -1.0, 6);
    return MUNIT_OK;
}

/* ============ HNSW lifecycle ============ */

static MunitResult test_hnsw_info_and_metric(const void* p, void* f) {
    (void)p; (void)f;
    /* Build with 'l2 and verify hnsw-info reports dim, metric, n-nodes. */
    ray_eval_str("(set __idx (hnsw-build " FIVE_VECS " 'l2 4 50))");
    munit_assert_int(eval_i64("(at (hnsw-info __idx) 'dim)"),     ==, 3);
    munit_assert_int(eval_i64("(at (hnsw-info __idx) 'nrows)"), ==, 5);
    munit_assert_int(eval_i64("(at (hnsw-info __idx) 'M)"),       ==, 4);
    munit_assert_int(eval_i64("(at (hnsw-info __idx) 'efc)"), ==, 50);
    /* metric is a SYM atom; verify via equality with 'l2. */
    ray_t* r = ray_eval_str("(== (at (hnsw-info __idx) 'metric) 'l2)");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_int(r->type, ==, -RAY_BOOL);
    munit_assert_uint(r->b8, ==, 1);
    ray_release(r);
    ray_eval_str("(hnsw-free __idx)");
    return MUNIT_OK;
}

static MunitResult test_ann_topk(const void* p, void* f) {
    (void)p; (void)f;
    ray_eval_str("(set __idx (hnsw-build " FIVE_VECS " 'cosine 8 100))");
    /* Row 0 must be the closest (identity). */
    munit_assert_int(eval_i64("(at (at (ann __idx [1.0 0.0 0.0] 1) '_rowid) 0)"), ==, 0);
    munit_assert_double_equal(eval_f64("(at (at (ann __idx [1.0 0.0 0.0] 1) '_dist) 0)"), 0.0, 6);
    ray_eval_str("(hnsw-free __idx)");
    return MUNIT_OK;
}

/* Build two indexes with the same data, one per metric (l2, cosine), verify
 * that the row-0-self query gives distance 0 in both. */
static MunitResult test_ann_each_metric(const void* p, void* f) {
    (void)p; (void)f;
    ray_eval_str("(set __idx_l2 (hnsw-build " FIVE_VECS " 'l2 8 100))");
    ray_eval_str("(set __idx_co (hnsw-build " FIVE_VECS " 'cosine 8 100))");
    munit_assert_double_equal(eval_f64("(at (at (ann __idx_l2 [1.0 0.0 0.0] 1) '_dist) 0)"), 0.0, 6);
    munit_assert_double_equal(eval_f64("(at (at (ann __idx_co [1.0 0.0 0.0] 1) '_dist) 0)"), 0.0, 6);
    ray_eval_str("(hnsw-free __idx_l2)");
    ray_eval_str("(hnsw-free __idx_co)");
    return MUNIT_OK;
}

/* Recall check — build a random 200×16 index, measure recall@5 vs exact knn. */
static MunitResult test_ann_recall(const void* p, void* f) {
    (void)p; (void)f;
    /* Use C random with fixed seed for reproducibility. */
    srand(42);
    const int N = 200, D = 16, K = 5, QUERIES = 10;

    /* Build an RAY_LIST of RAY_F32 vectors. */
    ray_t* list = ray_list_new(N);
    munit_assert_ptr_not_null(list);
    for (int i = 0; i < N; i++) {
        ray_t* v = ray_vec_new(RAY_F32, D);
        v->len = D;
        float* d = (float*)ray_data(v);
        for (int j = 0; j < D; j++) d[j] = (float)rand() / (float)RAND_MAX - 0.5f;
        list = ray_list_append(list, v);
        ray_release(v);
    }
    munit_assert_int(list->len, ==, N);

    /* Bind list as a global variable so eval_str can reach it. */
    int64_t sym = ray_sym_intern("__recall_list", 13);
    munit_assert_int(ray_env_set(sym, list), ==, RAY_OK);
    ray_release(list);

    ray_eval_str("(set __recall_idx (hnsw-build __recall_list 'l2 16 100))");

    int hits = 0, total = 0;
    for (int q = 0; q < QUERIES; q++) {
        /* Pick row q as the query (so exact match is guaranteed at position 0). */
        char expr[256];
        snprintf(expr, sizeof(expr),
                 "(at (at (ann __recall_idx (at __recall_list %d) %d) '_rowid) 0)", q, K);
        int64_t top_ann = eval_i64(expr);

        snprintf(expr, sizeof(expr),
                 "(at (at (knn __recall_list (at __recall_list %d) %d 'l2) '_rowid) 0)", q, K);
        int64_t top_exact = eval_i64(expr);

        if (top_ann == top_exact) hits++;
        total++;
    }
    ray_eval_str("(hnsw-free __recall_idx)");
    /* Self-query: every top-1 must be the row itself. */
    munit_assert_int(hits, ==, total);
    return MUNIT_OK;
}

/* ============ Persistence round-trip ============ */

static MunitResult test_hnsw_save_load(const void* p, void* f) {
    (void)p; (void)f;
    const char* dir = "/tmp/ray_hnsw_test_idx";
    /* Build and save in 'ip metric. */
    ray_eval_str("(set __idx (hnsw-build " FIVE_VECS " 'ip 4 50))");
    char expr[512];
    snprintf(expr, sizeof(expr), "(hnsw-save __idx \"%s\")", dir);
    ray_t* saved = ray_eval_str(expr);
    munit_assert_false(RAY_IS_ERR(saved));
    ray_eval_str("(hnsw-free __idx)");

    /* Load from disk, confirm metric still 'ip, then query. */
    snprintf(expr, sizeof(expr), "(set __idx2 (hnsw-load \"%s\"))", dir);
    ray_t* loaded = ray_eval_str(expr);
    munit_assert_false(RAY_IS_ERR(loaded));

    ray_t* metric_eq = ray_eval_str("(== (at (hnsw-info __idx2) 'metric) 'ip)");
    munit_assert_false(RAY_IS_ERR(metric_eq));
    munit_assert_int(metric_eq->type, ==, -RAY_BOOL);
    munit_assert_uint(metric_eq->b8, ==, 1);
    ray_release(metric_eq);

    /* ann on a known vector should still return row 0 first. */
    munit_assert_int(eval_i64("(at (at (ann __idx2 [1.0 0.0 0.0] 1) '_rowid) 0)"), ==, 0);
    ray_eval_str("(hnsw-free __idx2)");
    return MUNIT_OK;
}

/* ============ Handle lifecycle — double-free returns type error ============ */

static MunitResult test_hnsw_free_twice(const void* p, void* f) {
    (void)p; (void)f;
    ray_eval_str("(set __idx (hnsw-build " FIVE_VECS " 'cosine 4 50))");
    ray_t* r1 = ray_eval_str("(hnsw-free __idx)");
    munit_assert_false(RAY_IS_ERR(r1));
    /* Second free: handle attr is cleared, so unwrap fails → type error. */
    ray_t* r2 = ray_eval_str("(hnsw-free __idx)");
    munit_assert_true(RAY_IS_ERR(r2));
    return MUNIT_OK;
}

/* Rebinding a handle variable must not leak the previously-bound index.
 * ASan would report the leak on runtime teardown if rc→0 cleanup missed it. */
static MunitResult test_hnsw_rebind_no_leak(const void* p, void* f) {
    (void)p; (void)f;
    ray_eval_str("(set __idx (hnsw-build " FIVE_VECS " 'cosine 4 50))");
    /* Rebind: the old handle's rc hits zero, ray_free runs, the rc-cleanup
     * hook in src/mem/heap.c frees the underlying ray_hnsw_t. */
    ray_eval_str("(set __idx (hnsw-build " FIVE_VECS " 'l2 4 50))");
    ray_eval_str("(set __idx (hnsw-build " FIVE_VECS " 'ip 4 50))");
    ray_eval_str("(hnsw-free __idx)");
    return MUNIT_OK;
}

/* Handle produced inline (never bound) must not leak after the enclosing
 * expression completes. */
static MunitResult test_hnsw_inline_no_leak(const void* p, void* f) {
    (void)p; (void)f;
    ray_t* r = ray_eval_str(
        "(at (at (ann (hnsw-build " FIVE_VECS " 'l2 4 50) [1.0 0.0 0.0] 1) '_rowid) 0)");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    ray_release(r);
    return MUNIT_OK;
}

/* Handle left bound to a variable at runtime teardown must not leak.
 * ASan reports any leaked ray_hnsw_t at process exit. */
static MunitResult test_hnsw_teardown_no_leak(const void* p, void* f) {
    (void)p; (void)f;
    ray_eval_str("(set __leak_idx (hnsw-build " FIVE_VECS " 'cosine 4 50))");
    return MUNIT_OK;
}

/* Exercise the generic copy path directly.  ray_alloc_copy must deep-clone
 * the underlying ray_hnsw_t so the copy is a full, independent owner with
 * the same index contents — and freeing either does not disturb the other. */
static MunitResult test_hnsw_handle_cow(const void* p, void* f) {
    (void)p; (void)f;
    ray_eval_str("(set __idx (hnsw-build " FIVE_VECS " 'cosine 4 50))");
    ray_t* h = ray_env_get(ray_sym_intern("__idx", 5));
    munit_assert_ptr_not_null(h);
    munit_assert_int(h->type, ==, -RAY_I64);
    munit_assert_true(h->attrs & RAY_ATTR_HNSW);
    int64_t orig_ptr = h->i64;
    munit_assert_int(orig_ptr, !=, 0);

    /* Copy via the same path ray_cow uses.  The copy must OWN its own
     * ray_hnsw_t (deep-cloned from the source), not alias the original. */
    ray_t* copy = ray_alloc_copy(h);
    munit_assert_ptr_not_null(copy);
    munit_assert_int(copy->type, ==, -RAY_I64);
    munit_assert_true(copy->attrs & RAY_ATTR_HNSW);
    munit_assert_int(copy->i64, !=, 0);
    munit_assert_int(copy->i64, !=, orig_ptr);

    /* Both indexes should answer the same query — same data, independent memory. */
    ray_hnsw_t* src = (ray_hnsw_t*)(uintptr_t)orig_ptr;
    ray_hnsw_t* dup = (ray_hnsw_t*)(uintptr_t)copy->i64;
    munit_assert_int(dup->n_nodes, ==, src->n_nodes);
    munit_assert_int(dup->dim,     ==, src->dim);
    munit_assert_int(dup->metric,  ==, src->metric);

    float q[3] = {1.0f, 0.0f, 0.0f};
    int64_t ids_src[1], ids_dup[1];
    double  ds_src[1], ds_dup[1];
    munit_assert_int(ray_hnsw_search(src, q, 3, 1, 50, ids_src, ds_src), ==, 1);
    munit_assert_int(ray_hnsw_search(dup, q, 3, 1, 50, ids_dup, ds_dup), ==, 1);
    munit_assert_int(ids_src[0], ==, ids_dup[0]);
    munit_assert_double_equal(ds_src[0], ds_dup[0], 6);

    /* Releasing the copy frees the cloned index but leaves the original intact. */
    ray_release(copy);
    munit_assert_true(h->attrs & RAY_ATTR_HNSW);
    munit_assert_int(h->i64, ==, orig_ptr);
    munit_assert_int(eval_i64("(at (hnsw-info __idx) 'nrows)"), ==, 5);

    ray_eval_str("(hnsw-free __idx)");
    return MUNIT_OK;
}

/* ============ select ... nearest ... take — Phase 2 integration ============ */

/* Helper: build a 5-row test table with id / score / emb columns.  Runs
 * in the Rayfall env, then returns nothing.  The subsequent eval_* calls
 * reference `__docs` by name. */
static void build_docs(void) {
    ray_eval_str(
        "(set __docs (table [id score emb] "
        "  (list [0 1 2 3 4] "
        "        [0.9 0.2 0.8 0.1 0.7] "
        "        (list [1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0] "
        "              [1.0 1.0 0.0] [1.0 0.0 1.0]))))");
}

static MunitResult test_select_nearest_knn(const void* p, void* f) {
    (void)p; (void)f;
    build_docs();
    /* Top-3 nearest to [1,0,0] by cosine over full table.  Expect row 0
     * first (exact match, dist 0).  `select ... nearest` returns source
     * columns projected at the matching rowids; use the user's id column. */
    munit_assert_int(eval_i64(
        "(at (at (select {from: __docs nearest: (knn emb [1.0 0.0 0.0]) take: 3}) 'id) 0)"),
        ==, 0);
    /* `_dist` is only in the output when the user projects it explicitly. */
    munit_assert_double_equal(eval_f64(
        "(at (at (select {id: id d: _dist from: __docs "
        "                 nearest: (knn emb [1.0 0.0 0.0]) take: 3}) 'd) 0)"),
        0.0, 6);
    return MUNIT_OK;
}

static MunitResult test_select_nearest_ann(const void* p, void* f) {
    (void)p; (void)f;
    build_docs();
    ray_eval_str("(set __idx (hnsw-build (at __docs 'emb) 'cosine))");
    /* Top-1 must be exact match (row 0, id=0, dist=0). */
    munit_assert_int(eval_i64(
        "(at (at (select {from: __docs nearest: (ann __idx [1.0 0.0 0.0]) take: 1}) 'id) 0)"),
        ==, 0);
    munit_assert_double_equal(eval_f64(
        "(at (at (select {id: id d: _dist from: __docs "
        "                 nearest: (ann __idx [1.0 0.0 0.0]) take: 1}) 'd) 0)"),
        0.0, 6);
    ray_eval_str("(hnsw-free __idx)");
    return MUNIT_OK;
}

static MunitResult test_select_nearest_filter(const void* p, void* f) {
    (void)p; (void)f;
    build_docs();
    /* Filter keeps rows {0, 2, 4} (score > 0.5).  Top-2 nearest to
     * [1,0,0]: id=0 (exact), id=4 (cos=1/√2, dist≈0.293).  id=3 is
     * excluded because its score=0.1. */
    ray_t* r = ray_eval_str(
        "(select {from: __docs where: (> score 0.5) "
        "         nearest: (knn emb [1.0 0.0 0.0]) take: 2})");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_int(r->type, ==, RAY_TABLE);
    munit_assert_int(ray_table_nrows(r), ==, 2);
    ray_release(r);

    munit_assert_int(eval_i64(
        "(at (at (select {from: __docs where: (> score 0.5) "
        "                 nearest: (knn emb [1.0 0.0 0.0]) take: 2}) 'id) 0)"),
        ==, 0);
    munit_assert_int(eval_i64(
        "(at (at (select {from: __docs where: (> score 0.5) "
        "                 nearest: (knn emb [1.0 0.0 0.0]) take: 2}) 'id) 1)"),
        ==, 4);
    return MUNIT_OK;
}

static MunitResult test_select_nearest_empty_filter(const void* p, void* f) {
    (void)p; (void)f;
    build_docs();
    /* Filter rejects every row (score > 100) → empty result with schema. */
    ray_t* r = ray_eval_str(
        "(select {from: __docs where: (> score 100.0) "
        "         nearest: (knn emb [1.0 0.0 0.0]) take: 3})");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_int(r->type, ==, RAY_TABLE);
    munit_assert_int(ray_table_nrows(r), ==, 0);
    /* Schema preserved: id, score, emb (3 source columns).  `_dist` is
     * only present when the user projects it explicitly. */
    munit_assert_int(ray_table_ncols(r), ==, 3);
    ray_release(r);
    return MUNIT_OK;
}

static MunitResult test_select_nearest_take_default(const void* p, void* f) {
    (void)p; (void)f;
    build_docs();
    /* No take: default k=10, but table only has 5 rows → 5 results. */
    ray_t* r = ray_eval_str(
        "(select {from: __docs nearest: (knn emb [1.0 0.0 0.0])})");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_int(r->type, ==, RAY_TABLE);
    munit_assert_int(ray_table_nrows(r), ==, 5);
    ray_release(r);
    return MUNIT_OK;
}

static MunitResult test_select_nearest_asc_rejected(const void* p, void* f) {
    (void)p; (void)f;
    build_docs();
    /* nearest + asc on same select must error — the two orderings are
     * mutually exclusive. */
    ray_t* r = ray_eval_str(
        "(select {from: __docs "
        "         nearest: (knn emb [1.0 0.0 0.0]) "
        "         asc: score take: 2})");
    munit_assert_ptr_not_null(r);
    munit_assert_true(RAY_IS_ERR(r));
    return MUNIT_OK;
}

/* `_dist` must NOT leak into the output schema when the user didn't
 * project it.  (select {from: t nearest: ...}) must be shape-compatible
 * with (select {from: t}). */
static MunitResult test_select_nearest_implicit_no_dist(const void* p, void* f) {
    (void)p; (void)f;
    build_docs();
    ray_t* r = ray_eval_str(
        "(select {from: __docs nearest: (knn emb [1.0 0.0 0.0]) take: 3})");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_int(r->type, ==, RAY_TABLE);
    /* Must match source schema: id, score, emb — no _dist. */
    munit_assert_int(ray_table_ncols(r), ==, 3);
    /* Look up `_dist` explicitly: should fail (column absent). */
    int64_t dist_sym = ray_sym_intern("_dist", 5);
    munit_assert_ptr_equal(ray_table_get_col(r, dist_sym), NULL);
    ray_release(r);
    return MUNIT_OK;
}

/* STR columns in the source must project correctly through the rerank
 * gather path (previously errored with "nyi" and mis-copied DATE/TIME). */
static MunitResult test_select_nearest_str_col(const void* p, void* f) {
    (void)p; (void)f;
    ray_eval_str(
        "(set __tdocs (table [id title emb] "
        "  (list [0 1 2 3 4] "
        "        (list \"alpha\" \"beta\" \"gamma\" \"delta\" \"epsilon\") "
        "        (list [1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0] "
        "              [1.0 1.0 0.0] [1.0 0.0 1.0]))))");
    /* Top-1 by cosine to [1,0,0] is row 0 ("alpha"). */
    ray_t* r = ray_eval_str(
        "(at (at (select {id: id title: title from: __tdocs "
        "                 nearest: (knn emb [1.0 0.0 0.0]) take: 1}) 'title) 0)");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    /* STR atom. */
    munit_assert_int(r->type, ==, -RAY_STR);
    munit_assert_int((int)ray_str_len(r), ==, 5);
    munit_assert_memory_equal(5, ray_str_ptr(r), "alpha");
    ray_release(r);
    return MUNIT_OK;
}

/* SYM columns round-trip correctly through the rerank gather path.
 * (An earlier revision passed the ray_write_sym args in the wrong order,
 * producing garbage sym ids.) */
static MunitResult test_select_nearest_sym_col(const void* p, void* f) {
    (void)p; (void)f;
    /* A SYM column holding the 5 tickers; each test sym_id is global. */
    ray_eval_str(
        "(set __sdocs (table [id tag emb] "
        "  (list [0 1 2 3 4] "
        "        [alpha beta gamma delta epsilon] "
        "        (list [1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0] "
        "              [1.0 1.0 0.0] [1.0 0.0 1.0]))))");
    /* Top-1 cosine to [1,0,0] → row 0 → tag `alpha`. */
    ray_t* r = ray_eval_str(
        "(at (at (select {id: id tag: tag from: __sdocs "
        "                 nearest: (knn emb [1.0 0.0 0.0]) take: 1}) 'tag) 0)");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_int(r->type, ==, -RAY_SYM);
    int64_t expected = ray_sym_intern("alpha", 5);
    munit_assert_int(r->i64, ==, expected);
    ray_release(r);
    return MUNIT_OK;
}

/* Empty source table (0 rows) must not crash the oversample or heap
 * allocation — returns a well-shaped 0-row result. */
static MunitResult test_select_nearest_empty_source(const void* p, void* f) {
    (void)p; (void)f;
    /* Build a 3-column empty table with an empty list for `emb`. */
    ray_eval_str(
        "(set __empty (table [id emb] (list [] (list))))");
    ray_t* r = ray_eval_str(
        "(select {from: __empty nearest: (knn emb [1.0 0.0 0.0]) take: 5})");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_int(r->type, ==, RAY_TABLE);
    munit_assert_int(ray_table_nrows(r), ==, 0);
    munit_assert_int(ray_table_ncols(r), ==, 2);  /* id, emb — no _dist */
    ray_release(r);
    return MUNIT_OK;
}

/* Filter-aware iterative scan: a modestly selective filter (~50% pass rate).
 *
 * The prior oversample+refilter path pulled K × oversample_factor candidates
 * without filter awareness, then dropped non-matching ones — risking
 * undershoot for any filter with local rejections.  The iterative scan
 * pushes the predicate into the beam so rejected candidates don't consume
 * result slots.
 *
 * We verify:
 *  - full K rows returned,
 *  - every returned row passes the filter,
 *  - ordering is ascending by _dist.
 *
 * (Very-low-selectivity filters are a separate domain — HNSW beam is
 * locality-bounded by design; pgvector addresses this via
 * `iterative_scan_max_tuples` / multi-entry restart.) */
static MunitResult test_select_nearest_iterative_selective(const void* p, void* f) {
    (void)p; (void)f;
    srand(42);
    const int N = 300, D = 16, K = 10;

    ray_t* vlist = ray_list_new(N);
    ray_t* scores = ray_vec_new(RAY_F64, N);
    munit_assert_ptr_not_null(vlist);
    munit_assert_ptr_not_null(scores);
    scores->len = N;
    double* sdata = (double*)ray_data(scores);

    for (int i = 0; i < N; i++) {
        ray_t* v = ray_vec_new(RAY_F32, D);
        v->len = D;
        float* d = (float*)ray_data(v);
        for (int j = 0; j < D; j++) d[j] = (float)rand() / (float)RAND_MAX - 0.5f;
        vlist = ray_list_append(vlist, v);
        ray_release(v);
        /* ~50% pass the filter — enough density that iterative scan can
         * reliably fill K via graph traversal. */
        sdata[i] = (double)rand() / (double)RAND_MAX;
    }

    munit_assert_int(ray_env_set(ray_sym_intern("__it_list", 9), vlist), ==, RAY_OK);
    ray_release(vlist);
    munit_assert_int(ray_env_set(ray_sym_intern("__it_scores", 11), scores), ==, RAY_OK);
    ray_release(scores);

    ray_eval_str(
        "(set __it_tbl (table [id score emb] "
        "  (list (til 300) __it_scores __it_list)))");
    ray_eval_str("(set __it_idx (hnsw-build __it_list 'l2 16 100))");

    ray_t* r = ray_eval_str(
        "(select {id: id score: score d: _dist from: __it_tbl "
        "         where: (> score 0.5) "
        "         nearest: (ann __it_idx (at __it_list 0)) "
        "         take: 10})");
    munit_assert_ptr_not_null(r);
    munit_assert_false(RAY_IS_ERR(r));
    munit_assert_int(r->type, ==, RAY_TABLE);
    /* Must return K rows — iterative scan fills from graph neighborhood. */
    munit_assert_int(ray_table_nrows(r), ==, K);

    /* Every returned row passes the filter (score > 0.5). */
    ray_t* sc_col = ray_table_get_col(r, ray_sym_intern("score", 5));
    munit_assert_ptr_not_null(sc_col);
    munit_assert_int(sc_col->type, ==, RAY_F64);
    double* sc = (double*)ray_data(sc_col);
    for (int64_t i = 0; i < ray_table_nrows(r); i++)
        munit_assert_true(sc[i] > 0.5);

    /* Distances ascending. */
    ray_t* d_col = ray_table_get_col(r, ray_sym_intern("d", 1));
    munit_assert_ptr_not_null(d_col);
    double* dd = (double*)ray_data(d_col);
    for (int64_t i = 1; i < ray_table_nrows(r); i++)
        munit_assert_true(dd[i] >= dd[i - 1]);

    ray_release(r);
    ray_eval_str("(hnsw-free __it_idx)");
    return MUNIT_OK;
}

static MunitResult test_select_nearest_recall(const void* p, void* f) {
    (void)p; (void)f;
    /* 200×16 random vectors, self-query each one: ANN top-1 via select must
     * match brute-force top-1 via select for every query (recall@1 = 1.0). */
    srand(17);
    const int N = 200, D = 16;
    ray_t* list = ray_list_new(N);
    munit_assert_ptr_not_null(list);
    for (int i = 0; i < N; i++) {
        ray_t* v = ray_vec_new(RAY_F32, D);
        v->len = D;
        float* d = (float*)ray_data(v);
        for (int j = 0; j < D; j++) d[j] = (float)rand() / (float)RAND_MAX - 0.5f;
        list = ray_list_append(list, v);
        ray_release(v);
    }
    int64_t list_sym = ray_sym_intern("__rv_list", 9);
    munit_assert_int(ray_env_set(list_sym, list), ==, RAY_OK);
    ray_release(list);

    /* Build a table with row-id + emb. */
    ray_eval_str(
        "(set __rv (table [id emb] "
        "  (list (til 200) __rv_list)))");
    ray_eval_str("(set __rv_idx (hnsw-build __rv_list 'l2 16 100))");

    int hits = 0;
    for (int q = 0; q < 20; q++) {
        char expr[512];
        snprintf(expr, sizeof(expr),
            "(at (at (select {from: __rv "
            "                 nearest: (ann __rv_idx (at __rv_list %d)) "
            "                 take: 1}) 'id) 0)", q);
        int64_t ann_id = eval_i64(expr);

        snprintf(expr, sizeof(expr),
            "(at (at (select {from: __rv "
            "                 nearest: (knn emb (at __rv_list %d) 'l2) "
            "                 take: 1}) 'id) 0)", q);
        int64_t knn_id = eval_i64(expr);
        if (ann_id == knn_id && ann_id == q) hits++;
    }
    ray_eval_str("(hnsw-free __rv_idx)");
    munit_assert_int(hits, ==, 20);
    return MUNIT_OK;
}

/* ============ Suite table ============ */

static MunitTest embedding_tests[] = {
    { "/cos_dist_scalar",     test_cos_dist_scalar,     emb_setup, emb_teardown, 0, NULL },
    { "/l2_dist_scalar",      test_l2_dist_scalar,      emb_setup, emb_teardown, 0, NULL },
    { "/inner_prod_scalar",   test_inner_prod_scalar,   emb_setup, emb_teardown, 0, NULL },
    { "/norm_scalar",         test_norm_scalar,         emb_setup, emb_teardown, 0, NULL },
    { "/metric_length_err",   test_metric_length_mismatch, emb_setup, emb_teardown, 0, NULL },
    { "/metric_type_err",     test_metric_type_error,   emb_setup, emb_teardown, 0, NULL },
    { "/cos_dist_list",       test_cos_dist_list,       emb_setup, emb_teardown, 0, NULL },
    { "/l2_dist_list",        test_l2_dist_list,        emb_setup, emb_teardown, 0, NULL },
    { "/inner_prod_list",     test_inner_prod_list,     emb_setup, emb_teardown, 0, NULL },
    { "/norm_list",           test_norm_list,           emb_setup, emb_teardown, 0, NULL },
    { "/knn_cosine",          test_knn_cosine,          emb_setup, emb_teardown, 0, NULL },
    { "/knn_l2",              test_knn_l2,              emb_setup, emb_teardown, 0, NULL },
    { "/knn_ip",              test_knn_ip,              emb_setup, emb_teardown, 0, NULL },
    { "/hnsw_info_metric",    test_hnsw_info_and_metric, emb_setup, emb_teardown, 0, NULL },
    { "/ann_topk",            test_ann_topk,            emb_setup, emb_teardown, 0, NULL },
    { "/ann_each_metric",     test_ann_each_metric,     emb_setup, emb_teardown, 0, NULL },
    { "/ann_recall",          test_ann_recall,          emb_setup, emb_teardown, 0, NULL },
    { "/hnsw_save_load",      test_hnsw_save_load,      emb_setup, emb_teardown, 0, NULL },
    { "/hnsw_free_twice",     test_hnsw_free_twice,     emb_setup, emb_teardown, 0, NULL },
    { "/hnsw_rebind_no_leak", test_hnsw_rebind_no_leak, emb_setup, emb_teardown, 0, NULL },
    { "/hnsw_inline_no_leak", test_hnsw_inline_no_leak, emb_setup, emb_teardown, 0, NULL },
    { "/hnsw_teardown_no_leak", test_hnsw_teardown_no_leak, emb_setup, emb_teardown, 0, NULL },
    { "/hnsw_handle_cow",     test_hnsw_handle_cow,     emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_knn",  test_select_nearest_knn,  emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_ann",  test_select_nearest_ann,  emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_filter", test_select_nearest_filter, emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_empty_filter", test_select_nearest_empty_filter, emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_take_default", test_select_nearest_take_default, emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_asc_rejected", test_select_nearest_asc_rejected, emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_implicit_no_dist", test_select_nearest_implicit_no_dist, emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_empty_source", test_select_nearest_empty_source, emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_str_col", test_select_nearest_str_col, emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_sym_col", test_select_nearest_sym_col, emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_iterative_selective", test_select_nearest_iterative_selective, emb_setup, emb_teardown, 0, NULL },
    { "/select_nearest_recall", test_select_nearest_recall, emb_setup, emb_teardown, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL }
};

MunitSuite test_embedding_suite = {
    "/embedding", embedding_tests, NULL, 1, 0
};
