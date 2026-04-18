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
/*
 * test_datalog.c — Tests for the Datalog engine (src/ops/datalog.h)
 *
 * Covers: deep source provenance (CSR offsets + packed source refs).
 */
#include "munit.h"
#include <rayforce.h>
#include "mem/heap.h"
#include "ops/datalog.h"
#include <string.h>

static void* datalog_setup(const void* params, void* user_data) {
    (void)params; (void)user_data;
    ray_heap_init();
    (void)ray_sym_init();
    return NULL;
}

static void datalog_teardown(void* fixture) {
    (void)fixture;
    ray_sym_destroy();
    ray_heap_destroy();
}

/* Verify that dl_get_provenance_src_offsets and dl_get_provenance_src_data
 * are populated correctly for a simple one-rule derivation.
 *
 * Program:
 *   EDB: edge(1,2), edge(2,3), edge(3,4)
 *   Rule: path(X,Y) :- edge(X,Y)
 *
 * Expected after eval with DL_FLAG_PROVENANCE:
 *   path has 3 rows (one per edge row).
 *   prov_col[i] = 0  (rule index 0 fired for all rows)
 *   prov_src_offsets = [0, 1, 2, 3]  (one source entry per derived row)
 *   prov_src_data[i] = (edge_rel_idx << 32) | i
 */
static MunitResult test_source_provenance(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t src_vals[] = {1, 2, 3};
    int64_t dst_vals[] = {2, 3, 4};
    ray_t* src = ray_vec_from_raw(RAY_I64, src_vals, 3);
    ray_t* dst = ray_vec_from_raw(RAY_I64, dst_vals, 3);
    munit_assert_ptr_not_null(src);
    munit_assert_ptr_not_null(dst);

    ray_t* edge = ray_table_new(2);
    munit_assert_ptr_not_null(edge);
    edge = ray_table_add_col(edge, ray_sym_intern("edge__c0", 8), src);
    munit_assert_false(RAY_IS_ERR(edge));
    edge = ray_table_add_col(edge, ray_sym_intern("edge__c1", 8), dst);
    munit_assert_false(RAY_IS_ERR(edge));

    dl_program_t* prog = dl_program_new();
    munit_assert_ptr_not_null(prog);
    prog->flags |= DL_FLAG_PROVENANCE;

    int edge_idx = dl_add_edb(prog, "edge", edge, 2);
    munit_assert_int(edge_idx, ==, 0);

    /* path(X,Y) :- edge(X,Y) */
    dl_rule_t rule;
    dl_rule_init(&rule, "path", 2);
    dl_rule_head_var(&rule, 0, 0);
    dl_rule_head_var(&rule, 1, 1);
    int body = dl_rule_add_atom(&rule, "edge", 2);
    munit_assert_int(body, ==, 0);
    dl_body_set_var(&rule, body, 0, 0);
    dl_body_set_var(&rule, body, 1, 1);
    munit_assert_int(dl_add_rule(prog, &rule), ==, 0);

    munit_assert_int(dl_eval(prog), ==, 0);

    ray_t* out = dl_query(prog, "path");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 3);

    /* Rule-level provenance: all rows attributed to rule 0 */
    ray_t* prov = dl_get_provenance(prog, "path");
    munit_assert_ptr_not_null(prov);
    munit_assert_int((int)ray_len(prov), ==, 3);
    int64_t* pv = (int64_t*)ray_data(prov);
    munit_assert_int((int)pv[0], ==, 0);
    munit_assert_int((int)pv[1], ==, 0);
    munit_assert_int((int)pv[2], ==, 0);

    /* Deep source provenance: CSR offsets and packed source refs */
    ray_t* offsets = dl_get_provenance_src_offsets(prog, "path");
    ray_t* data    = dl_get_provenance_src_data(prog, "path");
    munit_assert_ptr_not_null(offsets);
    munit_assert_ptr_not_null(data);

    /* offsets: length nrows+1 = 4; each derived row has exactly 1 source */
    munit_assert_int((int)ray_len(offsets), ==, 4);
    munit_assert_int((int)ray_len(data),    ==, 3);

    int64_t* off      = (int64_t*)ray_data(offsets);
    int64_t* src_data = (int64_t*)ray_data(data);

    munit_assert_int((int)off[0], ==, 0);
    munit_assert_int((int)off[1], ==, 1);
    munit_assert_int((int)off[2], ==, 2);
    munit_assert_int((int)off[3], ==, 3);

    /* Each entry encodes (rel_idx << 32) | row_idx */
    for (int i = 0; i < 3; i++) {
        int64_t expected = ((int64_t)edge_idx << 32) | (int64_t)i;
        munit_assert(src_data[i] == expected);
    }

    dl_program_free(prog);
    ray_release(edge);
    ray_release(src);
    ray_release(dst);
    return MUNIT_OK;
}

/* Deep provenance is only populated when DL_FLAG_PROVENANCE is set.
 * Without the flag both getters must return NULL. */
static MunitResult test_source_prov_requires_flag(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t vals[] = {1, 2};
    ray_t* v = ray_vec_from_raw(RAY_I64, vals, 2);
    munit_assert_ptr_not_null(v);

    ray_t* tbl = ray_table_new(1);
    tbl = ray_table_add_col(tbl, ray_sym_intern("p__c0", 5), v);
    munit_assert_false(RAY_IS_ERR(tbl));

    dl_program_t* prog = dl_program_new();
    munit_assert_ptr_not_null(prog);
    /* DL_FLAG_PROVENANCE intentionally NOT set */

    dl_add_edb(prog, "p", tbl, 1);

    dl_rule_t rule;
    dl_rule_init(&rule, "q", 1);
    dl_rule_head_var(&rule, 0, 0);
    int body = dl_rule_add_atom(&rule, "p", 1);
    dl_body_set_var(&rule, body, 0, 0);
    dl_add_rule(prog, &rule);

    munit_assert_int(dl_eval(prog), ==, 0);

    munit_assert_null(dl_get_provenance_src_offsets(prog, "q"));
    munit_assert_null(dl_get_provenance_src_data(prog, "q"));

    dl_program_free(prog);
    ray_release(tbl);
    ray_release(v);
    return MUNIT_OK;
}

/* Verify cmp body literal filters tuples: rule keeps only rows where col0 < 60.
 *
 * Program:
 *   EDB: weight(50), weight(60), weight(75), weight(85)
 *   Rule: small(W) :- weight(W), (< W 60)
 *
 * Expected: small has exactly 1 row = 50.
 */
static MunitResult test_cmp_const_filter(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t vals[] = {50, 60, 75, 85};
    ray_t* col = ray_vec_from_raw(RAY_I64, vals, 4);
    munit_assert_ptr_not_null(col);

    ray_t* weight = ray_table_new(1);
    weight = ray_table_add_col(weight, ray_sym_intern("weight__c0", 10), col);
    munit_assert_false(RAY_IS_ERR(weight));

    dl_program_t* prog = dl_program_new();
    munit_assert_ptr_not_null(prog);

    int weight_idx = dl_add_edb(prog, "weight", weight, 1);
    munit_assert_int(weight_idx, ==, 0);

    /* small(W) :- weight(W), (< W 60) */
    dl_rule_t rule;
    dl_rule_init(&rule, "small", 1);
    dl_rule_head_var(&rule, 0, 0);  /* head var idx 0 = W */

    int body = dl_rule_add_atom(&rule, "weight", 1);
    dl_body_set_var(&rule, body, 0, 0);  /* weight(W) */

    int cmp = dl_rule_add_cmp_const(&rule, DL_CMP_LT, 0, 60);  /* W < 60 */
    munit_assert_int(cmp, >=, 0);

    rule.n_vars = 1;
    munit_assert_int(dl_add_rule(prog, &rule), ==, 0);
    munit_assert_int(dl_eval(prog), ==, 0);

    ray_t* out = dl_query(prog, "small");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 1);

    ray_t* out_col = ray_table_get_col_idx(out, 0);
    munit_assert_ptr_not_null(out_col);
    int64_t* od = (int64_t*)ray_data(out_col);
    munit_assert_int((int)od[0], ==, 50);

    dl_program_free(prog);
    ray_release(weight);
    ray_release(col);
    return MUNIT_OK;
}

/* Verify arithmetic assignment derives a new variable from input columns.
 *
 * Program:
 *   EDB: pair(2, 3), pair(5, 7), pair(10, 1)
 *   Rule: sum_rel(A, B, S) :- pair(A, B), (= S (+ A B))
 *
 * Expected: sum_rel has 3 rows: (2,3,5), (5,7,12), (10,1,11).
 */
static MunitResult test_arith_assignment(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t a_vals[] = {2, 5, 10};
    int64_t b_vals[] = {3, 7, 1};
    ray_t* a_col = ray_vec_from_raw(RAY_I64, a_vals, 3);
    ray_t* b_col = ray_vec_from_raw(RAY_I64, b_vals, 3);
    munit_assert_ptr_not_null(a_col);
    munit_assert_ptr_not_null(b_col);

    ray_t* pair = ray_table_new(2);
    pair = ray_table_add_col(pair, ray_sym_intern("pair__c0", 8), a_col);
    munit_assert_false(RAY_IS_ERR(pair));
    pair = ray_table_add_col(pair, ray_sym_intern("pair__c1", 8), b_col);
    munit_assert_false(RAY_IS_ERR(pair));

    dl_program_t* prog = dl_program_new();
    munit_assert_ptr_not_null(prog);
    munit_assert_int(dl_add_edb(prog, "pair", pair, 2), ==, 0);

    /* sum_rel(A, B, S) :- pair(A, B), (= S (+ A B)) */
    dl_rule_t rule;
    dl_rule_init(&rule, "sum_rel", 3);
    dl_rule_head_var(&rule, 0, 0);  /* A */
    dl_rule_head_var(&rule, 1, 1);  /* B */
    dl_rule_head_var(&rule, 2, 2);  /* S */

    int body = dl_rule_add_atom(&rule, "pair", 2);
    dl_body_set_var(&rule, body, 0, 0);  /* A */
    dl_body_set_var(&rule, body, 1, 1);  /* B */

    /* expr = (+ A B) */
    dl_expr_t* expr = dl_expr_binop(OP_ADD, dl_expr_var(0), dl_expr_var(1));
    munit_assert_ptr_not_null(expr);
    int as = dl_rule_add_assign(&rule, 2, DL_OP_EQ, expr);  /* S = A + B */
    munit_assert_int(as, >=, 0);

    rule.n_vars = 3;
    munit_assert_int(dl_add_rule(prog, &rule), ==, 0);
    munit_assert_int(dl_eval(prog), ==, 0);

    ray_t* out = dl_query(prog, "sum_rel");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 3);

    ray_t* s_col = ray_table_get_col_idx(out, 2);
    munit_assert_ptr_not_null(s_col);
    int64_t* sd = (int64_t*)ray_data(s_col);
    /* Sums must include 5, 12, 11 (order may differ). */
    int saw5 = 0, saw12 = 0, saw11 = 0;
    for (int i = 0; i < 3; i++) {
        if (sd[i] == 5)  saw5  = 1;
        if (sd[i] == 12) saw12 = 1;
        if (sd[i] == 11) saw11 = 1;
    }
    munit_assert_int(saw5,  ==, 1);
    munit_assert_int(saw12, ==, 1);
    munit_assert_int(saw11, ==, 1);

    dl_program_free(prog);
    ray_release(pair);
    ray_release(a_col);
    ray_release(b_col);
    return MUNIT_OK;
}

/* Verify dl_rule_add_agg populates body fields correctly. */
static MunitResult test_agg_builder(const void* params, void* fixture) {
    (void)params; (void)fixture;
    dl_rule_t rule;
    dl_rule_init(&rule, "stats", 1);
    dl_rule_head_var(&rule, 0, 0);

    int idx = dl_rule_add_agg(&rule, DL_AGG_COUNT, 0, "weight", 1, 0);
    munit_assert_int(idx, ==, 0);
    munit_assert_int(rule.body[0].type,           ==, DL_AGG);
    munit_assert_int(rule.body[0].agg_op,         ==, DL_AGG_COUNT);
    munit_assert_int(rule.body[0].agg_target_var, ==, 0);
    munit_assert_string_equal(rule.body[0].agg_pred, "weight");
    munit_assert_int(rule.body[0].agg_arity,      ==, 1);
    munit_assert_int(rule.body[0].agg_value_col,  ==, 0);
    munit_assert_int(rule.n_vars, ==, 1);

    dl_rule_t rule2;
    dl_rule_init(&rule2, "sum_stats", 1);
    dl_rule_head_var(&rule2, 0, 3);
    int idx2 = dl_rule_add_agg(&rule2, DL_AGG_SUM, 3, "readings", 4, 2);
    munit_assert_int(idx2, ==, 0);
    munit_assert_int(rule2.body[0].agg_op,        ==, DL_AGG_SUM);
    munit_assert_int(rule2.body[0].agg_target_var, ==, 3);
    munit_assert_int(rule2.body[0].agg_arity,     ==, 4);
    munit_assert_int(rule2.body[0].agg_value_col, ==, 2);
    munit_assert_int(rule2.n_vars,                ==, 4);
    return MUNIT_OK;
}

/* Aggregates over an IDB must be evaluated in a strictly higher stratum
 * than the IDB itself. Program:
 *   EDB: edge(1,2), edge(2,3)
 *   Rule R0: path(X,Y) :- edge(X,Y)
 *   Rule R1: path_count(N) :- (count ?N path)
 * After stratification: R1.stratum > R0.stratum. */
static MunitResult test_agg_stratifies_above_source(const void* params, void* fixture) {
    (void)params; (void)fixture;
    int64_t s_vals[] = {1, 2};
    int64_t d_vals[] = {2, 3};
    ray_t* sc = ray_vec_from_raw(RAY_I64, s_vals, 2);
    ray_t* dc = ray_vec_from_raw(RAY_I64, d_vals, 2);
    ray_t* edge = ray_table_new(2);
    edge = ray_table_add_col(edge, ray_sym_intern("edge__c0", 8), sc);
    edge = ray_table_add_col(edge, ray_sym_intern("edge__c1", 8), dc);

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "edge", edge, 2);

    dl_rule_t r0; dl_rule_init(&r0, "path", 2);
    dl_rule_head_var(&r0, 0, 0); dl_rule_head_var(&r0, 1, 1);
    int b = dl_rule_add_atom(&r0, "edge", 2);
    dl_body_set_var(&r0, b, 0, 0); dl_body_set_var(&r0, b, 1, 1);
    r0.n_vars = 2;
    dl_add_rule(prog, &r0);

    dl_rule_t r1; dl_rule_init(&r1, "path_count", 1);
    dl_rule_head_var(&r1, 0, 0);
    dl_rule_add_agg(&r1, DL_AGG_COUNT, 0, "path", 2, 0);
    r1.n_vars = 1;
    dl_add_rule(prog, &r1);

    munit_assert_int(dl_stratify(prog), ==, 0);
    munit_assert_int(prog->rules[1].stratum, >, prog->rules[0].stratum);

    dl_program_free(prog);
    ray_release(edge); ray_release(sc); ray_release(dc);
    return MUNIT_OK;
}

/* (count ?N weight) where weight has 4 rows -> N = 4. */
static MunitResult test_agg_count_edb(const void* params, void* fixture) {
    (void)params; (void)fixture;
    int64_t vals[] = {50, 60, 75, 85};
    ray_t* col = ray_vec_from_raw(RAY_I64, vals, 4);
    ray_t* weight = ray_table_new(1);
    weight = ray_table_add_col(weight, ray_sym_intern("weight__c0", 10), col);

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "weight", weight, 1);

    dl_rule_t r; dl_rule_init(&r, "wcount", 1);
    dl_rule_head_var(&r, 0, 0);
    dl_rule_add_agg(&r, DL_AGG_COUNT, 0, "weight", 1, 0);
    r.n_vars = 1;
    dl_add_rule(prog, &r);

    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "wcount");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 1);
    int64_t* od = (int64_t*)ray_data(ray_table_get_col_idx(out, 0));
    munit_assert_int((int)od[0], ==, 4);

    dl_program_free(prog);
    ray_release(weight); ray_release(col);
    return MUNIT_OK;
}

static ray_t* make_weight_edb(void) {
    int64_t vals[] = {50, 60, 75, 85};
    ray_t* col = ray_vec_from_raw(RAY_I64, vals, 4);
    ray_t* weight = ray_table_new(1);
    weight = ray_table_add_col(weight, ray_sym_intern("weight__c0", 10), col);
    ray_release(col);
    return weight;
}

static MunitResult test_agg_sum(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_t* weight = make_weight_edb();

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "weight", weight, 1);

    dl_rule_t r; dl_rule_init(&r, "wsum", 1);
    dl_rule_head_var(&r, 0, 0);
    dl_rule_add_agg(&r, DL_AGG_SUM, 0, "weight", 1, 0);
    r.n_vars = 1;
    dl_add_rule(prog, &r);

    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "wsum");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 1);
    int64_t* od = (int64_t*)ray_data(ray_table_get_col_idx(out, 0));
    munit_assert_int((int)od[0], ==, 270);

    dl_program_free(prog);
    ray_release(weight);
    return MUNIT_OK;
}

static MunitResult test_agg_min(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_t* weight = make_weight_edb();

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "weight", weight, 1);

    dl_rule_t r; dl_rule_init(&r, "wmin", 1);
    dl_rule_head_var(&r, 0, 0);
    dl_rule_add_agg(&r, DL_AGG_MIN, 0, "weight", 1, 0);
    r.n_vars = 1;
    dl_add_rule(prog, &r);

    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "wmin");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 1);
    int64_t* od = (int64_t*)ray_data(ray_table_get_col_idx(out, 0));
    munit_assert_int((int)od[0], ==, 50);

    dl_program_free(prog);
    ray_release(weight);
    return MUNIT_OK;
}

static MunitResult test_agg_max(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_t* weight = make_weight_edb();

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "weight", weight, 1);

    dl_rule_t r; dl_rule_init(&r, "wmax", 1);
    dl_rule_head_var(&r, 0, 0);
    dl_rule_add_agg(&r, DL_AGG_MAX, 0, "weight", 1, 0);
    r.n_vars = 1;
    dl_add_rule(prog, &r);

    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "wmax");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 1);
    int64_t* od = (int64_t*)ray_data(ray_table_get_col_idx(out, 0));
    munit_assert_int((int)od[0], ==, 85);

    dl_program_free(prog);
    ray_release(weight);
    return MUNIT_OK;
}

static MunitResult test_agg_avg(const void* params, void* fixture) {
    (void)params; (void)fixture;
    ray_t* weight = make_weight_edb();

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "weight", weight, 1);

    dl_rule_t r; dl_rule_init(&r, "wavg", 1);
    dl_rule_head_var(&r, 0, 0);
    dl_rule_add_agg(&r, DL_AGG_AVG, 0, "weight", 1, 0);
    r.n_vars = 1;
    dl_add_rule(prog, &r);

    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "wavg");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 1);
    int64_t* od = (int64_t*)ray_data(ray_table_get_col_idx(out, 0));
    munit_assert_int((int)od[0], ==, 67);

    dl_program_free(prog);
    ray_release(weight);
    return MUNIT_OK;
}

/* MIN over empty source -> rule produces no row. */
static MunitResult test_agg_min_empty(const void* params, void* fixture) {
    (void)params; (void)fixture;
    int64_t dummy = 0;
    ray_t* empty_vec = ray_vec_from_raw(RAY_I64, &dummy, 0);
    ray_t* weight = ray_table_new(1);
    weight = ray_table_add_col(weight, ray_sym_intern("weight__c0", 10), empty_vec);

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "weight", weight, 1);

    dl_rule_t r; dl_rule_init(&r, "wmin", 1);
    dl_rule_head_var(&r, 0, 0);
    dl_rule_add_agg(&r, DL_AGG_MIN, 0, "weight", 1, 0);
    r.n_vars = 1;
    dl_add_rule(prog, &r);

    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "wmin");
    /* Either NULL (no rel) or a 0-row table is acceptable "no row" semantics. */
    if (out) munit_assert_int((int)ray_table_nrows(out), ==, 0);

    dl_program_free(prog);
    ray_release(weight); ray_release(empty_vec);
    return MUNIT_OK;
}

/* COUNT over empty source -> 1 row with value 0 (well-defined). */
static MunitResult test_agg_count_empty(const void* params, void* fixture) {
    (void)params; (void)fixture;
    int64_t dummy = 0;
    ray_t* empty_vec = ray_vec_from_raw(RAY_I64, &dummy, 0);
    ray_t* weight = ray_table_new(1);
    weight = ray_table_add_col(weight, ray_sym_intern("weight__c0", 10), empty_vec);

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "weight", weight, 1);

    dl_rule_t r; dl_rule_init(&r, "wcnt", 1);
    dl_rule_head_var(&r, 0, 0);
    dl_rule_add_agg(&r, DL_AGG_COUNT, 0, "weight", 1, 0);
    r.n_vars = 1;
    dl_add_rule(prog, &r);

    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "wcnt");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 1);
    int64_t* od = (int64_t*)ray_data(ray_table_get_col_idx(out, 0));
    munit_assert_int((int)od[0], ==, 0);

    dl_program_free(prog);
    ray_release(weight); ray_release(empty_vec);
    return MUNIT_OK;
}

static MunitTest datalog_tests[] = {
    { "/source_provenance",         test_source_provenance,         datalog_setup, datalog_teardown, 0, NULL },
    { "/source_prov_requires_flag", test_source_prov_requires_flag, datalog_setup, datalog_teardown, 0, NULL },
    { "/cmp_const_filter",          test_cmp_const_filter,          datalog_setup, datalog_teardown, 0, NULL },
    { "/arith_assignment",          test_arith_assignment,          datalog_setup, datalog_teardown, 0, NULL },
    { "/agg_builder",                test_agg_builder,                datalog_setup, datalog_teardown, 0, NULL },
    { "/agg_stratifies_above_source", test_agg_stratifies_above_source, datalog_setup, datalog_teardown, 0, NULL },
    { "/agg_count_edb",              test_agg_count_edb,              datalog_setup, datalog_teardown, 0, NULL },
    { "/agg_sum",                    test_agg_sum,                    datalog_setup, datalog_teardown, 0, NULL },
    { "/agg_min",                    test_agg_min,                    datalog_setup, datalog_teardown, 0, NULL },
    { "/agg_max",                    test_agg_max,                    datalog_setup, datalog_teardown, 0, NULL },
    { "/agg_avg",                    test_agg_avg,                    datalog_setup, datalog_teardown, 0, NULL },
    { "/agg_min_empty",              test_agg_min_empty,              datalog_setup, datalog_teardown, 0, NULL },
    { "/agg_count_empty",            test_agg_count_empty,            datalog_setup, datalog_teardown, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL },
};

MunitSuite test_datalog_suite = { "/datalog", datalog_tests, NULL, 1, 0 };
