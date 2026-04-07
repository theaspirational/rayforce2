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

static MunitTest datalog_tests[] = {
    { "/source_provenance",         test_source_provenance,         datalog_setup, datalog_teardown, 0, NULL },
    { "/source_prov_requires_flag", test_source_prov_requires_flag, datalog_setup, datalog_teardown, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL },
};

MunitSuite test_datalog_suite = { "/datalog", datalog_tests, NULL, 1, 0 };
