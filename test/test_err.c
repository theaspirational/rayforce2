/*
 *   Copyright (c) 2025-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 */

#include "test.h"
#include <rayforce.h>
#include "core/runtime.h"
#include "mem/heap.h"
#include <string.h>

static ray_vm_t test_vm;

static void err_setup(void) {
    ray_heap_init();
    memset(&test_vm, 0, sizeof(test_vm));
    __VM = &test_vm;
}

static void err_teardown(void) {
    __VM = NULL;
    ray_heap_destroy();
}

static test_result_t test_error_basic(void) {
    ray_t* err = ray_error("type", NULL);
    TEST_ASSERT_NOT_NULL(err);
    TEST_ASSERT_TRUE(RAY_IS_ERR(err));
    TEST_ASSERT_EQ_U(err->slen, 4);
    TEST_ASSERT_MEM_EQ(4, err->sdata, "type");
    PASS();
}

static test_result_t test_error_with_message(void) {
    ray_t* err = ray_error("arity", "expected %d args, got %d", 2, 1);
    TEST_ASSERT_NOT_NULL(err);
    TEST_ASSERT_TRUE(RAY_IS_ERR(err));
    TEST_ASSERT_EQ_U(err->slen, 5);
    TEST_ASSERT_MEM_EQ(5, err->sdata, "arity");
    const char* msg = ray_error_msg();
    TEST_ASSERT_NOT_NULL(msg);
    TEST_ASSERT_STR_EQ(msg, "expected 2 args, got 1");
    PASS();
}

static test_result_t test_error_code_max_length(void) {
    ray_t* err = ray_error("longcode", "detail");
    TEST_ASSERT_NOT_NULL(err);
    TEST_ASSERT_TRUE(RAY_IS_ERR(err));
    /* sdata is 7 bytes max, so "longcode" (8 chars) truncated to "longcod" */
    TEST_ASSERT_EQ_U(err->slen, 7);
    TEST_ASSERT_MEM_EQ(7, err->sdata, "longcod");
    PASS();
}

static test_result_t test_error_null_not_error(void) {
    TEST_ASSERT_FALSE(RAY_IS_ERR(NULL));
    PASS();
}

static test_result_t test_error_normal_obj_not_error(void) {
    ray_t* i = ray_i64(42);
    TEST_ASSERT_NOT_NULL(i);
    TEST_ASSERT_FALSE(RAY_IS_ERR(i));
    PASS();
}

static test_result_t test_error_clear(void) {
    ray_error("test", "detail message");
    TEST_ASSERT_NOT_NULL(ray_error_msg());
    ray_error_clear();
    TEST_ASSERT_NULL(ray_error_msg());
    PASS();
}

const test_entry_t err_entries[] = {
    { "err/basic",              test_error_basic,              err_setup, err_teardown },
    { "err/with_message",       test_error_with_message,       err_setup, err_teardown },
    { "err/code_max_length",    test_error_code_max_length,    err_setup, err_teardown },
    { "err/null_not_error",     test_error_null_not_error,     err_setup, err_teardown },
    { "err/normal_obj_not_err", test_error_normal_obj_not_error, err_setup, err_teardown },
    { "err/clear",              test_error_clear,              err_setup, err_teardown },
    { NULL, NULL, NULL, NULL },
};
