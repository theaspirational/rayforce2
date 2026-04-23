/*
 *   Copyright (c) 2025-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 */

#define _DEFAULT_SOURCE   /* mkdtemp, strdup */

#include "munit.h"
#include <rayforce.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

/* Runtime API forward-declared here because core/runtime.h's `ray_vm_t`
 * definition collides with lang/eval.h's `ray_vm_t` when both are pulled
 * into the same TU (pre-existing duplication).  ray_err_t already comes
 * from <rayforce.h> above. */
typedef struct ray_runtime_s ray_runtime_t;
ray_runtime_t* ray_runtime_create(int argc, char** argv);
ray_runtime_t* ray_runtime_create_with_sym(const char* sym_path);
ray_runtime_t* ray_runtime_create_with_sym_err(const char* sym_path,
                                               ray_err_t* out_sym_err);
void           ray_runtime_destroy(ray_runtime_t* rt);
extern ray_runtime_t* __RUNTIME;

/* Import RAY_OK / RAY_ERR_IO enum values from rayforce.h -- they live in
 * the existing ray_err_t enum and are exposed via ray_err_from_obj /
 * ray_err_code_str; numeric values are part of the public surface. */

static char* make_tmpdir(void) {
    char tmpl[] = "/tmp/rayforce-rt-test-XXXXXX";
    char* dir = mkdtemp(tmpl);
    if (!dir) return NULL;
    return strdup(tmpl);
}

/* Absent sym file: stat fails with ENOENT, which is the "first run"
 * normal case.  out_sym_err must stay RAY_OK and runtime must come up. */
static MunitResult test_create_with_sym_absent_is_ok(const void* params, void* fixture) {
    (void)params; (void)fixture;
    char* dir = make_tmpdir();
    munit_assert_ptr_not_null(dir);
    char path[256];
    snprintf(path, sizeof(path), "%s/missing.sym", dir);

    ray_err_t err = RAY_ERR_OOM;  /* poison — should be overwritten */
    ray_runtime_t* rt = ray_runtime_create_with_sym_err(path, &err);
    munit_assert_ptr_not_null(rt);
    munit_assert_int((int)err, ==, (int)RAY_OK);

    ray_runtime_destroy(rt);
    rmdir(dir);
    free(dir);
    return MUNIT_OK;
}

/* Non-ENOENT stat failure must surface as RAY_ERR_IO.  We hit this by
 * passing a path whose parent exists but isn't a directory (ENOTDIR) —
 * portable across Linux/macOS without needing root or chmod games. */
static MunitResult test_create_with_sym_io_error_surfaces(const void* params, void* fixture) {
    (void)params; (void)fixture;
    char* dir = make_tmpdir();
    munit_assert_ptr_not_null(dir);

    /* Create a regular file, then ask to stat a path that treats it as a
     * directory prefix — POSIX returns ENOTDIR. */
    char blocker[256], path[256];
    snprintf(blocker, sizeof(blocker), "%s/not-a-dir", dir);
    snprintf(path, sizeof(path), "%s/not-a-dir/sym", dir);
    FILE* f = fopen(blocker, "w");
    munit_assert_ptr_not_null(f);
    fclose(f);

    ray_err_t err = RAY_OK;
    ray_runtime_t* rt = ray_runtime_create_with_sym_err(path, &err);
    munit_assert_ptr_not_null(rt);
    munit_assert_int((int)err, !=, (int)RAY_OK);

    ray_runtime_destroy(rt);
    unlink(blocker);
    rmdir(dir);
    free(dir);
    return MUNIT_OK;
}

/* The plain (non-_err) variant discards load result; runtime still comes
 * up cleanly regardless of sym-file state. */
static MunitResult test_create_with_sym_plain_variant_absent(const void* params, void* fixture) {
    (void)params; (void)fixture;
    char* dir = make_tmpdir();
    munit_assert_ptr_not_null(dir);
    char path[256];
    snprintf(path, sizeof(path), "%s/also-missing.sym", dir);

    ray_runtime_t* rt = ray_runtime_create_with_sym(path);
    munit_assert_ptr_not_null(rt);

    ray_runtime_destroy(rt);
    rmdir(dir);
    free(dir);
    return MUNIT_OK;
}

static MunitTest runtime_tests[] = {
    { "/create_with_sym_absent_is_ok",     test_create_with_sym_absent_is_ok,     NULL, NULL, 0, NULL },
    { "/create_with_sym_io_error_surfaces", test_create_with_sym_io_error_surfaces, NULL, NULL, 0, NULL },
    { "/create_with_sym_plain_variant_absent", test_create_with_sym_plain_variant_absent, NULL, NULL, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL },
};

MunitSuite test_runtime_suite = { "/runtime", runtime_tests, NULL, 1, 0 };
