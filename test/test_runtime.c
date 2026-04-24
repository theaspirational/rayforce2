/*
 *   Copyright (c) 2025-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 */

#define _DEFAULT_SOURCE   /* mkdtemp, strdup */

#include "munit.h"
#include <rayforce.h>
#include "core/runtime.h"   /* ray_runtime_t, ray_runtime_create*, __RUNTIME */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

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
    /* Pin the exact error code — the contract maps every non-ENOENT
     * stat failure to RAY_ERR_IO, so drift in the mapping should fail
     * this test loudly. */
    munit_assert_int((int)err, ==, (int)RAY_ERR_IO);

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

/* Corrupt sym file must surface as RAY_ERR_CORRUPT via the _err variant
 * (not silently downgraded to RAY_OK).  We fake a corrupt file by
 * writing random bytes — ray_sym_load expects a serialized RAY_LIST of
 * -RAY_STR entries, so arbitrary bytes will fail its header validation. */
static MunitResult test_create_with_sym_corrupt_file(const void* params, void* fixture) {
    (void)params; (void)fixture;
    char* dir = make_tmpdir();
    munit_assert_ptr_not_null(dir);

    char path[256];
    snprintf(path, sizeof(path), "%s/corrupt.sym", dir);
    FILE* f = fopen(path, "wb");
    munit_assert_ptr_not_null(f);
    /* Write STR_LIST_MAGIC ("STRL" little-endian) followed by a truncated
     * payload — header-count byte count=999 but no body — ray_col_load
     * will hit col_load_str_list's "corrupt" path, which maps to
     * RAY_ERR_CORRUPT via ray_err_from_obj. */
    uint32_t magic = 0x4C525453U;  /* STR_LIST_MAGIC */
    int64_t count = 999;           /* claims 999 strings, none present */
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&count, sizeof(count), 1, f);
    fclose(f);

    ray_err_t err = RAY_OK;
    ray_runtime_t* rt = ray_runtime_create_with_sym_err(path, &err);
    munit_assert_ptr_not_null(rt);
    /* Pin the exact error code: the contract maps corrupted sym data
     * to RAY_ERR_CORRUPT, distinct from I/O or OOM, so callers can
     * decide recovery policy. */
    munit_assert_int((int)err, ==, (int)RAY_ERR_CORRUPT);

    ray_runtime_destroy(rt);
    unlink(path);
    rmdir(dir);
    free(dir);
    return MUNIT_OK;
}

/* Load-before-builtins ordering is the whole reason
 * ray_runtime_create_with_sym exists: after a save/destroy/load cycle,
 * user-interned sym IDs must occupy exactly the slots they had before,
 * while builtins append afterwards.  Intern a distinctive name, save,
 * tear down, reload via the persistent-consumer entrypoint, and verify
 * the same string interns to the same ID. */
static MunitResult test_create_with_sym_load_preserves_user_ids(const void* params, void* fixture) {
    (void)params; (void)fixture;
    char* dir = make_tmpdir();
    munit_assert_ptr_not_null(dir);

    char path[256];
    snprintf(path, sizeof(path), "%s/ids.sym", dir);

    /* Phase 1: intern a name then persist the sym table. */
    ray_runtime_t* rt1 = ray_runtime_create(0, NULL);
    munit_assert_ptr_not_null(rt1);
    int64_t id_before = ray_sym_intern("rayforce-user-marker", 20);
    munit_assert_int((int)ray_sym_save(path), ==, (int)RAY_OK);
    ray_runtime_destroy(rt1);

    /* Phase 2: bring up a fresh runtime via the _with_sym variant so the
     * persisted table is loaded before builtins register. */
    ray_err_t err = RAY_ERR_OOM;
    ray_runtime_t* rt2 = ray_runtime_create_with_sym_err(path, &err);
    munit_assert_ptr_not_null(rt2);
    munit_assert_int((int)err, ==, (int)RAY_OK);

    /* Same string must re-intern to the same ID (not shift because of
     * builtins claiming the low slots first). */
    int64_t id_after = ray_sym_intern("rayforce-user-marker", 20);
    munit_assert_int((int)id_after, ==, (int)id_before);

    ray_runtime_destroy(rt2);
    unlink(path);
    /* ray_sym_save may also create a lock file. */
    char lock_path[320];
    snprintf(lock_path, sizeof(lock_path), "%s.lk", path);
    unlink(lock_path);
    rmdir(dir);
    free(dir);
    return MUNIT_OK;
}

/* Sym file whose stat st_size exceeds mem_budget/2 must trigger the
 * pre-flight OOM guard and surface RAY_ERR_OOM through out_sym_err.
 * We use ftruncate to create a sparse file without actually allocating
 * the backing bytes.  Budget auto-detects ~80% of RAM, so a sparse
 * file ~10 EB guarantees tripping the half-budget ceiling on any
 * realistic dev/CI host. */
static MunitResult test_create_with_sym_oversized_file(const void* params, void* fixture) {
    (void)params; (void)fixture;
    char* dir = make_tmpdir();
    munit_assert_ptr_not_null(dir);

    /* Skip on platforms with 32-bit off_t — the sparse size we want
     * (>> 4 GB) isn't representable and the shift in that case would
     * be undefined. */
    if (sizeof(off_t) < 8) {
        free(dir);
        return MUNIT_SKIP;
    }

    char path[256];
    snprintf(path, sizeof(path), "%s/huge.sym", dir);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    munit_assert_int(fd, >=, 0);
    /* 4 EB sparse — bigger than any plausible mem_budget/2 (<1 ZB of
     * RAM).  Build via int64_t to keep the shift well-defined, then
     * cast to off_t after the width guard above has passed. */
    int64_t huge64 = (int64_t)1 << 62;
    off_t huge = (off_t)huge64;
    int rc = ftruncate(fd, huge);
    close(fd);
    if (rc != 0) {
        /* Some filesystems (tmpfs on limited hosts) reject the giant
         * ftruncate — skip rather than fail spuriously. */
        unlink(path);
        rmdir(dir);
        free(dir);
        return MUNIT_SKIP;
    }

    ray_err_t err = RAY_OK;
    ray_runtime_t* rt = ray_runtime_create_with_sym_err(path, &err);
    munit_assert_ptr_not_null(rt);
    munit_assert_int((int)err, ==, (int)RAY_ERR_OOM);

    ray_runtime_destroy(rt);
    unlink(path);
    rmdir(dir);
    free(dir);
    return MUNIT_OK;
}

static MunitTest runtime_tests[] = {
    { "/create_with_sym_absent_is_ok",     test_create_with_sym_absent_is_ok,     NULL, NULL, 0, NULL },
    { "/create_with_sym_io_error_surfaces", test_create_with_sym_io_error_surfaces, NULL, NULL, 0, NULL },
    { "/create_with_sym_plain_variant_absent", test_create_with_sym_plain_variant_absent, NULL, NULL, 0, NULL },
    { "/create_with_sym_corrupt_file",     test_create_with_sym_corrupt_file,     NULL, NULL, 0, NULL },
    { "/create_with_sym_load_preserves_user_ids", test_create_with_sym_load_preserves_user_ids, NULL, NULL, 0, NULL },
    { "/create_with_sym_oversized_file",   test_create_with_sym_oversized_file,   NULL, NULL, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL },
};

MunitSuite test_runtime_suite = { "/runtime", runtime_tests, NULL, 1, 0 };
