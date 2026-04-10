# Builtin Vector Dispatch — Design Document

## Problem

Rayforce2's element-wise builtins (+, -, *, /, %, ==, <, >, neg, abs, floor, ceil, sum, min, max, ...) have three overlapping execution paths in `atomic_map_binary_op` (eval.c):

1. **binop_vec** — typed pointer loops, rc==1 reuse, parallel. Only handles same-type I64/I32/F64, no nulls, no slices.
2. **Generic integer fast path** — per-element `READ_INT` macro with runtime `switch(dag_opcode)`. Handles mixed widths but prevents SIMD vectorization.
3. **DAG executor path** — builds a graph, runs optimizer + morsel executor. Correct but heavy overhead for simple vector ops.
4. **Element-by-element fallback** — `collection_elem` + `fn()` per element. Safe for slices and lists but ~10x slower.

This produces 400+ lines of tangled code, inconsistent slice handling, and the compiler cannot vectorize any path except binop_vec.

Rayforce1 has ONE path per builtin category:
- `binop_map(partial_fn, x, y)` — handles atoms, vectors, parallel dispatch, rc==1 reuse
- `unop_map(partial_fn, x)` — same for unary
- `unop_fold(partial_fn, x)` — same for reductions (sum, min, max)
- Each `partial_fn` has a `MTYPE2(x->type, y->type)` switch with typed pointer loops

## Target Architecture

### Three dispatch helpers (in `src/ops/dispatch.c`, new file)

```
ray_binop_map(partial_fn, x, y)    — binary element-wise (atom×atom, atom×vec, vec×vec)
ray_unop_map(partial_fn, x)        — unary element-wise
ray_unop_fold(partial_fn, x)       — unary reduction (sum, min, max, avg)
```

Each helper does:
1. **Atom fast path** — both scalars → call `partial_fn` directly
2. **Length check** — vec×vec must match
3. **Output type inference** — from `partial_fn` probe on first element
4. **Output allocation** — rc==1 reuse when type matches, no slices
5. **Parallel dispatch** — `ray_pool_dispatch` when `len >= RAY_PARALLEL_THRESHOLD`
6. **Single-threaded fallback** — direct call for small vectors

### Partial functions (in existing builtin files)

Each operation gets a `_partial` function with signature:

```c
// Binary partial: operates on a chunk [offset, offset+len) writing into out
ray_t* ray_add_partial(ray_t* x, ray_t* y, int64_t len, int64_t offset, ray_t* out);

// Unary partial: same pattern
ray_t* ray_neg_partial(ray_t* x, int64_t len, int64_t offset, ray_t* out);

// Fold partial: reduces a chunk and returns a scalar
ray_t* ray_sum_partial(ray_t* x, int64_t len, int64_t offset);
```

Each partial function contains a type switch:
```c
ray_t* ray_add_partial(ray_t* x, ray_t* y, int64_t len, int64_t offset, ray_t* out) {
    switch (MTYPE2(x->type, y->type)) {
        // atom × atom
        case MTYPE2(-RAY_I64, -RAY_I64): return ray_i64(x->i64 + y->i64);
        // atom × vec
        case MTYPE2(-RAY_I64, RAY_I64): {
            int64_t sv = x->i64;
            int64_t* r = (int64_t*)ray_data(y) + offset;
            int64_t* o = (int64_t*)ray_data(out) + offset;
            for (int64_t i = 0; i < len; i++) o[i] = sv + r[i];
            return NULL;  // success, result in out
        }
        // vec × atom
        case MTYPE2(RAY_I64, -RAY_I64): { ... }
        // vec × vec
        case MTYPE2(RAY_I64, RAY_I64): { ... }
        // cross-type: I32+I64 → I64
        case MTYPE2(RAY_I32, -RAY_I64): { ... widen I32 to I64 in loop ... }
        // float promotion
        case MTYPE2(RAY_I64, -RAY_F64): { ... }
        case MTYPE2(RAY_F64, RAY_F64): { ... }
        // temporal
        case MTYPE2(-RAY_DATE, -RAY_I64): return ray_date(x->i64 + y->i64);
        case MTYPE2(RAY_DATE, -RAY_I64): { ... }
        ...
    }
}
```

**Key property**: the inner loop has NO runtime dispatch. Types are resolved once per chunk in the switch, then the loop body is a single typed expression. The compiler auto-vectorizes to SIMD.

### Evaluator changes (eval.c)

`atomic_map_binary_op` shrinks to:

```c
ray_t* atomic_map_binary_op(ray_binary_fn fn, uint16_t opcode, ray_t* left, ray_t* right) {
    if (!is_collection(left) && !is_collection(right))
        return fn(left, right);
    // fn itself handles vectors via ray_binop_map — just call it
    return fn(left, right);
}
```

Wait — that collapses to just `fn(left, right)`. The `FN_ATOMIC` flag becomes unnecessary for builtins that handle vectors internally. It's only needed for user-defined lambdas that need auto-mapping.

So the evaluator dispatch becomes:
- `op_call2`: if `FN_ATOMIC` and user lambda → auto-map via `atomic_map_binary` (existing boxed path)
- `op_call2`: if builtin → always call `fn(left, right)` directly. The builtin handles vectors itself.

### Slice handling

Slices are handled at the `collection_elem` level (already fixed) and in the `ray_binop_map` helper which rejects slices from rc==1 reuse. The partial functions use `ray_data()` which requires non-slice vectors — `ray_binop_map` ensures this by falling back to per-element dispatch for slices.

### Null handling

Two tiers in the partial function:
1. **No nulls (fast)**: tight typed loop, compiler vectorizes
2. **Has nulls (bitmap)**: same typed loop but with `null_bm[i/8] & (1 << i%8)` check per 8-element batch

The null bitmap check is a bitwise AND — much faster than `ray_vec_is_null()` function call.

### What stays in the DAG executor

The DAG executor handles **query pipelines**: `select`, `update`, `filter`, `group`, `join`, `sort`. These chain multiple operations and benefit from morsel-driven fusion. Individual builtin calls (`(+ 1 vec)`) do NOT go through the DAG — they use `ray_binop_map` directly.

## Files Changed

| File | Change |
|------|--------|
| `src/ops/dispatch.c` (NEW) | `ray_binop_map`, `ray_unop_map`, `ray_unop_fold` |
| `src/ops/dispatch.h` (NEW) | Public API for dispatch helpers |
| `src/ops/arith.c` | `ray_add_partial`, `ray_sub_partial`, `ray_mul_partial`, `ray_div_partial`, `ray_mod_partial` + rewire `ray_add_fn` etc. |
| `src/ops/cmp.c` | `ray_eq_partial`, `ray_lt_partial`, etc. + rewire comparison builtins |
| `src/ops/agg.c` | `ray_sum_partial`, `ray_min_partial`, etc. + rewire aggregation builtins |
| `src/lang/eval.c` | Delete `atomic_map_binary_op`'s fast paths (integer loop, DAG path). Keep only: binop_vec call → per-element fallback for lists. Simplify `op_call2` dispatch. |
| `src/lang/eval_internal.h` | `collection_elem` slice fix (already done) |

## Migration Strategy

Phase 1: `dispatch.c` + arith partial functions (+, -, *, /, %)
Phase 2: comparison partial functions (==, !=, <, <=, >, >=)
Phase 3: unary partial functions (neg, abs, not, floor, ceil, round)
Phase 4: aggregation partial functions (sum, avg, min, max, count)
Phase 5: delete dead code from eval.c

Each phase: implement → test (597/597) → benchmark → commit.

## Performance Target

| Operation | Current | Target | Rayforce1 |
|-----------|---------|--------|-----------|
| `(+ 1 (til 1e8))` | 50ms | 35ms | ~32ms |
| `(sum (til 1e8))` | 45ms | 33ms | 31ms |
| `(> 50000000 (til 1e8))` | 44ms | 35ms | ~32ms |

The remaining gap vs rayforce1 after typed dispatch is `til` materialization (800MB write). Closing that requires lazy `til` — a separate design.
