# Remove Sentinel Nulls from Executor — Design

## Problem

The morsel executor (`expr.c`) uses `INT64_MIN` as a null sentinel in integer loops and `NaN` in float loops. This:
- Steals valid values from the type range (`INT64_MIN` is a real number)
- Duplicates null propagation already done by `propagate_nulls_binary`
- Adds per-element branches that prevent SIMD auto-vectorization
- Blocks routing ALL operations through the DAG (temporal, narrow types, div)

## Where Sentinels Live

Three code locations in `src/ops/expr.c`:

### 1. `fold_binary_const` (lines 131-142) — constant folding
Used during DAG optimization to fold `CONST op CONST`. Not a hot path.
Sentinel checks: `li==INT64_MIN||ri==INT64_MIN` before arithmetic.

### 2. `expr_exec_binary` (lines 664-713) — morsel hot loop
The core vectorized execution path. Switch-per-opcode with typed pointer loops.
- **I64 arithmetic** (664-677): `a[j]==N||b[j]==N ? N : compute`
- **I64 comparison** (704-713): `a[j]==N ? ... : compare`
- **F64 comparison** (691-697): `NaN` checks (F64_ISNAN)

### 3. `binary_range` (lines 1344-1400) — per-element fallback
Sequential loop used when the fused morsel path doesn't apply.
- **I64 arithmetic** (1348-1358): same sentinel pattern
- **I64 comparison** (1371-1382): same sentinel pattern
- **F64 comparison** (1386-1397): NaN checks

## How Nulls Already Work

1. Input vectors may have `RAY_ATTR_HAS_NULLS` with a bitmap
2. After morsel execution, `propagate_nulls_binary(lhs, rhs, result, ...)` copies input null bits to output bitmap
3. For comparisons, `clear_null_comparisons` zeroes output at null positions

The sentinel checks are REDUNDANT with steps 2-3. They only prevent computing on garbage values at null positions — but the null bit masks the result anyway.

## The Fix

### For arithmetic (add, sub, mul, min2, max2):
Delete sentinel checks. Compute unconditionally. Let `propagate_nulls_binary` handle nulls.

Before: `d[j] = (a[j]==N||b[j]==N) ? N : (uint64_t)a[j] + (uint64_t)b[j]`
After:  `d[j] = (int64_t)((uint64_t)a[j] + (uint64_t)b[j])`

### For div/mod:
Delete sentinel null checks. Keep only `b==0` check (actual div-by-zero).
Set bitmap null via post-pass scanning the divisor.

Before: `if (b[j]==0||a[j]==N||b[j]==N) { d[j]=N; continue; }`
After:  `if (b[j]==0) { d[j]=0; continue; }` + post-pass sets null bits

### For I64 comparisons:
Delete sentinel checks. Compare unconditionally. `clear_null_comparisons` zeros null positions.

Before: `d[j] = (a[j]==N&&b[j]==N) ? 1 : (a[j]==N||b[j]==N) ? 0 : a[j]==b[j]`
After:  `d[j] = a[j] == b[j]`

### For F64 comparisons:
Keep NaN checks — NaN IS the IEEE 754 null for floats, and `NaN != NaN` is correct behavior. `NaN < x` is false (correct for null < value). No change needed.

### For constant folding:
Delete sentinel checks — constant atoms use typed nulls (`RAY_ATOM_IS_NULL`), not sentinels.

### Div-by-zero null post-pass:
After `exec_elementwise_binary` returns, scan the divisor for zeros and set bitmap nulls on the result. Only for `OP_DIV` and `OP_MOD`. Both the parallel and sequential paths need this.

### Missing I32 output path in `binary_range`:
Add `else if (out_type == RAY_I32 || out_type == RAY_DATE || out_type == RAY_TIME)` block with typed I32 arithmetic. Currently absent — I32 output falls through without writing.

## Impact

- `INT64_MIN` becomes a valid value throughout the system
- Morsel loops have fewer branches → better auto-vectorization
- All ops can go through the DAG (remove integer fast path from eval.c)
- Performance: fewer branches in hot loops = faster

## Files Changed

| File | Change |
|------|--------|
| `src/ops/expr.c` | Remove sentinels from fold_binary_const, expr_exec_binary, binary_range. Add I32 output to binary_range. Add div-by-zero post-pass. |
| `src/lang/eval.c` | Remove integer fast path (lines 283-418). Remove DAG restrictions for temporal/narrow/div. |

## Risks

- F64 NaN checks are NOT removed — they implement correct IEEE 754 semantics
- The `propagate_nulls_binary` post-pass must be correct for all paths
- `clear_null_comparisons` must handle all comparison types
- Div-by-zero post-pass adds O(n) scan of divisor — negligible vs the O(n) computation
