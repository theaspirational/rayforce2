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

#include "lang/eval_internal.h"
#include "core/pool.h"
#include "mem/heap.h"
#include "mem/cow.h"

/* ══════════════════════════════════════════
 * Typed vector arithmetic — rayforce1 pattern
 *
 * Each operation dispatches on MTYPE2(left_type, right_type) once,
 * then runs a tight typed-pointer loop that the compiler vectorizes.
 * Output buffer reuses input when rc==1 and types match.
 * ══════════════════════════════════════════ */

#define MTYPE2(a, b) (((int)(a) + 128) * 256 + ((int)(b) + 128))

/* Typed loop macros: atom+vec, vec+atom, vec+vec.
 * lt/rt/ot = element C types, OP = binary expression macro. */
#define LOOP_A_V(lval, rptr, optr, n, OP)              \
    for (int64_t _i = 0; _i < (n); _i++)               \
        (optr)[_i] = OP((lval), (rptr)[_i]);

#define LOOP_V_A(lptr, rval, optr, n, OP)              \
    for (int64_t _i = 0; _i < (n); _i++)               \
        (optr)[_i] = OP((lptr)[_i], (rval));

#define LOOP_V_V(lptr, rptr, optr, n, OP)              \
    for (int64_t _i = 0; _i < (n); _i++)               \
        (optr)[_i] = OP((lptr)[_i], (rptr)[_i]);

/* Op macros — expand to typed expressions the compiler can vectorize */
#define OP_ADD_I64(a,b) ((int64_t)((uint64_t)(a)+(uint64_t)(b)))
#define OP_SUB_I64(a,b) ((int64_t)((uint64_t)(a)-(uint64_t)(b)))
#define OP_MUL_I64(a,b) ((int64_t)((uint64_t)(a)*(uint64_t)(b)))
#define OP_ADD_I32(a,b) ((int32_t)((uint32_t)(a)+(uint32_t)(b)))
#define OP_SUB_I32(a,b) ((int32_t)((uint32_t)(a)-(uint32_t)(b)))
#define OP_MUL_I32(a,b) ((int32_t)((uint32_t)(a)*(uint32_t)(b)))
#define OP_ADD_F64(a,b) ((a)+(b))
#define OP_SUB_F64(a,b) ((a)-(b))
#define OP_MUL_F64(a,b) ((a)*(b))
#define OP_EQ_I64(a,b)  ((uint8_t)((a)==(b)))
#define OP_NE_I64(a,b)  ((uint8_t)((a)!=(b)))
#define OP_LT_I64(a,b)  ((uint8_t)((a)<(b)))
#define OP_LE_I64(a,b)  ((uint8_t)((a)<=(b)))
#define OP_GT_I64(a,b)  ((uint8_t)((a)>(b)))
#define OP_GE_I64(a,b)  ((uint8_t)((a)>=(b)))
#define OP_EQ_F64(a,b)  ((uint8_t)((a)==(b)))
#define OP_NE_F64(a,b)  ((uint8_t)((a)!=(b)))
#define OP_LT_F64(a,b)  ((uint8_t)((a)<(b)))
#define OP_LE_F64(a,b)  ((uint8_t)((a)<=(b)))
#define OP_GT_F64(a,b)  ((uint8_t)((a)>(b)))
#define OP_GE_F64(a,b)  ((uint8_t)((a)>=(b)))
/* MIN2/MAX2 */
#define OP_MIN2_I64(a,b) ((a)<(b)?(a):(b))
#define OP_MAX2_I64(a,b) ((a)>(b)?(a):(b))
#define OP_MIN2_F64(a,b) ((a)<(b)?(a):(b))
#define OP_MAX2_F64(a,b) ((a)>(b)?(a):(b))

/* Context for parallel typed dispatch */
typedef struct {
    ray_t* left;
    ray_t* right;
    ray_t* out;
    uint16_t opcode;
} binop_vec_ctx_t;

/* Emit typed loop for a single type width.
 * T=C type, W=width tag, SV_EXPR=scalar read expression */
#define TYPED_LOOP(T, lptr, rptr, lsv, rsv, optr, n, xv, yv, OPNAME)    \
    do {                                                                  \
        if (xv && yv)    LOOP_V_V(lptr, rptr, optr, n, OPNAME)           \
        else if (xv)     LOOP_V_A(lptr, rsv,  optr, n, OPNAME)           \
        else             LOOP_A_V(lsv,  rptr, optr, n, OPNAME)           \
    } while(0)

/* Parallel worker: typed dispatch per chunk */
static void binop_vec_worker(void* ctx_, uint32_t wid, int64_t start, int64_t end) {
    (void)wid;
    binop_vec_ctx_t* c = (binop_vec_ctx_t*)ctx_;
    int64_t n = end - start;
    ray_t* x = c->left;
    ray_t* y = c->right;
    ray_t* out = c->out;
    int8_t ot = out->type;
    bool xv = ray_is_vec(x), yv = ray_is_vec(y);
    uint16_t opc = c->opcode;

    /* For arithmetic: output type matches input type.
     * For comparisons: inputs are I64/I32/F64, output is BOOL (U8). */
    bool is_cmp = (opc >= OP_EQ && opc <= OP_GE);

    /* Resolve data pointers once */
    if (is_cmp) {
        /* Comparison: read inputs at their type, write BOOL output */
        uint8_t* restrict od = (uint8_t*)ray_data(out) + start;
        /* Determine input type from the vector operand */
        int8_t it = xv ? x->type : yv ? y->type : RAY_I64;
        if (it == RAY_I64 || it == RAY_TIMESTAMP) {
            int64_t* ld = xv ? (int64_t*)ray_data(x) + start : NULL;
            int64_t* rd = yv ? (int64_t*)ray_data(y) + start : NULL;
            int64_t lsv = xv ? 0 : x->i64;
            int64_t rsv = yv ? 0 : y->i64;
            switch (opc) {
                case OP_EQ: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_EQ_I64); break;
                case OP_NE: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_NE_I64); break;
                case OP_LT: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_LT_I64); break;
                case OP_LE: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_LE_I64); break;
                case OP_GT: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_GT_I64); break;
                case OP_GE: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_GE_I64); break;
                default: break;
            }
        } else if (it == RAY_I32 || it == RAY_DATE || it == RAY_TIME) {
            int32_t* ld = xv ? (int32_t*)ray_data(x) + start : NULL;
            int32_t* rd = yv ? (int32_t*)ray_data(y) + start : NULL;
            int32_t lsv = xv ? 0 : (int32_t)x->i32;
            int32_t rsv = yv ? 0 : (int32_t)y->i32;
            switch (opc) {
                case OP_EQ: TYPED_LOOP(int32_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_EQ_I64); break;
                case OP_NE: TYPED_LOOP(int32_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_NE_I64); break;
                case OP_LT: TYPED_LOOP(int32_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_LT_I64); break;
                case OP_LE: TYPED_LOOP(int32_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_LE_I64); break;
                case OP_GT: TYPED_LOOP(int32_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_GT_I64); break;
                case OP_GE: TYPED_LOOP(int32_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_GE_I64); break;
                default: break;
            }
        } else if (it == RAY_F64) {
            double* ld = xv ? (double*)ray_data(x) + start : NULL;
            double* rd = yv ? (double*)ray_data(y) + start : NULL;
            double lsv = xv ? 0 : x->f64;
            double rsv = yv ? 0 : y->f64;
            switch (opc) {
                case OP_EQ: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_EQ_F64); break;
                case OP_NE: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_NE_F64); break;
                case OP_LT: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_LT_F64); break;
                case OP_LE: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_LE_F64); break;
                case OP_GT: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_GT_F64); break;
                case OP_GE: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_GE_F64); break;
                default: break;
            }
        }
    } else if (ot == RAY_I64 || ot == RAY_TIMESTAMP) {
        int64_t* restrict od = (int64_t*)ray_data(out) + start;
        int64_t* ld = xv ? (int64_t*)ray_data(x) + start : NULL;
        int64_t* rd = yv ? (int64_t*)ray_data(y) + start : NULL;
        int64_t lsv = xv ? 0 : x->i64, rsv = yv ? 0 : y->i64;
        switch (opc) {
            case OP_ADD: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_ADD_I64); break;
            case OP_SUB: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_SUB_I64); break;
            case OP_MUL: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_MUL_I64); break;
            case OP_MIN2: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_MIN2_I64); break;
            case OP_MAX2: TYPED_LOOP(int64_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_MAX2_I64); break;
            default: break;
        }
    } else if (ot == RAY_I32 || ot == RAY_DATE || ot == RAY_TIME) {
        int32_t* restrict od = (int32_t*)ray_data(out) + start;
        int32_t* ld = xv ? (int32_t*)ray_data(x) + start : NULL;
        int32_t* rd = yv ? (int32_t*)ray_data(y) + start : NULL;
        int32_t lsv = xv ? 0 : (int32_t)x->i32, rsv = yv ? 0 : (int32_t)y->i32;
        switch (opc) {
            case OP_ADD: TYPED_LOOP(int32_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_ADD_I32); break;
            case OP_SUB: TYPED_LOOP(int32_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_SUB_I32); break;
            case OP_MUL: TYPED_LOOP(int32_t,ld,rd,lsv,rsv,od,n,xv,yv,OP_MUL_I32); break;
            default: break;
        }
    } else if (ot == RAY_F64) {
        double* restrict od = (double*)ray_data(out) + start;
        double* ld = xv ? (double*)ray_data(x) + start : NULL;
        double* rd = yv ? (double*)ray_data(y) + start : NULL;
        double lsv = xv ? 0 : x->f64, rsv = yv ? 0 : y->f64;
        switch (opc) {
            case OP_ADD: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_ADD_F64); break;
            case OP_SUB: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_SUB_F64); break;
            case OP_MUL: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_MUL_F64); break;
            case OP_MIN2: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_MIN2_F64); break;
            case OP_MAX2: TYPED_LOOP(double,ld,rd,lsv,rsv,od,n,xv,yv,OP_MAX2_F64); break;
            default: break;
        }
    }
    #undef TYPED_LOOP
}

/* Infer output type for arithmetic on two operands */
static int8_t infer_arith_type(ray_t* x, ray_t* y) {
    int8_t xt = ray_is_atom(x) ? -(x->type) : x->type;
    int8_t yt = ray_is_atom(y) ? -(y->type) : y->type;
    if (xt == RAY_F64 || yt == RAY_F64) return RAY_F64;
    if (xt == RAY_I64 || yt == RAY_I64) return RAY_I64;
    if (xt == RAY_TIMESTAMP || yt == RAY_TIMESTAMP) return RAY_TIMESTAMP;
    if (xt == RAY_I32 || yt == RAY_I32) return RAY_I32;
    if (xt == RAY_DATE || yt == RAY_DATE) return RAY_DATE;
    if (xt == RAY_TIME || yt == RAY_TIME) return RAY_TIME;
    if (xt == RAY_I16 || yt == RAY_I16) return RAY_I16;
    return RAY_I64;
}

/* Fast vector binary op: typed dispatch + rc==1 reuse + parallel.
 * Returns NULL when the fast path doesn't apply (caller falls through).
 * opcode is the DAG opcode (OP_ADD, OP_SUB, OP_MUL, OP_EQ, OP_LT, etc.) */
ray_t* binop_vec(ray_t* x, ray_t* y, uint16_t opcode) {
    bool xv = ray_is_vec(x), yv = ray_is_vec(y);
    bool xa = ray_is_atom(x), ya = ray_is_atom(y);
    if (!(xv || xa) || !(yv || ya) || (!xv && !yv)) return NULL;

    /* Skip when nulls present — generic path handles null propagation */
    if (xv && (x->attrs & RAY_ATTR_HAS_NULLS)) return NULL;
    if (yv && (y->attrs & RAY_ATTR_HAS_NULLS)) return NULL;
    if (xa && RAY_ATOM_IS_NULL(x)) return NULL;
    if (ya && RAY_ATOM_IS_NULL(y)) return NULL;

    int64_t len = xv ? x->len : y->len;
    if (xv && yv && x->len != y->len) return NULL;

    /* Determine input type — both operands must match */
    int8_t xt = xv ? x->type : -(x->type);
    int8_t yt = yv ? y->type : -(y->type);
    /* Skip temporal types — output depends on op (DATE-DATE→I32 etc.) */
    bool x_temporal = (xt == RAY_DATE || xt == RAY_TIME || xt == RAY_TIMESTAMP);
    bool y_temporal = (yt == RAY_DATE || yt == RAY_TIME || yt == RAY_TIMESTAMP);
    if (x_temporal || y_temporal) return NULL;
    /* Both operands must have the same type */
    if (xt != yt) return NULL;
    /* Only handle I64, I32, F64 */
    if (xt != RAY_I64 && xt != RAY_I32 && xt != RAY_F64) return NULL;
    /* Only handle opcodes with typed loop implementations */
    if (opcode != OP_ADD && opcode != OP_SUB && opcode != OP_MUL &&
        opcode != OP_MIN2 && opcode != OP_MAX2 &&
        opcode != OP_EQ && opcode != OP_NE && opcode != OP_LT &&
        opcode != OP_LE && opcode != OP_GT && opcode != OP_GE)
        return NULL;

    /* Output type: BOOL for comparisons, same as input for arithmetic */
    bool is_cmp = (opcode >= OP_EQ && opcode <= OP_GE);
    int8_t ot = is_cmp ? RAY_BOOL : xt;

    /* rc==1 buffer reuse (arithmetic only, not slices — slices alias
     * their parent's buffer so writing into them corrupts the parent) */
    ray_t* out;
    if (!is_cmp && xv && x->rc == 1 && x->type == ot &&
        !(x->attrs & RAY_ATTR_SLICE)) {
        out = x;
        ray_retain(out);
    } else if (!is_cmp && yv && y->rc == 1 && y->type == ot &&
               !(y->attrs & RAY_ATTR_SLICE)) {
        out = y;
        ray_retain(out);
    } else {
        out = ray_vec_new(ot, len);
    }
    if (!out || RAY_IS_ERR(out)) return out;
    out->len = len;

    binop_vec_ctx_t ctx = { .left = x, .right = y, .out = out, .opcode = opcode };
    ray_pool_t* pool = ray_pool_get();
    if (pool && len >= RAY_PARALLEL_THRESHOLD)
        ray_pool_dispatch(pool, binop_vec_worker, &ctx, len);
    else
        binop_vec_worker(&ctx, 0, 0, len);

    return out;
}

/* Binary arithmetic */
ray_t* ray_add_fn(ray_t* a, ray_t* b) {

    /* Temporal + integer arithmetic (only int types, not float) */
    if (is_temporal(a) && is_numeric(b) && b->type != -RAY_F64) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b))
            return ray_typed_null(a->type);

        int64_t v = as_i64(b);
        if (a->type == -RAY_DATE)      return ray_date(a->i64 + v);
        if (a->type == -RAY_TIME)      return ray_time(a->i64 + v);
        if (a->type == -RAY_TIMESTAMP) return ray_timestamp(a->i64 + v);
    }
    if (is_numeric(a) && a->type != -RAY_F64 && is_temporal(b)) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b))
            return ray_typed_null(b->type);

        int64_t v = as_i64(a);
        if (b->type == -RAY_DATE)      return ray_date(b->i64 + v);
        if (b->type == -RAY_TIME)      return ray_time(b->i64 + v);
        if (b->type == -RAY_TIMESTAMP) return ray_timestamp(b->i64 + v);
    }
    /* Reject float + temporal */
    if ((a->type == -RAY_F64 && is_temporal(b)) || (is_temporal(a) && b->type == -RAY_F64))
        return ray_error("type", NULL);
    /* Reject null_numeric + temporal (for null floats etc) */
    if (is_numeric(a) && RAY_ATOM_IS_NULL(a) && is_temporal(b))
        return ray_error("type", NULL);
    if (is_temporal(a) && is_numeric(b) && RAY_ATOM_IS_NULL(b))
        return ray_error("type", NULL);
    /* DATE + TIME → TIMESTAMP */
    if (a->type == -RAY_DATE && b->type == -RAY_TIME) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIMESTAMP);
        return ray_timestamp(a->i64 * 86400000000000LL + b->i64 * 1000000LL);
    }
    if (a->type == -RAY_TIME && b->type == -RAY_DATE) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIMESTAMP);
        return ray_timestamp(b->i64 * 86400000000000LL + a->i64 * 1000000LL);
    }
    /* TIME + TIME → TIME */
    if (a->type == -RAY_TIME && b->type == -RAY_TIME) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIME);
        return ray_time(a->i64 + b->i64);
    }
    /* TIME + TIMESTAMP → TIMESTAMP (add ms as ns) */
    if (a->type == -RAY_TIME && b->type == -RAY_TIMESTAMP) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIMESTAMP);
        return ray_timestamp(b->i64 + a->i64 * 1000000LL);
    }
    if (a->type == -RAY_TIMESTAMP && b->type == -RAY_TIME) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIMESTAMP);
        return ray_timestamp(a->i64 + b->i64 * 1000000LL);
    }

    if (!is_numeric(a) || !is_numeric(b))
        return ray_error("type", "cannot add %s and %s",
                         ray_type_name(abs(a->type)), ray_type_name(abs(b->type)));
    /* Null propagation */
    if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return null_for_promoted(a, b);
    if (is_float_op(a, b)) return make_f64(as_f64(a) + as_f64(b));
    int8_t rt = promote_int_type(a, b);
    return make_typed_int(rt, as_i64(a) + as_i64(b));
}

ray_t* ray_sub_fn(ray_t* a, ray_t* b) {

    /* Temporal - int null propagation (both operands) */
    if (is_temporal(a) && is_numeric(b)) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b))
            return ray_typed_null(a->type);
    }
    if (is_numeric(a) && is_temporal(b)) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b))
            return ray_typed_null(b->type);
    }
    /* DATE - int → DATE */
    if (a->type == -RAY_DATE && is_numeric(b)) {
        return ray_date(a->i64 - as_i64(b));
    }
    /* DATE - DATE → i32 (days difference) */
    if (a->type == -RAY_DATE && b->type == -RAY_DATE) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_I32);
        return ray_i32((int32_t)(a->i64 - b->i64));
    }
    /* DATE - TIME → TIMESTAMP */
    if (a->type == -RAY_DATE && b->type == -RAY_TIME) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIMESTAMP);
        return ray_timestamp(a->i64 * 86400000000000LL - b->i64 * 1000000LL);
    }
    /* TIME - int → TIME */
    if (a->type == -RAY_TIME && is_numeric(b)) {
        return ray_time(a->i64 - as_i64(b));
    }
    /* int - TIME → TIME (negative) */
    if (is_numeric(a) && b->type == -RAY_TIME) {
        return ray_time(as_i64(a) - b->i64);
    }
    /* TIME - TIME → TIME */
    if (a->type == -RAY_TIME && b->type == -RAY_TIME) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIME);
        return ray_time(a->i64 - b->i64);
    }
    /* TIMESTAMP - int → TIMESTAMP */
    if (a->type == -RAY_TIMESTAMP && is_numeric(b)) {
        return ray_timestamp(a->i64 - as_i64(b));
    }
    /* TIMESTAMP - TIME → TIMESTAMP */
    if (a->type == -RAY_TIMESTAMP && b->type == -RAY_TIME) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIMESTAMP);
        return ray_timestamp(a->i64 - b->i64 * 1000000LL);
    }
    /* TIMESTAMP - TIMESTAMP → int (nanos difference) */
    if (a->type == -RAY_TIMESTAMP && b->type == -RAY_TIMESTAMP) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_I64);
        return make_i64(a->i64 - b->i64);
    }
    /* TIMESTAMP - DATE → error */
    if (a->type == -RAY_TIMESTAMP && b->type == -RAY_DATE)
        return ray_error("type", NULL);

    if (!is_numeric(a) || !is_numeric(b))
        return ray_error("type", "cannot subtract %s and %s",
                         ray_type_name(abs(a->type)), ray_type_name(abs(b->type)));
    /* Null propagation */
    if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return null_for_promoted(a, b);
    if (is_float_op(a, b)) {
        double r = as_f64(a) - as_f64(b);
        if (r == 0.0) r = 0.0; /* normalize -0.0 to +0.0 */
        return make_f64(r);
    }
    int8_t rt = promote_int_type_right(a, b);
    return make_typed_int(rt, as_i64(a) - as_i64(b));
}

ray_t* ray_mul_fn(ray_t* a, ray_t* b) {

    /* int * TIME → TIME, TIME * int → TIME */
    if (is_numeric(a) && b->type == -RAY_TIME) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIME);
        return ray_time(as_i64(a) * b->i64);
    }
    if (a->type == -RAY_TIME && is_numeric(b)) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return ray_typed_null(-RAY_TIME);
        return ray_time(a->i64 * as_i64(b));
    }
    /* TIME * TIME → error */
    if (a->type == -RAY_TIME && b->type == -RAY_TIME)
        return ray_error("type", NULL);

    if (!is_numeric(a) || !is_numeric(b))
        return ray_error("type", "cannot multiply %s and %s",
                         ray_type_name(abs(a->type)), ray_type_name(abs(b->type)));
    /* Null propagation */
    if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) return null_for_promoted(a, b);
    if (is_float_op(a, b)) return make_f64(as_f64(a) * as_f64(b));
    int8_t rt = promote_int_type(a, b);
    return make_typed_int(rt, as_i64(a) * as_i64(b));
}

ray_t* ray_div_fn(ray_t* a, ray_t* b) {
    /* Temporal / numeric → temporal (same type as left operand) */
    if (is_temporal(a) && is_numeric(b)) {
        if (RAY_ATOM_IS_NULL(b) || RAY_ATOM_IS_NULL(a))
            return ray_typed_null(a->type);
        if (is_float_op(a, b)) {
            double bv = as_f64(b);
            if (bv == 0.0)
                return ray_typed_null(a->type);
            int64_t result = (int64_t)floor((double)a->i64 / bv);
            if (a->type == -RAY_TIME)      return ray_time(result);
            if (a->type == -RAY_DATE)      return ray_date(result);
            return ray_timestamp(result);
        }
        int64_t bv = as_i64(b);
        if (bv == 0)
            return ray_typed_null(a->type);
        int64_t av = a->i64;
        int64_t q = av / bv;
        if ((av ^ bv) < 0 && q * bv != av) q--;
        if (a->type == -RAY_TIME)      return ray_time(q);
        if (a->type == -RAY_DATE)      return ray_date(q);
        return ray_timestamp(q);
    }
    if (!is_numeric(a) || !is_numeric(b))
        return ray_error("type", "cannot divide %s by %s",
                         ray_type_name(abs(a->type)), ray_type_name(abs(b->type)));
    /* u8: unsigned byte division — div by 0 returns 0 */
    if (a->type == -RAY_U8) {
        uint8_t bv = (uint8_t)as_i64(b);
        if (bv == 0 || RAY_ATOM_IS_NULL(b)) return make_u8(0);
        if (RAY_ATOM_IS_NULL(a)) return make_u8(0);
        return make_u8((uint8_t)((uint8_t)as_i64(a) / bv));
    }
    /* Null propagation — null operand → typed null matching left operand type */
    if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b))
        return ray_typed_null(a->type);

    /* Integer (floor) division — always returns integer.
     * Float operands are converted to i64 via floor(a/b). */
    if (is_float_op(a, b)) {
        double bv = as_f64(b);
        if (bv == 0.0)
            return ray_typed_null(a->type);
        double result = floor(as_f64(a) / bv);
        /* Return type matches LEFT operand */
        if (a->type == -RAY_F64) return make_f64(result);
        if (a->type == -RAY_I16) return make_i16((int16_t)(int64_t)result);
        if (a->type == -RAY_I32) return make_i32((int32_t)(int64_t)result);
        if (result >= (double)INT64_MIN && result <= (double)INT64_MAX)
            return make_i64((int64_t)result);
        return ray_typed_null(-RAY_I64);
    }
    int64_t bv = as_i64(b);
    if (bv == 0)
        return ray_typed_null(a->type);

    int64_t av = as_i64(a);
    /* Floor division (toward -inf) */
    int64_t q = av / bv;
    if ((av ^ bv) < 0 && q * bv != av) q--;
    /* Return type matches LEFT operand for i16/i32 */
    if (a->type == -RAY_I16) return make_i16((int16_t)q);
    if (a->type == -RAY_I32) return make_i32((int32_t)q);
    return make_i64(q);
}

ray_t* ray_mod_fn(ray_t* a, ray_t* b) {
    /* Temporal % numeric → temporal (same type as left operand) */
    if (is_temporal(a) && is_numeric(b)) {
        if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b))
            return ray_typed_null(a->type);
        int64_t bv;
        if (b->type == -RAY_F64) {
            double bvf = b->f64;
            if (bvf == 0.0)
                return ray_typed_null(a->type);
            bv = (int64_t)bvf;
        } else {
            bv = as_i64(b);
        }
        if (bv == 0)
            return ray_typed_null(a->type);

        int64_t av = a->i64;
        int64_t q = av / bv;
        if ((av ^ bv) < 0 && q * bv != av) q--;
        int64_t result = av - bv * q;
        if (a->type == -RAY_TIME)      return ray_time(result);
        if (a->type == -RAY_DATE)      return ray_date(result);
        return ray_timestamp(result);
    }
    if (!is_numeric(a) || !is_numeric(b))
        return ray_error("type", "cannot mod %s by %s",
                         ray_type_name(abs(a->type)), ray_type_name(abs(b->type)));

    /* u8: unsigned byte modulo, no null sentinel — mod by 0 returns 0 */
    if (b->type == -RAY_U8) {
        uint8_t bv = b->u8;
        if (bv == 0) return make_u8(0);
        return make_u8((uint8_t)((uint8_t)as_i64(a) % bv));
    }
    if (a->type == -RAY_U8) {
        /* a is u8 but b is not u8 — treat as integer, result follows b's type */
    }

    /* Null propagation and division by zero: null type follows RIGHT operand */
    if (RAY_ATOM_IS_NULL(a) || RAY_ATOM_IS_NULL(b)) {
        int8_t rt = (b->type == -RAY_F64 || a->type == -RAY_F64) ? -RAY_F64 : b->type;
        return ray_typed_null(rt);
    }

    /* Float modulo: result = a - b * floor(a/b), type follows RIGHT or f64 */
    if (is_float_op(a, b)) {
        double av = as_f64(a), bv = as_f64(b);
        if (bv == 0.0) {
            int8_t rt = (b->type == -RAY_F64 || a->type == -RAY_F64) ? -RAY_F64 : b->type;
            return ray_typed_null(rt);
        }
        double result = av - bv * floor(av / bv);
        /* Snap tiny residual to 0 */
        if (fabs(result) < 1e-12 || fabs(result - fabs(bv)) < 1e-12) result = bv > 0 ? 0.0 : -0.0;
        if (b->type == -RAY_F64 || a->type == -RAY_F64) return make_f64(result);
        if (b->type == -RAY_I32) return make_i32((int32_t)(int64_t)result);
        if (b->type == -RAY_I16) return make_i16((int16_t)(int64_t)result);
        return make_i64((int64_t)result);
    }

    /* Integer modulo: result = a - b * floor(a/b), sign follows b (divisor) */
    int64_t av = as_i64(a), bv = as_i64(b);
    if (bv == 0)
        return ray_typed_null(b->type);

    int64_t q = av / bv;
    if ((av ^ bv) < 0 && q * bv != av) q--;  /* floor division */
    int64_t result = av - bv * q;
    /* Result type follows RIGHT operand */
    if (b->type == -RAY_I32) return make_i32((int32_t)result);
    if (b->type == -RAY_I16) return make_i16((int16_t)result);
    if (b->type == -RAY_U8) return make_u8((uint8_t)result);
    return make_i64(result);
}

ray_t* ray_neg_fn(ray_t* x) {
    if (RAY_ATOM_IS_NULL(x)) { ray_retain(x); return x; }
    if (x->type == -RAY_I64) return make_i64(-x->i64);
    if (x->type == -RAY_F64) return make_f64(-x->f64);
    return ray_error("type", NULL);
}

/* round: round to nearest integer (ties go away from zero), returns f64 */
ray_t* ray_round_fn(ray_t* x) {
    if (RAY_ATOM_IS_NULL(x)) return ray_typed_null(-RAY_F64);
    if (x->type == -RAY_F64) return make_f64(round(x->f64));
    if (is_numeric(x)) return make_f64(round(as_f64(x)));
    return ray_error("type", NULL);
}

/* floor: round toward -inf, returns f64 for f64, identity for int */
ray_t* ray_floor_fn(ray_t* x) {
    if (RAY_ATOM_IS_NULL(x)) { ray_retain(x); return x; }
    if (x->type == -RAY_F64) return make_f64(floor(x->f64));
    if (is_numeric(x)) { ray_retain(x); return x; }
    return ray_error("type", NULL);
}

/* ceil: round toward +inf, returns f64 for f64, identity for int */
ray_t* ray_ceil_fn(ray_t* x) {
    if (RAY_ATOM_IS_NULL(x)) { ray_retain(x); return x; }
    if (x->type == -RAY_F64) return make_f64(ceil(x->f64));
    if (is_numeric(x)) { ray_retain(x); return x; }
    return ray_error("type", NULL);
}

/* abs: absolute value, preserves type */
ray_t* ray_abs_fn(ray_t* x) {
    if (RAY_ATOM_IS_NULL(x)) { ray_retain(x); return x; }
    if (x->type == -RAY_F64) return make_f64(fabs(x->f64));
    if (x->type == -RAY_I64) return make_i64(x->i64 < 0 ? -x->i64 : x->i64);
    if (x->type == -RAY_I32) return make_i64(x->i32 < 0 ? -(int64_t)x->i32 : x->i32);
    if (x->type == -RAY_I16) return make_i64(x->i16 < 0 ? -(int64_t)x->i16 : x->i16);
    return ray_error("type", NULL);
}

/* sqrt: square root, returns f64 */
ray_t* ray_sqrt_fn(ray_t* x) {
    if (RAY_ATOM_IS_NULL(x)) return ray_typed_null(-RAY_F64);
    if (x->type == -RAY_F64) return make_f64(sqrt(x->f64));
    if (is_numeric(x)) return make_f64(sqrt(as_f64(x)));
    return ray_error("type", NULL);
}

/* log: natural logarithm, returns f64 */
ray_t* ray_log_fn(ray_t* x) {
    if (RAY_ATOM_IS_NULL(x)) return ray_typed_null(-RAY_F64);
    if (x->type == -RAY_F64) return make_f64(log(x->f64));
    if (is_numeric(x)) return make_f64(log(as_f64(x)));
    return ray_error("type", NULL);
}

/* exp: e^x, returns f64 */
ray_t* ray_exp_fn(ray_t* x) {
    if (RAY_ATOM_IS_NULL(x)) return ray_typed_null(-RAY_F64);
    if (x->type == -RAY_F64) return make_f64(exp(x->f64));
    if (is_numeric(x)) return make_f64(exp(as_f64(x)));
    return ray_error("type", NULL);
}
