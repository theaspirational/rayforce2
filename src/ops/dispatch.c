/*
 *   Typed vector dispatch — parallel, rc==1 reuse, slice-safe.
 *   See dispatch.h for API documentation.
 */

#include "ops/dispatch.h"
#include "core/pool.h"
#include "mem/heap.h"
#include "core/types.h"

/* ── Parallel worker contexts ──────────────────────────────────── */

typedef struct {
    ray_binop_partial_fn partial;
    ray_t* x;
    ray_t* y;
    ray_t* out;
} binop_ctx_t;

static void binop_worker(void* ctx_, uint32_t wid, int64_t start, int64_t end) {
    (void)wid;
    binop_ctx_t* c = (binop_ctx_t*)ctx_;
    c->partial(c->x, c->y, end - start, start, c->out);
}

typedef struct {
    ray_unop_partial_fn partial;
    ray_t* x;
    ray_t* out;
} unop_ctx_t;

static void unop_worker(void* ctx_, uint32_t wid, int64_t start, int64_t end) {
    (void)wid;
    unop_ctx_t* c = (unop_ctx_t*)ctx_;
    c->partial(c->x, end - start, start, c->out);
}

/* ── Helpers ───────────────────────────────────────────────────── */

/* Infer output type by probing the partial function on the first element.
 * Returns the positive vector type (e.g., RAY_I64, RAY_F64, RAY_BOOL). */
/* Infer output type from operand types.
 * Returns 0 if the type combination isn't handleable. */
static int8_t infer_binop_type(ray_t* x, ray_t* y) {
    int8_t xt = ray_is_vec(x) ? x->type : -(x->type);
    int8_t yt = ray_is_vec(y) ? y->type : -(y->type);
    /* Same type → same output */
    if (xt == yt) return xt;
    /* Only handle numeric types */
    return 0;  /* mixed types → generic path handles promotion */
}

/* ── ray_binop_map ─────────────────────────────────────────────── */

ray_t* ray_binop_map(ray_binop_partial_fn partial, ray_t* x, ray_t* y) {
    bool xv = ray_is_vec(x), yv = ray_is_vec(y);
    bool xa = ray_is_atom(x), ya = ray_is_atom(y);

    /* Both atoms → direct call */
    if (xa && ya)
        return partial(x, y, 0, 0, NULL);

    /* Need at least one vector */
    if (!xv && !yv)
        return NULL;  /* can't handle (e.g., list inputs) */

    int64_t len;
    if (xv && yv)
        len = x->len < y->len ? x->len : y->len;  /* zip-map: truncate to shorter */
    else
        len = xv ? x->len : y->len;
    if (len == 0) {
        /* For same-type non-temporal, we know the output type.
         * Temporal ops fall through — output type is op-dependent
         * (e.g., DATE-DATE→I32, DATE+int→DATE). */
        int8_t xt = xv ? x->type : -(x->type);
        int8_t yt = yv ? y->type : -(y->type);
        bool temporal = (xt==RAY_DATE||xt==RAY_TIME||xt==RAY_TIMESTAMP||
                         yt==RAY_DATE||yt==RAY_TIME||yt==RAY_TIMESTAMP);
        if (!temporal && xt == yt)
            return ray_vec_new(xt, 0);
        return NULL;
    }

    /* Reject slices — ray_data() doesn't work on them */
    if (xv && (x->attrs & RAY_ATTR_SLICE)) return NULL;
    if (yv && (y->attrs & RAY_ATTR_SLICE)) return NULL;

    /* Reject nulls — per-element null propagation needs the slow path */
    if (xv && (x->attrs & RAY_ATTR_HAS_NULLS)) return NULL;
    if (yv && (y->attrs & RAY_ATTR_HAS_NULLS)) return NULL;
    if (xa && RAY_ATOM_IS_NULL(x)) return NULL;
    if (ya && RAY_ATOM_IS_NULL(y)) return NULL;

    /* Reject temporal types — output type rules are operation-dependent */
    int8_t xt = xv ? x->type : -(x->type);
    int8_t yt = yv ? y->type : -(y->type);
    bool x_temp = (xt == RAY_DATE || xt == RAY_TIME || xt == RAY_TIMESTAMP);
    bool y_temp = (yt == RAY_DATE || yt == RAY_TIME || yt == RAY_TIMESTAMP);
    if (x_temp || y_temp) return NULL;

    /* Infer output type — require same type for now.
     * Mixed-width type promotion has complex q semantics
     * (scalar broadcast preserves vector type) that the
     * generic path in eval.c handles correctly. */
    int8_t ot = infer_binop_type(x, y);
    if (ot == 0) return NULL;
    /* Require both operands to match output type */
    if (xv && x->type != ot) return NULL;
    if (yv && y->type != ot) return NULL;
    if (xa && -(x->type) != ot) return NULL;
    if (ya && -(y->type) != ot) return NULL;

    /* Probe: verify the partial handles this type combo BEFORE allocating.
     * Use a tiny temp vector so we don't corrupt a rc==1 reused input. */
    {
        ray_t* tmp = ray_vec_new(ot, 1);
        if (!tmp || RAY_IS_ERR(tmp)) return NULL;
        tmp->len = 1;
        ray_t* probe = partial(x, y, 1, 0, tmp);
        ray_release(tmp);
        if (probe) {
            if (probe == RAY_PARTIAL_UNSUPPORTED) return NULL;
            if (RAY_IS_ERR(probe)) return probe;
            ray_release(probe);
            return NULL;
        }
    }

    /* Allocate output: rc==1 reuse when type matches */
    ray_t* out;
    if (xv && x->rc == 1 && x->type == ot && !(x->attrs & RAY_ATTR_HAS_NULLS))
        { out = x; ray_retain(out); }
    else if (yv && y->rc == 1 && y->type == ot && !(y->attrs & RAY_ATTR_HAS_NULLS))
        { out = y; ray_retain(out); }
    else
        out = ray_vec_new(ot, len);
    if (!out || RAY_IS_ERR(out)) return out;
    out->len = len;

    /* Dispatch full range */
    {
        ray_pool_t* pool = ray_pool_get();
        binop_ctx_t ctx = { .partial = partial, .x = x, .y = y, .out = out };
        if (pool && len >= RAY_PARALLEL_THRESHOLD)
            ray_pool_dispatch(pool, binop_worker, &ctx, len);
        else
            partial(x, y, len, 0, out);
    }

    return out;
}

/* ── ray_unop_map ──────────────────────────────────────────────── */

ray_t* ray_unop_map(ray_unop_partial_fn partial, ray_t* x) {
    if (ray_is_atom(x))
        return partial(x, 0, 0, NULL);

    if (!ray_is_vec(x))
        return NULL;  /* can't handle lists */

    int64_t len = x->len;
    if (len == 0) return ray_vec_new(x->type, 0);

    if (x->attrs & RAY_ATTR_SLICE) return NULL;

    /* Probe output type */
    ray_t* probe = partial(x, 0, 0, NULL);
    if (!probe || RAY_IS_ERR(probe)) return probe;
    int8_t ot = -(probe->type);
    ray_release(probe);

    /* Allocate with rc==1 reuse */
    ray_t* out;
    if (x->rc == 1 && x->type == ot && !(x->attrs & RAY_ATTR_HAS_NULLS))
        { out = x; ray_retain(out); }
    else
        out = ray_vec_new(ot, len);
    if (!out || RAY_IS_ERR(out)) return out;
    out->len = len;

    ray_pool_t* pool = ray_pool_get();
    unop_ctx_t ctx = { .partial = partial, .x = x, .out = out };
    if (pool && len >= RAY_PARALLEL_THRESHOLD)
        ray_pool_dispatch(pool, unop_worker, &ctx, len);
    else
        partial(x, len, 0, out);

    return out;
}

/* ── ray_unop_fold ─────────────────────────────────────────────── */

ray_t* ray_unop_fold(ray_unop_partial_fn partial, ray_t* x) {
    if (ray_is_atom(x))
        return partial(x, 0, 0, NULL);

    if (!ray_is_vec(x))
        return NULL;

    int64_t len = x->len;
    if (len == 0) return ray_i64(0);

    if (x->attrs & RAY_ATTR_SLICE) return NULL;

    /* Single-threaded fold — partial returns a scalar */
    return partial(x, len, 0, NULL);
}
