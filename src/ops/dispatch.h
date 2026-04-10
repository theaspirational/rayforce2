/*
 *   Typed vector dispatch — rayforce1 pattern
 *
 *   ray_binop_map / ray_unop_map / ray_unop_fold handle:
 *     - atom fast path (direct call to partial_fn)
 *     - vector length check
 *     - output type inference (probe first element)
 *     - rc==1 buffer reuse
 *     - parallel dispatch for large vectors
 *     - per-element fallback for slices/lists
 */

#ifndef RAY_DISPATCH_H
#define RAY_DISPATCH_H

#include "rayforce.h"

/* Type-pair key for switch dispatch in partial functions.
 * Atoms have negative type, vectors have positive type. */
#define MTYPE2(a, b) (((int)(a) + 128) * 256 + ((int)(b) + 128))

/* Sentinel: partial returns this to signal "type combo not handled" */
#define RAY_PARTIAL_UNSUPPORTED ((ray_t*)(uintptr_t)1)

/* Binary partial function: operates on chunk [offset, offset+len).
 * - atom×atom: len=0, offset=0, out=NULL → returns result atom
 * - vec ops:   writes into out[offset..offset+len) → returns NULL on success
 * - not handled: returns RAY_PARTIAL_UNSUPPORTED
 * - error:     returns ray_error(...) */
typedef ray_t* (*ray_binop_partial_fn)(ray_t* x, ray_t* y,
                                        int64_t len, int64_t offset, ray_t* out);

/* Unary partial function: same pattern.
 * - atom:     len=0, offset=0, out=NULL → returns result atom
 * - vec ops:  writes into out[offset..offset+len) → returns NULL */
typedef ray_t* (*ray_unop_partial_fn)(ray_t* x,
                                       int64_t len, int64_t offset, ray_t* out);

/* Binary element-wise dispatch: handles atoms, vectors, parallel. */
ray_t* ray_binop_map(ray_binop_partial_fn partial, ray_t* x, ray_t* y);

/* Unary element-wise dispatch. */
ray_t* ray_unop_map(ray_unop_partial_fn partial, ray_t* x);

/* Unary reduction dispatch (sum, min, max, avg). */
ray_t* ray_unop_fold(ray_unop_partial_fn partial, ray_t* x);

#endif
