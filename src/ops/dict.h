/*
 *   Copyright (c) 2025-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 */

#ifndef RAY_OPS_DICT_H
#define RAY_OPS_DICT_H

/*
 * dict.h -- shared helpers for the RAY_LIST + RAY_ATTR_DICT pair-array
 * representation of dicts.  A dict is a RAY_LIST whose items alternate
 * as [k0, v0, k1, v1, ...].  Keys are sym atoms (type -RAY_SYM).
 */

#include <rayforce.h>
#include "lang/eval.h"   /* RAY_ATTR_DICT */
#include "mem/heap.h"    /* ray_cow */

/* Probe dict by sym_id.  Returns the value pointer (not retained — caller
 * must retain if keeping).  Returns NULL if dict is NULL, not a dict, or
 * the key is missing.  Does not allocate.  Extracted from the existing
 * (at dict key) path in collection.c for reuse by env path-resolution. */
static inline ray_t* dict_probe_by_sym(ray_t* dict, int64_t key_sym) {
    if (!dict || RAY_IS_ERR(dict)) return NULL;
    if (dict->type != RAY_LIST || !(dict->attrs & RAY_ATTR_DICT)) return NULL;
    ray_t** items = (ray_t**)ray_data(dict);
    int64_t n = dict->len;
    for (int64_t i = 0; i < n; i += 2) {
        ray_t* k = items[i];
        if (k && k->type == -RAY_SYM && k->i64 == key_sym) {
            return items[i + 1];
        }
    }
    return NULL;
}

/* Probe a dict, table, or other sym-indexable container by sym_id.  Used
 * by env-path resolution to walk through `t.OrderId` where `t` can be a
 * table or a namespace dict.  Returns a borrowed pointer or NULL.
 * Containers handled:
 *   - RAY_LIST + RAY_ATTR_DICT: pair-array lookup (dict_probe_by_sym).
 *   - RAY_TABLE: schema-based column lookup (ray_table_get_col).
 * Anything else returns NULL so the caller can surface an error. */
static inline ray_t* container_probe_by_sym(ray_t* v, int64_t key_sym) {
    if (!v || RAY_IS_ERR(v)) return NULL;
    if (v->type == RAY_LIST && (v->attrs & RAY_ATTR_DICT))
        return dict_probe_by_sym(v, key_sym);
    if (v->type == RAY_TABLE)
        return ray_table_get_col(v, key_sym);
    return NULL;
}

/* Remove `key_sym` from `dict`.  Ownership of `dict` is always consumed
 * (same contract as dict_upsert): on success the input ref is transferred
 * into the returned dict (via ray_cow); on error it is released.  If the
 * key isn't present, returns the input unchanged (one-ref transferred).
 * Callers get a ray_t* they own with rc=1 (or a shared rc if the key was
 * absent and no COW happened). */
static inline ray_t* dict_remove(ray_t* dict, int64_t key_sym) {
    if (!dict || RAY_IS_ERR(dict)) return dict;
    if (dict->type != RAY_LIST || !(dict->attrs & RAY_ATTR_DICT)) {
        ray_release(dict);
        return ray_error("type", NULL);
    }

    /* Probe without cloning to decide whether we need to COW at all. */
    ray_t** items = (ray_t**)ray_data(dict);
    int64_t found = -1;
    for (int64_t i = 0; i < dict->len; i += 2) {
        ray_t* k = items[i];
        if (k && k->type == -RAY_SYM && k->i64 == key_sym) {
            found = i;
            break;
        }
    }
    if (found < 0) return dict;   /* no change — pass the owned ref back */

    ray_t* out = ray_cow(dict);
    if (!out || RAY_IS_ERR(out)) {
        ray_release(dict);
        return out ? out : ray_error("oom", NULL);
    }
    out->attrs |= RAY_ATTR_DICT;

    ray_t** out_items = (ray_t**)ray_data(out);
    if (out_items[found])     ray_release(out_items[found]);
    if (out_items[found + 1]) ray_release(out_items[found + 1]);

    /* Shift remaining pairs left. */
    for (int64_t i = found; i + 2 < out->len; i++) {
        out_items[i] = out_items[i + 2];
    }
    out->len -= 2;
    return out;
}

/* Upsert (key_sym -> val) into `dict`.  If `dict` is NULL (or not a dict),
 * starts a fresh empty dict and inserts.  Returns a ray_t* with rc=1 owned
 * by the caller.  Does NOT consume `val` (internally retained via
 * ray_list_append / retain-replace).
 *
 * Ownership: `dict` is always consumed (one ref).  On success, the ref is
 * transferred into the returned object (via ray_cow).  On error, the ref
 * is explicitly released here so the caller does not need to guard.
 *
 * Existing-key path: COW the dict, release old value, retain new value,
 * replace in place.  Missing-key path: COW (or fresh), append key atom +
 * value. */
static inline ray_t* dict_upsert(ray_t* dict, int64_t key_sym, ray_t* val) {
    bool need_new = (!dict || dict->type != RAY_LIST || !(dict->attrs & RAY_ATTR_DICT));
    ray_t* out = NULL;
    if (need_new) {
        if (dict) ray_release(dict);   /* non-dict input: consume & replace */
        out = ray_list_new(2);
        if (!out || RAY_IS_ERR(out)) return out ? out : ray_error("oom", NULL);
        out->attrs |= RAY_ATTR_DICT;
    } else {
        /* Existing-key fast path: probe without cloning to pick the branch. */
        ray_t** items0 = (ray_t**)ray_data(dict);
        for (int64_t i = 0; i < dict->len; i += 2) {
            ray_t* k = items0[i];
            if (k && k->type == -RAY_SYM && k->i64 == key_sym) {
                out = ray_cow(dict);
                if (!out || RAY_IS_ERR(out)) {
                    ray_release(dict);
                    return out ? out : ray_error("oom", NULL);
                }
                out->attrs |= RAY_ATTR_DICT;
                ray_t** items = (ray_t**)ray_data(out);
                ray_t* old = items[i + 1];
                ray_retain(val);
                items[i + 1] = val;
                if (old) ray_release(old);
                return out;
            }
        }
        /* Missing key — need to append */
        out = ray_cow(dict);
        if (!out || RAY_IS_ERR(out)) {
            ray_release(dict);
            return out ? out : ray_error("oom", NULL);
        }
        out->attrs |= RAY_ATTR_DICT;
    }

    /* Append (key_atom, val).  ray_list_append consumes `out` — error path
     * returns the error pointer; matching the existing codebase idiom in
     * builtins.c:1057-1063. */
    ray_t* key_atom = ray_sym(key_sym);
    if (!key_atom || RAY_IS_ERR(key_atom)) {
        ray_release(out);
        return key_atom ? key_atom : ray_error("oom", NULL);
    }
    out = ray_list_append(out, key_atom);
    ray_release(key_atom);
    if (!out || RAY_IS_ERR(out)) return out ? out : ray_error("oom", NULL);
    out->attrs |= RAY_ATTR_DICT;  /* preserve DICT flag across realloc */

    out = ray_list_append(out, val);
    if (!out || RAY_IS_ERR(out)) return out ? out : ray_error("oom", NULL);
    out->attrs |= RAY_ATTR_DICT;
    return out;
}

#endif /* RAY_OPS_DICT_H */
