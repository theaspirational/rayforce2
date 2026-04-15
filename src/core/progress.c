/*
 *   Copyright (c) 2025-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 *
 *   Pull-based progress reporting. Zero cost when no callback is
 *   registered; single main-thread pointer/int stores at sync points
 *   otherwise. Workers never touch this state.
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "rayforce.h"
#include <time.h>
#include <string.h>

static ray_progress_cb g_cb;
static void*           g_user;
static uint64_t        g_min_ms = 2000;
static uint64_t        g_tick_ms = 100;

/* Active-query state — only touched by the main executor thread.
 * A dedicated thread would need atomics, but since every writer is
 * the main thread we can use plain loads/stores. */
static const char* g_op_name;
static const char* g_phase;
static uint64_t    g_rows_done;
static uint64_t    g_rows_total;
static uint64_t    g_start_ns;
static uint64_t    g_last_fire_ns;
static bool        g_showing;

static inline uint64_t mono_ns(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC_COARSE
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void ray_progress_set_callback(ray_progress_cb cb, void* user,
                                uint64_t min_ms, uint64_t tick_interval_ms) {
    g_cb = cb;
    g_user = user;
    if (min_ms) g_min_ms = min_ms;
    if (tick_interval_ms) g_tick_ms = tick_interval_ms;
}

static void fire(uint64_t now_ns, bool final) {
    ray_progress_t snap = {
        .op_name     = g_op_name ? g_op_name : "",
        .phase       = g_phase ? g_phase : "",
        .rows_done   = g_rows_done,
        .rows_total  = g_rows_total,
        .elapsed_sec = (double)(now_ns - g_start_ns) / 1e9,
        .final       = final,
    };
    g_cb(&snap, g_user);
    g_last_fire_ns = now_ns;
    g_showing = true;
}

void ray_progress_update(const char* op_name, const char* phase,
                         uint64_t rows_done, uint64_t rows_total) {
    if (!g_cb) return;

    /* Lazy-start the query clock on first call after ray_progress_end
     * (or on very first call). Callers don't need a separate begin
     * hook — the first update sets the query start time. */
    if (g_start_ns == 0) {
        g_start_ns = mono_ns();
        g_last_fire_ns = 0;
        g_showing = false;
    }

    if (op_name)    g_op_name = op_name;
    if (phase)      g_phase = phase;
    if (rows_done)  g_rows_done = rows_done;
    if (rows_total) g_rows_total = rows_total;

    uint64_t now = mono_ns();
    uint64_t elapsed_ms = (now - g_start_ns) / 1000000ull;
    if (elapsed_ms < g_min_ms) return;

    uint64_t since_last = g_last_fire_ns ? (now - g_last_fire_ns) / 1000000ull : g_tick_ms;
    if (since_last < g_tick_ms) return;

    fire(now, false);
}

void ray_progress_end(void) {
    if (!g_cb) {
        g_start_ns = 0;
        return;
    }
    if (g_showing) {
        /* Final 100% tick — only if the bar was actually shown, so
         * short queries don't flash anything at all. */
        uint64_t now = mono_ns();
        if (g_rows_total) g_rows_done = g_rows_total;
        fire(now, true);
    }
    g_op_name = NULL;
    g_phase = NULL;
    g_rows_done = 0;
    g_rows_total = 0;
    g_start_ns = 0;
    g_last_fire_ns = 0;
    g_showing = false;
}
