# Datalog Aggregates + ray-exomem Consumer Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add aggregate body literals (`count`, `sum`, `min`, `max`, `avg`), float constants in expressions, and a `between` sugar to the rayforce2 Datalog engine; then delete the procedural Rust derivations in ray-exomem `system_schema.rs::native_derived_relations` and replace them with declarative rules expressed entirely in the engine.

**Architecture:**
- Phase A (rayforce2, branch `feature/datalog-aggregates`): extend `dl_body_t` with a new literal type `DL_AGG`; lower `(count ?N pred)` etc. to a post-fixpoint aggregation pass within the containing rule's stratum. Introduce a tagged constant value (`DL_CONST_I64` / `DL_CONST_F64`) so `dl_expr_const` accepts both. Add `(between ?x lo hi)` as parser sugar lowering to two `DL_CMP` literals.
- Phase B (ray-exomem, separate branch `feature/declarative-derivations`): consume the new operators. Express health-band derivations as Datalog rules attached during exom seeding, and remove the procedural `native_derived_relations` / `known_derived_samples` paths. The procedural lookup tables in `web.rs` for `health/recommended-water-ml` etc. are deleted.

**Tech Stack:** rayforce2 C engine (datalog.h/datalog.c), munit, ray-exomem Rust (`brain.rs`, `system_schema.rs`, `server.rs`, `auth/routes.rs`).

**Branch state at plan write time:** `feature/datalog-aggregates` HEAD = commit `3ebc6ba` (`test(datalog): cmp + arith body literals work as documented`) which proves cmp + arith already work in upstream master and is the baseline this plan extends.

---

## Phase A — rayforce2: Datalog++ extensions

### Task A1: Define aggregate body-literal type + constants

**Files:**
- Modify: `src/ops/datalog.h` (add `DL_AGG`, `dl_agg_op_t`, fields on `dl_body_t`)

- [ ] **Step 1: Add `DL_AGG` to body literal type constants**

In `src/ops/datalog.h`, after the existing `DL_INTERVAL 5` line, add:

```c
#define DL_AGG      6   /* aggregate: (count ?N pred), (sum ?S ?expr pred), ... */
```

- [ ] **Step 2: Add aggregate operator enum**

In `src/ops/datalog.h`, after the comparison-operator block, add:

```c
/* ===== Aggregate operators (for DL_AGG) ===== */
#define DL_AGG_COUNT 0
#define DL_AGG_SUM   1
#define DL_AGG_MIN   2
#define DL_AGG_MAX   3
#define DL_AGG_AVG   4
```

- [ ] **Step 3: Extend `dl_body_t` with aggregate fields**

In `src/ops/datalog.h`, inside the `dl_body_t` struct (after the existing `interval_*` fields), add:

```c
    int     agg_op;                /* aggregate operator (for DL_AGG) */
    int     agg_target_var;        /* variable that receives the aggregate result */
    char    agg_pred[64];          /* predicate name being aggregated over */
    int     agg_arity;             /* arity of agg_pred */
    int     agg_value_col;         /* column index inside agg_pred to aggregate (sum/min/max/avg) */
```

- [ ] **Step 4: Add public builder API**

In `src/ops/datalog.h`, after `dl_rule_add_interval(...)`, add:

```c
/* Add an aggregate body literal: (op ?target pred col)
 *  - op: DL_AGG_COUNT (col is ignored), DL_AGG_SUM/MIN/MAX/AVG
 *  - target_var: variable that receives the aggregate result
 *  - pred: predicate to aggregate over
 *  - pred_arity: arity of that predicate
 *  - value_col: which column to aggregate (ignored for COUNT)
 * Returns body literal index. */
int dl_rule_add_agg(dl_rule_t* rule, int op, int target_var,
                    const char* pred, int pred_arity, int value_col);
```

- [ ] **Step 5: Compile to verify header parses**

Run: `make lib`
Expected: clean build, no `-Werror` failures.

- [ ] **Step 6: Commit**

```bash
git add src/ops/datalog.h
git commit -m "feat(datalog): declare DL_AGG body literal + aggregate operators"
```

### Task A2: Implement `dl_rule_add_agg` builder

**Files:**
- Modify: `src/ops/datalog.c` (add builder near existing `dl_rule_add_interval`)
- Test: `test/test_datalog.c`

- [ ] **Step 1: Write the failing test**

Append to `test/test_datalog.c` (before the `datalog_tests[]` array):

```c
/* Verify dl_rule_add_agg populates body fields correctly. */
static MunitResult test_agg_builder(const void* params, void* fixture) {
    (void)params; (void)fixture;
    dl_rule_t rule;
    dl_rule_init(&rule, "stats", 1);
    dl_rule_head_var(&rule, 0, 0);

    int idx = dl_rule_add_agg(&rule, DL_AGG_COUNT, 0, "weight", 1, 0);
    munit_assert_int(idx, ==, 0);
    munit_assert_int(rule.body[0].type,           ==, DL_AGG);
    munit_assert_int(rule.body[0].agg_op,         ==, DL_AGG_COUNT);
    munit_assert_int(rule.body[0].agg_target_var, ==, 0);
    munit_assert_string_equal(rule.body[0].agg_pred, "weight");
    munit_assert_int(rule.body[0].agg_arity,      ==, 1);
    munit_assert_int(rule.body[0].agg_value_col,  ==, 0);
    return MUNIT_OK;
}
```

Add to the `datalog_tests[]` array:

```c
    { "/agg_builder", test_agg_builder, datalog_setup, datalog_teardown, 0, NULL },
```

- [ ] **Step 2: Run test, expect failure**

Run: `make test 2>&1 | grep -E "agg_builder|error"`
Expected: link error `undefined symbol dl_rule_add_agg`.

- [ ] **Step 3: Implement the builder**

In `src/ops/datalog.c`, after `dl_rule_add_interval` (around line 363), add:

```c
int dl_rule_add_agg(dl_rule_t* rule, int op, int target_var,
                    const char* pred, int pred_arity, int value_col) {
    if (rule->n_body >= DL_MAX_BODY) return -1;
    int idx = rule->n_body++;
    dl_body_t* b = &rule->body[idx];
    memset(b, 0, sizeof(*b));
    b->type           = DL_AGG;
    b->agg_op         = op;
    b->agg_target_var = target_var;
    snprintf(b->agg_pred, sizeof(b->agg_pred), "%s", pred);
    b->agg_arity      = pred_arity;
    b->agg_value_col  = value_col;
    return idx;
}
```

- [ ] **Step 4: Run test, expect pass**

Run: `make test 2>&1 | grep -E "agg_builder"`
Expected: `/datalog/agg_builder [ OK ]`.

- [ ] **Step 5: Commit**

```bash
git add src/ops/datalog.c test/test_datalog.c
git commit -m "feat(datalog): dl_rule_add_agg builder for aggregate literals"
```

### Task A3: Stratification — treat aggregates as non-monotonic

Aggregates are non-monotonic (adding a fact can change `min`); they must live in a higher stratum than the predicate they aggregate. The existing `dl_stratify` topological sort treats negation as the only non-monotonic edge — extend it to treat aggregate dependencies the same way.

**Files:**
- Modify: `src/ops/datalog.c` (function building the negation edge set, currently driven by `DL_NEG`)

- [ ] **Step 1: Locate stratification edge construction**

Run: `grep -n "DL_NEG" src/ops/datalog.c | head`
Identify the loop that pushes negative dependency edges (typically inside `dl_stratify` or a helper named `dl_build_dep_graph`).

- [ ] **Step 2: Write the failing test**

Append to `test/test_datalog.c`:

```c
/* Aggregates over an IDB must be evaluated in a strictly higher stratum
 * than the IDB itself. Program:
 *   EDB: edge(1,2), edge(2,3)
 *   Rule R0: path(X,Y) :- edge(X,Y)
 *   Rule R1: path_count(N) :- (count ?N path)
 * After stratification: R1.stratum > R0.stratum. */
static MunitResult test_agg_stratifies_above_source(const void* params, void* fixture) {
    (void)params; (void)fixture;
    int64_t s_vals[] = {1, 2};
    int64_t d_vals[] = {2, 3};
    ray_t* sc = ray_vec_from_raw(RAY_I64, s_vals, 2);
    ray_t* dc = ray_vec_from_raw(RAY_I64, d_vals, 2);
    ray_t* edge = ray_table_new(2);
    edge = ray_table_add_col(edge, ray_sym_intern("edge__c0", 8), sc);
    edge = ray_table_add_col(edge, ray_sym_intern("edge__c1", 8), dc);

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "edge", edge, 2);

    dl_rule_t r0; dl_rule_init(&r0, "path", 2);
    dl_rule_head_var(&r0, 0, 0); dl_rule_head_var(&r0, 1, 1);
    int b = dl_rule_add_atom(&r0, "edge", 2);
    dl_body_set_var(&r0, b, 0, 0); dl_body_set_var(&r0, b, 1, 1);
    r0.n_vars = 2;
    dl_add_rule(prog, &r0);

    dl_rule_t r1; dl_rule_init(&r1, "path_count", 1);
    dl_rule_head_var(&r1, 0, 0);
    dl_rule_add_agg(&r1, DL_AGG_COUNT, 0, "path", 2, 0);
    r1.n_vars = 1;
    dl_add_rule(prog, &r1);

    munit_assert_int(dl_stratify(prog), ==, 0);
    munit_assert_int(prog->rules[1].stratum, >, prog->rules[0].stratum);

    dl_program_free(prog);
    ray_release(edge); ray_release(sc); ray_release(dc);
    return MUNIT_OK;
}
```

Add `{ "/agg_stratifies_above_source", test_agg_stratifies_above_source, ... }` to `datalog_tests[]`.

- [ ] **Step 3: Run test, expect failure**

Run: `make test 2>&1 | grep agg_stratifies`
Expected: assertion failure `r1.stratum > r0.stratum` because the aggregate edge is currently invisible to the stratifier (or the program is mis-stratified).

- [ ] **Step 4: Add aggregate edges to dependency graph**

In the dependency-graph builder identified in Step 1, add a branch for `DL_AGG` mirroring the `DL_NEG` branch:

```c
} else if (b->type == DL_AGG) {
    int src_idx = dl_find_rel(prog, b->agg_pred);
    if (src_idx >= 0) {
        /* aggregate creates non-monotonic dependency: head depends negatively on source */
        add_dep_edge(graph, head_rel_idx, src_idx, /*negative=*/true);
    }
}
```

(Use the helper names actually present in `datalog.c` — adapt the snippet to match.)

- [ ] **Step 5: Run test, expect pass**

Run: `make test 2>&1 | grep agg_stratifies`
Expected: `[ OK ]`.

- [ ] **Step 6: Commit**

```bash
git add src/ops/datalog.c test/test_datalog.c
git commit -m "feat(datalog): aggregates participate in stratification (non-monotonic)"
```

### Task A4: Evaluate aggregates inside `dl_compile_rule`

Aggregates fire after the source predicate's stratum reaches fixpoint. The simplest implementation: when compiling a rule whose body contains a `DL_AGG` literal, generate a graph node that scans the source IDB/EDB and reduces it.

**Files:**
- Modify: `src/ops/datalog.c` — extend `dl_compile_rule`

- [ ] **Step 1: Write failing test (count over EDB)**

Append:

```c
/* (count ?N weight) where weight has 4 rows -> N = 4. */
static MunitResult test_agg_count_edb(const void* params, void* fixture) {
    (void)params; (void)fixture;
    int64_t vals[] = {50, 60, 75, 85};
    ray_t* col = ray_vec_from_raw(RAY_I64, vals, 4);
    ray_t* weight = ray_table_new(1);
    weight = ray_table_add_col(weight, ray_sym_intern("weight__c0", 10), col);

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "weight", weight, 1);

    dl_rule_t r; dl_rule_init(&r, "wcount", 1);
    dl_rule_head_var(&r, 0, 0);
    dl_rule_add_agg(&r, DL_AGG_COUNT, 0, "weight", 1, 0);
    r.n_vars = 1;
    dl_add_rule(prog, &r);

    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "wcount");
    munit_assert_ptr_not_null(out);
    munit_assert_int((int)ray_table_nrows(out), ==, 1);
    int64_t* od = (int64_t*)ray_data(ray_table_get_col_idx(out, 0));
    munit_assert_int((int)od[0], ==, 4);

    dl_program_free(prog);
    ray_release(weight); ray_release(col);
    return MUNIT_OK;
}
```

Add to `datalog_tests[]`.

- [ ] **Step 2: Run test, expect failure**

Run: `make test 2>&1 | grep agg_count_edb`
Expected: assertion failure (`wcount` table has 0 rows because compiler ignores `DL_AGG`).

- [ ] **Step 3: Extend `dl_compile_rule`**

Locate `dl_compile_rule` in `src/ops/datalog.c` (function declared at `datalog.h:278`). After the loop that emits nodes for atoms / cmp / assign, add a branch:

```c
case DL_AGG: {
    int src_idx = dl_find_rel(prog, b->agg_pred);
    if (src_idx < 0) return NULL;  /* unknown predicate */
    ray_t* src_table = prog->rels[src_idx].table;
    int64_t nrows = src_table ? ray_table_nrows(src_table) : 0;

    int64_t result;
    switch (b->agg_op) {
        case DL_AGG_COUNT: result = nrows; break;
        case DL_AGG_SUM:
        case DL_AGG_MIN:
        case DL_AGG_MAX:
        case DL_AGG_AVG: {
            ray_t* vc = ray_table_get_col_idx(src_table, b->agg_value_col);
            int64_t* vd = vc ? (int64_t*)ray_data(vc) : NULL;
            if (!vd || nrows == 0) { result = 0; break; }
            int64_t acc = vd[0];
            for (int64_t i = 1; i < nrows; i++) {
                if (b->agg_op == DL_AGG_SUM) acc += vd[i];
                else if (b->agg_op == DL_AGG_MIN) { if (vd[i] < acc) acc = vd[i]; }
                else if (b->agg_op == DL_AGG_MAX) { if (vd[i] > acc) acc = vd[i]; }
                else /* AVG */ acc += vd[i];
            }
            result = (b->agg_op == DL_AGG_AVG) ? acc / nrows : acc;
            break;
        }
        default: return NULL;
    }
    /* Bind result to the aggregate's target variable in the head's binding row. */
    bind_const_to_var(g, b->agg_target_var, result);
    break;
}
```

(Adapt `bind_const_to_var` to whatever helper the existing `DL_ASSIGN` path uses.)

- [ ] **Step 4: Run test, expect pass**

Run: `make test 2>&1 | grep agg_count_edb`
Expected: `[ OK ]`.

- [ ] **Step 5: Add SUM/MIN/MAX/AVG tests**

Append four parallel tests using the same `weight` EDB (50, 60, 75, 85):
- `test_agg_sum`: expect 270
- `test_agg_min`: expect 50
- `test_agg_max`: expect 85
- `test_agg_avg`: expect 67 (integer div: 270/4)

Each rule looks like:
```c
dl_rule_add_agg(&r, DL_AGG_SUM, 0, "weight", 1, 0);
```

- [ ] **Step 6: Run all aggregate tests**

Run: `make test 2>&1 | grep '/datalog/agg'`
Expected: all `[ OK ]`.

- [ ] **Step 7: Commit**

```bash
git add src/ops/datalog.c test/test_datalog.c
git commit -m "feat(datalog): evaluate count/sum/min/max/avg aggregates"
```

### Task A5: Surface-syntax parser for aggregates

Allow Rayfall users to write `(count ?N weight)` and `(sum ?S weight 0)` in rule bodies, parsed by `dl_parse_body_clause` (`src/ops/datalog.c:2336`).

**Files:**
- Modify: `src/ops/datalog.c` (add `dl_is_aggregate`, branch inside `dl_parse_body_clause`)

- [ ] **Step 1: Write failing test through surface parser**

Append a test that goes through `ray_rule_fn` (the special form). Use existing test_lang harnesses as a template — e.g. find how `test_lang_rf.inc` calls `eval_rf_string("(rule ...)")`.

Skeleton:

```c
static MunitResult test_agg_parse_count(const void* params, void* fixture) {
    (void)params; (void)fixture;
    /* (rule (wcount ?N) (count ?N weight)) */
    /* Run via lang eval helper, then dl_query. */
    ...
}
```

If a lang harness is unavailable from `test_datalog.c`, the equivalent integration test must live alongside the rule-evaluation tests in `test/test_lang_rf.inc`.

- [ ] **Step 2: Add `dl_is_aggregate` helper**

In `src/ops/datalog.c` near `dl_is_assignment` (around line 2281):

```c
static bool dl_is_aggregate(ray_t* clause) {
    if (!is_list(clause) || ray_len(clause) < 3) return false;
    ray_t** ce = (ray_t**)ray_data(clause);
    if (ce[0]->type != -RAY_SYM) return false;
    ray_t* name = ray_sym_str(ce[0]->i64);
    if (!name) return false;
    const char* n = ray_str_ptr(name);
    return strcmp(n, "count") == 0 || strcmp(n, "sum") == 0
        || strcmp(n, "min")   == 0 || strcmp(n, "max") == 0
        || strcmp(n, "avg")   == 0;
}

static int dl_agg_op_from_name(const char* n) {
    if (strcmp(n, "count") == 0) return DL_AGG_COUNT;
    if (strcmp(n, "sum")   == 0) return DL_AGG_SUM;
    if (strcmp(n, "min")   == 0) return DL_AGG_MIN;
    if (strcmp(n, "max")   == 0) return DL_AGG_MAX;
    if (strcmp(n, "avg")   == 0) return DL_AGG_AVG;
    return -1;
}
```

- [ ] **Step 3: Branch in `dl_parse_body_clause`**

In `src/ops/datalog.c`, immediately before the assignment branch (line 2400), insert:

```c
/* -- Aggregate: (count ?N pred) | (sum ?S pred col) | ... -- */
if (dl_is_aggregate(clause)) {
    ray_t* op_name = ray_sym_str(ce[0]->i64);
    int op = dl_agg_op_from_name(ray_str_ptr(op_name));

    if (!is_dl_var(ce[1]))
        return ray_error("type", "aggregate target must be a ?variable");
    int target_vi = dl_var_get_or_create(vars, ce[1]->i64);

    if (ce[2]->type != -RAY_SYM)
        return ray_error("type", "aggregate source predicate must be a symbol");
    ray_t* pred_str = ray_sym_str(ce[2]->i64);

    int value_col = 0;
    if (op != DL_AGG_COUNT) {
        if (clen < 4 || ce[3]->type != -RAY_I64)
            return ray_error("type", "sum/min/max/avg requires explicit column index");
        value_col = (int)ce[3]->i64;
    }

    /* Need predicate arity — best effort: use existing relation if registered,
     * otherwise default to 1 (engine will validate at eval time). */
    int rel_idx = dl_find_rel(rule_prog_unused_param, ray_str_ptr(pred_str));
    int arity = (rel_idx >= 0) ? rule_prog_unused_param->rels[rel_idx].arity : 1;

    dl_rule_add_agg(rule, op, target_vi, ray_str_ptr(pred_str), arity, value_col);
    return NULL;
}
```

Note: `dl_parse_body_clause` does not currently take the `dl_program_t*`. Adding it requires updating its signature plus all callers (`dl_parse_inline_rule`, `ray_rule_fn`, etc.). Either thread the program through or store the pred name + look up arity at compile time. Pick one approach in this step, document it in the commit.

- [ ] **Step 4: Run test, expect pass**

Run: `make test 2>&1 | grep -E "agg_parse"`
Expected: `[ OK ]`.

- [ ] **Step 5: Commit**

```bash
git add src/ops/datalog.c test/test_datalog.c test/test_lang_rf.inc
git commit -m "feat(datalog): surface syntax (count|sum|min|max|avg ?v pred [col])"
```

### Task A6: Float constants in `dl_expr_t`

Today `dl_expr_t.const_val` is `int64_t` and `dl_expr_const(int64_t)` is the only way to build a constant. Add a sibling `dl_expr_const_f64(double)` and a tag field so the evaluator and arithmetic builtins know which representation to use.

**Files:**
- Modify: `src/ops/datalog.h` (extend `dl_expr_t`, add `DL_EXPR_CONST_F64` enum value)
- Modify: `src/ops/datalog.c` (extend `dl_build_expr`, add `dl_expr_const_f64`)
- Test: `test/test_datalog.c`

- [ ] **Step 1: Extend the AST**

In `src/ops/datalog.h`, change `dl_expr_kind_t`:

```c
typedef enum {
    DL_EXPR_CONST,        /* integer constant (back-compat) */
    DL_EXPR_CONST_F64,    /* float constant */
    DL_EXPR_VAR,
    DL_EXPR_BINOP,
} dl_expr_kind_t;
```

And the struct:

```c
typedef struct dl_expr {
    dl_expr_kind_t  kind;
    int64_t         const_val;   /* DL_EXPR_CONST */
    double          const_f64;   /* DL_EXPR_CONST_F64 */
    int             var_idx;
    int             binop;
    struct dl_expr *left, *right;
} dl_expr_t;
```

- [ ] **Step 2: Builder + parser hook**

In `src/ops/datalog.c` add:

```c
dl_expr_t* dl_expr_const_f64(double v) {
    dl_expr_t* e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->kind = DL_EXPR_CONST_F64;
    e->const_f64 = v;
    return e;
}
```

In `dl_build_expr` (line 2203), accept `RAY_F64` literals:

```c
if (node->type == -RAY_F64)
    return dl_expr_const_f64(node->f64);
```

- [ ] **Step 3: Evaluator coverage**

Locate the expression evaluator (likely `dl_eval_expr` or inlined inside `dl_compile_rule`). Add a `DL_EXPR_CONST_F64` arm that returns the float as the materialized value. Coerce when mixed-arithmetic is requested (promote i64 → f64 if either side is f64).

- [ ] **Step 4: Test float assignment**

Append:

```c
/* Float arithmetic: (= ?z (+ 1.5 2.5)) -> 4.0 (assuming i64 coercion to f64). */
static MunitResult test_arith_assign_f64(const void* params, void* fixture) {
    (void)params; (void)fixture;
    int64_t one[] = { 1 };
    ray_t* col = ray_vec_from_raw(RAY_I64, one, 1);
    ray_t* trig = ray_table_new(1);
    trig = ray_table_add_col(trig, ray_sym_intern("trig__c0", 8), col);

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "trig", trig, 1);

    dl_rule_t r; dl_rule_init(&r, "fres", 2);
    dl_rule_head_var(&r, 0, 0); dl_rule_head_var(&r, 1, 1);
    int b = dl_rule_add_atom(&r, "trig", 1);
    dl_body_set_var(&r, b, 0, 0);

    dl_expr_t* e = dl_expr_binop(OP_ADD, dl_expr_const_f64(1.5), dl_expr_const_f64(2.5));
    dl_rule_add_assign(&r, 1, DL_OP_EQ, e);
    r.n_vars = 2;
    dl_add_rule(prog, &r);

    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "fres");
    munit_assert_int((int)ray_table_nrows(out), ==, 1);
    /* Assert the result column type or value depending on coercion choice. */
    munit_assert_ptr_not_null(out);

    dl_program_free(prog);
    ray_release(trig); ray_release(col);
    return MUNIT_OK;
}
```

- [ ] **Step 5: Commit**

```bash
git add src/ops/datalog.h src/ops/datalog.c test/test_datalog.c
git commit -m "feat(datalog): float constants in expressions; mixed-mode arithmetic"
```

### Task A7: `(between ?x lo hi)` parser sugar

Pure parser-level rewrite: `(between ?x lo hi)` lowers to `(>= ?x lo)` + `(<= ?x hi)` — two `DL_CMP` body literals. No new evaluator code.

**Files:**
- Modify: `src/ops/datalog.c` (`dl_parse_body_clause` — add branch)
- Test: `test/test_datalog.c`

- [ ] **Step 1: Failing test**

```c
/* (rule (mid ?w) (weight ?w) (between ?w 60 80))
 * weight has 50, 60, 75, 85 -> mid has 60, 75. */
static MunitResult test_between_sugar(const void* params, void* fixture) {
    (void)params; (void)fixture;
    int64_t vals[] = {50, 60, 75, 85};
    ray_t* col = ray_vec_from_raw(RAY_I64, vals, 4);
    ray_t* weight = ray_table_new(1);
    weight = ray_table_add_col(weight, ray_sym_intern("weight__c0", 10), col);

    dl_program_t* prog = dl_program_new();
    dl_add_edb(prog, "weight", weight, 1);

    /* Manually build to mirror what the parser will produce; once the
     * parser is in, swap to a (rule ...) eval call. */
    dl_rule_t r; dl_rule_init(&r, "mid", 1);
    dl_rule_head_var(&r, 0, 0);
    int b = dl_rule_add_atom(&r, "weight", 1); dl_body_set_var(&r, b, 0, 0);
    dl_rule_add_cmp_const(&r, DL_CMP_GE, 0, 60);
    dl_rule_add_cmp_const(&r, DL_CMP_LE, 0, 80);
    r.n_vars = 1;
    dl_add_rule(prog, &r);
    munit_assert_int(dl_eval(prog), ==, 0);
    ray_t* out = dl_query(prog, "mid");
    munit_assert_int((int)ray_table_nrows(out), ==, 2);

    dl_program_free(prog);
    ray_release(weight); ray_release(col);
    return MUNIT_OK;
}
```

This test passes immediately because it sidesteps the parser. Then add a parallel surface-syntax test that requires the parser change.

- [ ] **Step 2: Implement parser sugar**

In `dl_parse_body_clause`, before the assignment branch:

```c
if (clen == 4 && ce[0]->type == -RAY_SYM) {
    ray_t* nm = ray_sym_str(ce[0]->i64);
    if (nm && strcmp(ray_str_ptr(nm), "between") == 0) {
        if (!is_dl_var(ce[1]))
            return ray_error("type", "between target must be a ?variable");
        int vi = dl_var_get_or_create(vars, ce[1]->i64);
        if (ce[2]->type != -RAY_I64 || ce[3]->type != -RAY_I64)
            return ray_error("type", "between bounds must be integer constants");
        dl_rule_add_cmp_const(rule, DL_CMP_GE, vi, ce[2]->i64);
        dl_rule_add_cmp_const(rule, DL_CMP_LE, vi, ce[3]->i64);
        return NULL;
    }
}
```

- [ ] **Step 3: Test surface parsing via lang harness**

Add to `test/test_lang_rf.inc`:

```c
/* Confirm (between ?x lo hi) lowers to two cmp literals at the parser. */
```

(Use the project's existing `eval_rf_string` / rule-eval helper; mirror nearby tests.)

- [ ] **Step 4: Run + commit**

```bash
make test
git add src/ops/datalog.c test/test_datalog.c test/test_lang_rf.inc
git commit -m "feat(datalog): (between ?x lo hi) parser sugar lowers to two cmps"
```

### Task A8: Push Phase A branch

- [ ] **Step 1: Push to origin**

```bash
git push origin feature/datalog-aggregates
```

- [ ] **Step 2: Open draft PR against `RayforceDB/rayforce2:master`**

Use `gh pr create --draft --base master --head theaspirational:feature/datalog-aggregates`. Title: "feat(datalog): aggregates, float constants, and `between` sugar". Body lists Task IDs from this plan.

---

## Phase B — ray-exomem: declarative health derivations

Phase B runs on a separate branch `feature/declarative-derivations` in `/Users/aspirational/Documents/code/lynx/Teide/ray-exomem`. It depends on Phase A's float support only if templates use float thresholds; the integer-only path can land first.

**Note on dependency direction:** `ray-exomem` consumes `rayforce2` via the sibling checkout (`build.rs:97`). To use Phase A features locally, switch the rayforce2 checkout to `feature/datalog-aggregates` before building ray-exomem. To consume from the upstream merge, wait until the PR from Task A8 is merged into `RayforceDB/rayforce2:master` and the sibling is back on `master`.

### Task B1: Capture current native_derived_relations behavior in tests

**Files:**
- Test: `src/system_schema.rs` (extend the existing `tests` module that contains `native_derived_relations` cases — see line 816)

- [ ] **Step 1: Read the existing test**

```bash
sed -n '810,860p' /Users/aspirational/Documents/code/lynx/Teide/ray-exomem/src/system_schema.rs
```

Note the inputs (age=30, height=175, weight=75) and the expected derived relations.

- [ ] **Step 2: Add a regression test that pins the public-facing rule strings**

In the existing test module, add:

```rust
#[test]
fn native_derived_water_band_for_default_profile() {
    let exom = "alice/personal/health/main";
    let mut brain = Brain::new();
    let ctx = MutationContext::default();
    brain.assert_fact(HEALTH_PROFILE_HEIGHT_CM_FACT_ID, PROFILE_HEIGHT_CM, "175",
        1.0, "test", None, None, &ctx).unwrap();
    brain.assert_fact(HEALTH_PROFILE_WEIGHT_KG_FACT_ID, PROFILE_WEIGHT_KG, "75",
        1.0, "test", None, None, &ctx).unwrap();

    let rels = native_derived_relations(exom, &brain);
    let band = rels.iter().find(|r| r.name == HEALTH_WATER_BAND).unwrap();
    assert_eq!(band.sample_tuples, vec![vec!["medium".to_string()]]);
}
```

Repeat for `step_band` with age 30 → "high".

- [ ] **Step 3: Run + commit**

```bash
cargo test -p ray-exomem system_schema::tests::native_derived_water_band_for_default_profile
git add src/system_schema.rs
git commit -m "test(system_schema): pin water/step band derivations before refactor"
```

### Task B2: Express water_band as a Datalog rule

The current Rust:

```rust
let band = if weight_kg < 60 && height_cm < 170 { "small" }
           else if weight_kg >= 85 || height_cm >= 185 { "large" }
           else { "medium" };
```

Three rules in Datalog (using `==`/`<`/`>=` already supported by the engine):

```scheme
(rule {exom}
  (health/water-band "small")
  (?w_id 'health/profile/weight_kg ?w)
  (?h_id 'health/profile/height_cm ?h)
  (< ?w 60)
  (< ?h 170))

(rule {exom}
  (health/water-band "large")
  (?w_id 'health/profile/weight_kg ?w)
  (?h_id 'health/profile/height_cm ?h)
  (>= ?w 85))

(rule {exom}
  (health/water-band "large")
  (?w_id 'health/profile/weight_kg ?w)
  (?h_id 'health/profile/height_cm ?h)
  (>= ?h 185))

(rule {exom}
  (health/water-band "medium")
  (?w_id 'health/profile/weight_kg ?w)
  (?h_id 'health/profile/height_cm ?h)
  (not (health/water-band "small"))
  (not (health/water-band "large")))
```

The fourth rule uses negation; this requires the existing stratifier (already in upstream) to keep `medium` in a higher stratum than `small`/`large`.

**Files:**
- Modify: `src/auth/routes.rs` (`health_bootstrap_rules`, lines 204-225)

- [ ] **Step 1: Replace `health_bootstrap_rules` body with the four rules above**

Show full replacement code (the existing function returns a `Vec<String>`, format the new rules with `format!` so the `{exom}` token is interpolated):

```rust
fn health_bootstrap_rules(exom: &str) -> Vec<String> {
    vec![
        format!(
            r#"(rule {exom} (health/water-band "small") \
                (?w_id 'health/profile/weight_kg ?w) \
                (?h_id 'health/profile/height_cm ?h) \
                (< ?w 60) (< ?h 170))"#
        ),
        format!(
            r#"(rule {exom} (health/water-band "large") \
                (?w_id 'health/profile/weight_kg ?w) \
                (?h_id 'health/profile/height_cm ?h) \
                (>= ?w 85))"#
        ),
        format!(
            r#"(rule {exom} (health/water-band "large") \
                (?w_id 'health/profile/weight_kg ?w) \
                (?h_id 'health/profile/height_cm ?h) \
                (>= ?h 185))"#
        ),
        format!(
            r#"(rule {exom} (health/water-band "medium") \
                (?w_id 'health/profile/weight_kg ?w) \
                (?h_id 'health/profile/height_cm ?h) \
                (not (health/water-band "small")) \
                (not (health/water-band "large")))"#
        ),
        // step bands inserted in Task B3
    ]
}
```

(Remove the embedded escaped newlines once you confirm the rule parser handles single-line strings; the `\` is just for readability here.)

- [ ] **Step 2: Run cargo build to surface format errors**

```bash
cargo build --release --features postgres
```

- [ ] **Step 3: Verify against running daemon**

(per CLAUDE.md: "When adding or modifying any db/rayfall interactions test them against the running ray-exomem daemon.")

```bash
set -a; source .env; set +a
ray-exomem stop
ray-exomem serve --bind 127.0.0.1:9780 \
  --auth-provider google --google-client-id "$GOOGLE_CLIENT_ID" \
  --allowed-domains "$ALLOWED_DOMAINS" --database-url "$DATABASE_URL" &
SERVE_PID=$!
sleep 3
# As an authenticated session, query water-band:
curl -s 'http://127.0.0.1:9780/ray-exomem/api/query' \
  -H 'Cookie: <captured-session>' \
  -d '{"exom":"<your-email>/personal/health/main","rayfall":"(query (?b) (health/water-band ?b))"}'
kill $SERVE_PID
```

Expected: returns `["medium"]` for the default profile (weight=75, height=175).

- [ ] **Step 4: Commit**

```bash
git add src/auth/routes.rs
git commit -m "refactor(onboarding): water_band as declarative Datalog rules"
```

### Task B3: Express step_band as a Datalog rule

Mirror Task B2 for step bands — three rules covering `age < 30 → high`, `age < 50 → medium`, otherwise `gentle`.

- [ ] **Step 1: Append to `health_bootstrap_rules`**

```rust
        format!(r#"(rule {exom} (health/step-band "high")
                    (?id 'health/profile/age ?a) (< ?a 30))"#),
        format!(r#"(rule {exom} (health/step-band "medium")
                    (?id 'health/profile/age ?a) (>= ?a 30) (< ?a 50))"#),
        format!(r#"(rule {exom} (health/step-band "gentle")
                    (?id 'health/profile/age ?a) (>= ?a 50))"#),
```

- [ ] **Step 2: Test against running daemon (same harness as B2)**

Expected: `(query (?b) (health/step-band ?b))` returns `["high"]` for age=30 (matches `< 50` and `>= 30`, but **NOT** `< 30` — boundary check: original Rust uses `if age < 30 → "high"` — so `30 → "medium"`, not `"high"`. Adjust the rules accordingly:

```rust
        format!(r#"(rule {exom} (health/step-band "high")
                    (?id 'health/profile/age ?a) (< ?a 30))"#),
        format!(r#"(rule {exom} (health/step-band "medium")
                    (?id 'health/profile/age ?a) (>= ?a 30) (< ?a 50))"#),
        format!(r#"(rule {exom} (health/step-band "gentle")
                    (?id 'health/profile/age ?a) (>= ?a 50))"#),
```

The existing Rust returns "medium" for age=30 (because `30 < 30` is false → falls through to next branch which is `< 50` → "medium"). Confirm the rules match by re-reading `system_schema.rs:676`:

```rust
let band = if age < 30 { "high" }
           else if age < 50 { "medium" }
           else { "gentle" };
```

So age=30 → "medium". The rule set above matches.

- [ ] **Step 3: Commit**

```bash
git add src/auth/routes.rs
git commit -m "refactor(onboarding): step_band as declarative Datalog rules"
```

### Task B4: Delete `native_derived_relations` and consumers

**Files:**
- Modify: `src/system_schema.rs` (delete `native_derived_relations` function and its call sites)
- Modify: `src/server.rs` (delete `known_derived_samples` block at line 3583-3641)

- [ ] **Step 1: Inventory call sites**

```bash
grep -n "native_derived_relations\|known_derived_samples" src/
```

Expected hits: `system_schema.rs:640,703,741,816`, `server.rs:3585,3598`.

- [ ] **Step 2: Remove call sites in `server.rs`**

In `src/server.rs`, delete lines 3583-3641 (the entire `known_derived_samples` block including the hardcoded "small"/"medium"/"large" → "2000"/"2500"/"3000" lookup). Replace with:

```rust
let known_derived_samples: HashMap<String, Vec<Vec<serde_json::Value>>> = HashMap::new();
```

(or remove the variable entirely if downstream code can be updated to skip the field.)

- [ ] **Step 3: Remove `native_derived_relations` from `system_schema.rs`**

Delete:
- The `native_derived_relations` function (lines 640-698)
- Its inclusion in `builtin_rule_specs` (line 702-706)
- The test `native_derived_water_band_for_default_profile` only if it now relies on deleted code; otherwise rewrite it to query the engine end-to-end.

Also delete now-unused constants if no other consumer remains:
- `HEALTH_WATER_BAND`, `HEALTH_STEP_BAND` (lines 80-81)
- `HEALTH_PROFILE_*_FACT_ID` constants
- `latest_active_fact` helper (if unused after deletion)

- [ ] **Step 4: Build + test**

```bash
cargo build --release --features postgres
cargo test
```

Expected: green; the regression test from B1 now goes through the engine because `health_bootstrap_rules` does the work.

- [ ] **Step 5: Run end-to-end against daemon**

Repeat the query harness from Task B2/B3. Confirm both `health/water-band` and `health/step-band` return correct values without any procedural Rust in the path.

- [ ] **Step 6: Commit**

```bash
git add src/system_schema.rs src/server.rs
git commit -m "refactor(onboarding): delete procedural native_derived_relations"
```

### Task B5: Replace hardcoded recommended-water/steps lookup

The bootstrap rules at `auth/routes.rs:207-223` already express water-ml/steps-per-day as Datalog. After Task B2/B3 the engine itself derives `water-band` and `step-band`, so these existing rules naturally become reachable. Verify and clean up:

- [ ] **Step 1: Confirm bootstrap rules still work**

Run the daemon with a fresh DB; log in; query:

```scheme
(query (?ml) (health/recommended-water-ml ?ml))
(query (?sp) (health/recommended-steps-per-day ?sp))
```

Expected: returns the value matching the derived band. (E.g. default profile → `medium` band → `"2500"` ml.)

- [ ] **Step 2: If broken, debug by querying intermediate predicates**

```scheme
(query (?b) (health/water-band ?b))
(query (?b) (health/step-band ?b))
```

This isolates whether B2/B3 rules fired correctly.

- [ ] **Step 3: Commit any fixes**

```bash
git commit -am "fix(onboarding): wire derived bands into recommended-* lookups"
```

### Task B6: Pin rayforce2 dependency in CLAUDE.md

If Phase A features are needed (aggregates / float / between), the consumer must check out the matching rayforce2 branch.

**Files:**
- Modify: `/Users/aspirational/Documents/code/lynx/Teide/ray-exomem/CLAUDE.md` (Important gotchas section)

- [ ] **Step 1: Add a note**

Insert under "Important gotchas":

```markdown
- The bootstrap health rules use `<` / `>=` / `not` in Datalog rule bodies. These are
  available in upstream rayforce2 master (commit 287aebf+). If aggregate-based rules
  are added (count/sum/min/max/avg) or float thresholds are introduced, the sibling
  rayforce2 checkout must be on `feature/datalog-aggregates` or a master that
  includes the merged PR.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note rayforce2 datalog feature dependency"
```

---

## Out of scope

- Onboarding template system (separate plan): TOML-driven exom seeding, `--onboarding-templates` CLI flag, `/welcome` UI, multi-template catalog. That work consumes the artifacts of this plan but is independently scoped.
- ASP overlay / weak constraints / disjunctive rules: parked.
- Group-by aggregates (e.g. `count` per partition): only ungrouped aggregates in scope; group-by is a follow-up.

## Verification checklist

- [ ] `make test` passes in rayforce2 with all `/datalog/*` tests green
- [ ] `cargo test` passes in ray-exomem with no procedural derivations remaining
- [ ] Live daemon returns correct `water-band` / `step-band` / `recommended-water-ml` / `recommended-steps-per-day` for default profile (weight=75, height=175, age=30) → `medium` / `medium` / `2500` / `9000`
- [ ] Boundary smoke test: weight=85 → `large` band → `3000` ml; age=29 → `high` step band → `10000` steps
- [ ] `grep -r "native_derived_relations" src/` returns no hits in ray-exomem
