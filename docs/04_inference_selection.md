# Stage 4 — Multi-representative tournament selection

Sampling `n` candidates is far more likely to produce a correct query than
greedy decoding, but only if the right one can be picked out. This stage does
that in three steps: collapse candidates that behave identically, adjudicate the
surviving alternatives with an LLM judge shown several phrasings of each, and
fuse the judge's verdicts with how many samples support each answer.

Gold labels are used **only** to compute the final metric. Clustering and
adjudication never see them.

## Input contract

A JSON list, one record per question:

```jsonc
{
  "db_id": "california_schools",
  "question": "...",
  "evidence": "...",              // Bird external knowledge; may be empty
  "gold_sql": "SELECT ...",       // or "query"
  "reward_list": [1, 0, 1, ...],  // per-candidate 0/1 vs. gold; else trials[].reward
  "trials": [
     {"pred_sql": "SELECT ...", "reward": 1.0},
     ...
  ]
}
```

Loading lives in `bird_common.load_samples`. Candidate SQL comes from
`trials[i].pred_sql`; correctness from `reward_list[i]` when present, otherwise
`trials[i].reward`.

Also required:

- **Databases** at `<BON_DB_DIR>/<db_id>/<db_id>.sqlite`, on a local disk.
  Running sqlite from a network or object-storage mount yields intermittent
  `disk I/O error`s, which corrupt execution signatures and therefore the
  clustering — silently.
- **Linked schema cache** (optional but recommended) mapping
  `db_id|||question -> schema text`, passed as `SCHEMA_CACHE`. On a key miss the
  judge prompt falls back to the raw DDL without an error, and adjudication gets
  weaker.
- **A judge LLM** on any OpenAI-compatible endpoint. The paper uses GLM-5.2 with
  `temperature=0` and thinking disabled.

## Environment variables

| Variable | Meaning |
|---|---|
| `BON_DATASET` | `bird` or `spider` (selects the execution-match function) |
| `BON_DB_DIR` | database root |
| `BIRD_FILE` | candidate file |
| `EXEC_CACHE` | execution-result cache path |
| `MULTIREP_CACHE` | pairwise-verdict cache path |
| `MULTIREP_RMAX` | representatives per class (3 in the paper; 1 for the single-representative ablation) |
| `SCHEMA_CACHE` | linked schema cache for the judge prompt |
| `MULTIREP_MODEL` | judge model id |
| `JUDGE_API_KEY`, `JUDGE_API_BASE` | judge endpoint |

## Pipeline

```bash
cd inference_selection
export BON_DATASET=bird BON_DB_DIR=/path/to/local/bird_dev_databases
SRC=<candidates.json>; TAG=<short_tag>
EC=cache/exec_${TAG}.json; MC=cache/pairs_${TAG}.json
SCHEMA=<rendered_schema_cache.json>

BIRD_FILE=$SRC EXEC_CACHE=$EC python build_exec_cache_generic.py --workers 32
BIRD_FILE=$SRC EXEC_CACHE=$EC python report_baseline.py
BIRD_FILE=$SRC EXEC_CACHE=$EC MULTIREP_CACHE=$MC MULTIREP_RMAX=3 SCHEMA_CACHE=$SCHEMA \
    python run_multirep_tournament.py
BIRD_FILE=$SRC EXEC_CACHE=$EC MULTIREP_CACHE=$MC MULTIREP_RMAX=3 python eval_multirep.py
```

`scripts/run_bon32_pipeline.sh` and `scripts/run_spider_dev_all.sh` wrap this.

### 1. Execution cache

`build_exec_cache_generic.py` executes every candidate once and stores
`{ok, error, row_count, result_sig, preview, profile}` keyed by
`db|||normalised_sql`. The signature is `md5` over the sorted stringified rows,
so row order is ignored (matching the Bird evaluation protocol); every failed
execution collapses to the single signature `ERR`.

Note the deliberate asymmetry with training: the ORDER BY stage of the
hierarchical reward *is* order-sensitive, so the training signal is strictly
stricter than the metric.

### 2. Clustering and accounting

`gated_common.build_clusters` groups the candidates by signature. Each class
records `{sig, n, rep_sql, members, preview}` and classes are sorted by size, so
the largest class is exactly the majority-voting prediction. `classify` then
splits the questions:

- single class correct, or several classes all correct -> settled, no judge call;
- at least two classes with both correct and incorrect ones -> **contested**,
  the only set that needs adjudication;
- single class wrong, or all classes wrong -> unreachable.

`report_baseline.py` prints this breakdown together with the majority-voting EX
and the pass@n ceiling. On Bird-dev, 72% of questions fall into a single class
and need no adjudication at all.

### 3. Multi-representative tournament

`run_multirep_tournament.py` takes, for each contested question, the top
`K=3` classes, and from each class up to `RMAX=3` of its most frequent distinct
writings (whitespace-normalised, ordered by frequency then length). Classes are
paired, and every cross pair of representatives is judged in both orders,
`RMAX x RMAX x 2` calls per class pair, with all verdicts cached.

Multiple representatives are the main noise-reduction mechanism: averaging over
three phrasings removes the case where the judge confidently misreads one
awkward formulation. This lowers variance rather than tuning a number.

The judge prompt (`run_gated_reasoning_judge.py`) is built so the model does not
have to count anything itself — row and column counts, headers, count/order
differences and sample rows from the symmetric difference of the two result sets
are computed in code and handed over as facts. On top of that it carries an
ordered checklist (columns and column order, column count, filter precision,
aggregation/grouping intent, malformed results, join path and evidence formulas)
and an anti-over-interpretation prior counterweighting the judge's systematic
preference for the more complex query. The model writes 2-5 sentences of
reasoning and a final `FINAL: A/B` line.

Runtime is dominated by these calls (order of 10k for Bird-dev at n=32, tens of
minutes at 32-way concurrency). The cache is flushed every 1000 verdicts and the
script resumes, so interrupting it is safe.

### 4. Aggregation and selection

`eval_multirep.py` averages each class pair's verdicts into `p_ij` (A=1, B=0,
tie=0.5), computes each class's round-robin win rate against the other top-`K`
classes, and selects

```
score[i] = win_rate[i] + k * (n_i / n)
```

taking the argmax and breaking ties toward the larger class. The second term is
the consistency prior: to overturn the majority answer the judge's win-rate
margin must exceed `k` times the vote-share margin, so answers backed by more
samples are proportionally harder to overturn. `k = 0` degenerates to a pure
judge and is measurably worse; `k -> inf` recovers majority voting.

The script sweeps the aggregation variant (`biased` uses one presentation order,
`debiased` averages both), `K`, and `k`, and prints the best configuration. Run
it a second time with `MULTIREP_RMAX=1` for the single-representative ablation;
the verdict cache is reused, so it is instant.

## Dataset differences

`BON_DATASET` selects how a SQL is scored against gold:

- `bird` -> `bird_reward_label.bird_reward`: set comparison, column order strict,
  row order ignored;
- `spider` -> `reward_func.reward.binary_reward`: `eval_exec_match` with column
  permutation enumeration, i.e. insensitive to column order.

Near-saturated datasets behave differently from Bird: on Spider-dev, majority
voting alone is already around 89%, leaving the tournament much less headroom.
