# LadderSQL

**Effective Supervision for Agentic Text-to-SQL via Hierarchical Reward and Sample Selection**

Official implementation of the LadderSQL paper (under submission to PVLDB).

LadderSQL improves how much supervision an RL-trained Text-to-SQL agent extracts
from a small annotated corpus. It combines three ideas:

1. **Disagreement-based sample selection** — probe every training example with
   `G=16` rollouts *before* training and keep only those whose outcomes disagree
   (some correct, some not). Examples that are already mastered or entirely out
   of reach give group-relative optimisation no usable contrast, so they are
   dropped. This removes 43–63% of the raw corpus and roughly two thirds of the
   training compute.
2. **Structure-adaptive hierarchical reward** — instead of a single execution
   bit, score a prediction by *how far it got*. The gold query is decomposed
   along the logical clause order (FROM/JOIN → WHERE → GROUP BY → HAVING →
   SELECT → ORDER BY); each active stage is verified against the gold clause,
   verification stops at the first failure, and the longest verified prefix
   earns a rung on a ladder instantiated from that gold query's own structure
   (`rho_max=0.70`, `delta=0.05`, `rho_min=0.10`; an exact execution match still
   earns 1.0). WHERE/HAVING are compared **by execution**, not by text, so an
   equivalent predicate written differently is not penalised.
3. **Multi-representative tournament** — at inference, sample `n` candidates,
   partition them into execution-equivalence classes, and adjudicate the top-3
   classes with an LLM judge. Each class is represented by up to 3 of its most
   frequent distinct writings to remove sensitivity to any single surface form.
   The final choice fuses round-robin win rate with the class's vote share, so
   answers supported by more samples are proportionally harder to overturn.

Training runs as a multi-turn ReAct loop: the model executes its intermediate
queries against the database, observes the results (or errors), and revises.

## Results

Execution accuracy (EX, %), best-of-32 selection unless noted.

| Model | Backbone | Train examples | Bird-dev | Spider-dev | Spider-test |
|---|---|---|---|---|---|
| LadderSQL-7B  | Qwen2.5-Coder-7B-Instruct  | ~7k | 72.5 | 88.9 | 89.0 |
| LadderSQL-14B | Qwen2.5-Coder-14B-Instruct | ~5k | **73.7** | **89.1** | **90.2** |

Greedy decoding with the 14B model reaches 68.8 on Bird-dev; majority voting
over 32 candidates reaches 72.4.

## Repository layout

Each top-level directory maps to one stage of the paper.

```
schema_construction/     Sec. 3.2  question-aware schema construction
  apex_sql_runner/         driver scripts for the Bird schema-linking stage
  value_retrieval/         ChromaDB value index + keyword-based cell retrieval
  cache_build/             merge linked schema + descriptions + values, render prompt text
  direct_linking/          single-call LLM schema linking (Spider)
  eval/                    schema-linking quality metrics (SRR / recall / precision / F1)
sample_selection/        Sec. 3.3  disagreement-based sample selection
training/                Sec. 3.4  multi-turn RL training
  original_agent/           ReAct agent, prompts, DB tools, trainer entrypoint
  reward_func/              hierarchical reward + execution-match baselines
  scripts/                  training launchers (+ scripts/ablation/)
inference_selection/     Sec. 3.5  execution clustering + multi-representative tournament
third_party/             vendored dependencies (see NOTICE)
tools/                   figure generation
docs/                    per-stage reproduction notes
```

Key files:

| Paper | File |
|---|---|
| Eq. (3) selection criterion | `sample_selection/filter_variance_zero.py` |
| Alg. 1 hierarchical reward (Bird) | `training/reward_func/composite_reward_dynamic_bird.py` |
| Alg. 1 hierarchical reward (Spider) | `training/reward_func/composite_reward_dynamic.py` |
| Binary execution reward (ablation) | `training/reward_func/reward.py` |
| Multi-turn ReAct agent | `training/original_agent/ReAct_agent_redundant_schema.py` |
| GRPO trainer entrypoint | `training/original_agent/train_ReAct_agent.py` |
| Eq. (9) execution-equivalence clustering | `inference_selection/gated_common.py` |
| Eq. (10) pairwise adjudication | `inference_selection/run_multirep_tournament.py` |
| Eq. (11) selection with consistency prior | `inference_selection/eval_multirep.py` |
| Judge prompt + code-computed result facts | `inference_selection/run_gated_reasoning_judge.py` |

## Setup

Hardware used in the paper: one node with 8x NVIDIA H20 (80GB), a 128-core Intel
Xeon CPU, Ubuntu 22.04.5 LTS.

```bash
# Download and extract the ZIP from the anonymous artifact link supplied
# with the submission, then enter the extracted project directory.
cd /path/to/extracted-project

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# agentlightning is vendored (not on PyPI at this version) -- make it importable
export PYTHONPATH="$PWD/third_party:$PYTHONPATH"

cp .env.example .env    # then fill in your API keys and `source` / export them
```

> **Paths.** The launcher scripts were written for our cluster. Every path that
> you must supply appears as a `/path/to/...` placeholder (datasets, model
> weights, checkpoint output). Paths under `/root/...` are local scratch
> directories that the scripts create themselves; change them if that location
> is not writable for you. Grep for `/path/to/` before your first run.

### Data

Download the benchmarks yourself and point the placeholders at them:

- **Bird** — [bird-bench.github.io](https://bird-bench.github.io/). Training uses
  `BIRD23-train-filtered` (6,601 of 9,428 pairs); evaluation uses the original
  Bird-dev release (1,534 pairs). Keep the per-database
  `database_description/*.csv` files: the schema stage reads column descriptions
  from them.
- **Spider** — [yale-lily.github.io/spider](https://yale-lily.github.io/spider),
  plus Spider-Syn / Spider-Realistic / Spider-DK for the robustness evaluation.

SQLite files must live on a **local** disk. Running them from a network or
object-storage mount produces intermittent `disk I/O error`s that silently
corrupt execution results and therefore the clustering in Sec. 3.5.

## Stage 1 — Question-aware schema construction

For Bird we reproduce APEX-SQL; its source is obtained from its own authors and
is **not** redistributed here (see NOTICE). `schema_construction/apex_sql_runner/`
holds the driver we used: `chat.py` (OpenAI-compatible LLM client with GLM
thinking disabled), `convert_parquet.py` (parquet -> JSON), and `run_apex_sl.py`
(32-way concurrent runner with resume and rate-limit retry). Details in
[docs/01_schema_construction.md](docs/01_schema_construction.md).

For Spider, one LLM call selects tables and columns from the full schema:

```bash
cd schema_construction/direct_linking
python schema_linking_v2.py            # -> relevant_schema_cache/*.json
```

Then index database cell values and attach the ones matching question literals:

```bash
cd schema_construction/value_retrieval
python build_chroma_train.py           # index textual cells (all-MiniLM-L6-v2)
python extract_keywords_train.py       # extract literals from each question
python train_value_retrieval.py        # top-5 matching cells per keyword
```

Finally merge linked schema + LLM-generated descriptions + retrieved values and
render the text that goes into the prompt:

```bash
cd schema_construction/cache_build
python build_column_profiles.py
python build_schema_cache_train.py     # structured candidate schema per question
python build_rendered_cache_train.py   # -> {"db_id|||question": "<schema text>"}
```

The rendered cache is what training and inference consume (`--schema-truncate
cache --schema-cache-path <file>`). The key format must be exactly
`f"{db_id}|||{question}"`; a mismatch silently falls back to the raw DDL.

Schema quality (paper Table 6):

```bash
cd schema_construction/eval && python eval_srr_v14.py
```

## Stage 2 — Disagreement-based sample selection

Serve the untrained backbone, draw `G=16` rollouts per training example under
exactly the training configuration, then keep only the examples with mixed
outcomes.

```bash
cd sample_selection

# 1. serve the backbone on 8 GPUs
bash scripts/start_vllm_redundant.sh

# 2. 16 rollouts per example, recording the final SQL and its 0/1 outcome
python sample_bird_redundant_with_sql.py \
    --parquet /path/to/nl2sql_dataset/bird/bird_clean_data/train.parquet \
    --db-dir  /path/to/bird/train_databases \
    --output  outputs/bird_train_sampling.json \
    --n-trials 16 --num-gpus 8

# 3. Eq. (3): drop all-correct and all-wrong groups
python filter_variance_zero.py \
    --input  outputs/bird_train_sampling.json \
    --output outputs/bird_train_filtered.json

# 4. to parquet for the trainer
python json_to_parquet.py \
    --input  outputs/bird_train_filtered.json \
    --output outputs/train_bird_14b_redundant_filtered.parquet
```

`scripts/run_bird_redundant_sampling.sh` wraps steps 1-2 (model/database
staging, vLLM startup, sampling) and is resumable — rerun it to continue.
Spider uses `sample_spider_redundant_with_sql.py`.

Retained set sizes (paper Table 7): Bird-train 3,756 (7B) / 2,694 (14B);
Spider-train 3,615 (7B) / 2,604 (14B). Selection is re-run per backbone: an
example that is informative for the 7B model can be trivial for the 14B one.

## Stage 3 — Agentic RL training with the hierarchical reward

```bash
export PYTHONPATH="$PWD/third_party:$PYTHONPATH"
bash training/scripts/train_bird_14b_redundant_dlc.sh
```

The launcher stages data locally and then calls the trainer directly, which is
also the way to run it by hand:

```bash
cd training
python -m original_agent.train_ReAct_agent \
    --agent-variant redundant \
    --schema-truncate cache --schema-cache-path <rendered_schema_cache.json> \
    --dataset bird --max-turns 8 \
    --local-model-path /path/to/models/Qwen2.5-Coder-14B-Instruct \
    --train-data <filtered_train.parquet> --val-data <dev_500.parquet> \
    --epochs 2 --train-batch-size 16 --lr 1e-6 \
    --sample 16 --temperature 1.1 --entropy-coeff 0.0015 \
    --n-gpus 8 --max-prompt-length 21504 --max-response-length 3072 \
    --max-model-len 24576 --default-local-dir <checkpoint_dir>
```

Settings used in the paper: GRPO, 16 rollouts per question, 2 epochs, batch size
16, lr `1e-6`, sampling temperature 1.1, entropy coefficient `1.5e-3`, at most 8
turns on Bird and 10 on Spider. Observation tokens produced by the environment
are masked out of the loss. The reward is computed on the final SQL of each
trajectory.

Reward knobs (environment variables, read by
`training/reward_func/composite_reward_dynamic_bird.py`):

| Variable | Default | Meaning |
|---|---|---|
| `REWARD_MAX_STAGE` | `0.70` | reward of the last active stage (`rho_max`) |
| `REWARD_DECREMENT` | `0.05` | decay per stage going backwards (`delta`) |
| `REWARD_AGG_MODE` | `gating` | `gating` = longest verified prefix (paper); `sum` = accumulate passed stages |

Set-operation queries (UNION/INTERSECT/EXCEPT) bypass the ladder and are scored
by exact execution match alone.

### Ablations (paper Table 5)

| Variant | How |
|---|---|
| w/o sample selection | `training/scripts/ablation/train_bird_14b_redundant_fulldata_dlc.sh` (train on the unfiltered corpus) |
| w/o hierarchical reward | `training/scripts/ablation/train_bird_14b_redundant_binaryreward_dlc.sh` (binary execution reward) |
| w/o multi-turn interaction | `training/scripts/ablation/train_bird_14b_redundant_singleturn_dlc.sh` (single-turn train *and* eval) |
| w/ single representative | Stage 4 with `MULTIREP_RMAX=1` |
| w/o tournament | majority voting: `inference_selection/report_baseline.py` |

## Stage 4 — Multi-representative tournament selection

Internals, input contract and full environment-variable list:
[docs/04_inference_selection.md](docs/04_inference_selection.md).

Input is a JSON list with one record per question, each carrying `db_id`,
`question`, `evidence`, the gold SQL, and `trials[]` with `pred_sql` plus its
0/1 outcome. Gold labels are used **only** for the final metric — clustering and
adjudication never see them.

```bash
cd inference_selection
export BON_DATASET=bird
export BON_DB_DIR=/path/to/local/bird_dev_databases      # local disk, not a network mount

# 1. execution signature of every candidate (this drives the clustering)
BIRD_FILE=<candidates.json> EXEC_CACHE=cache/exec_<tag>.json \
    python build_exec_cache_generic.py --workers 32

# 2. baselines: majority voting and the pass@n ceiling
BIRD_FILE=<candidates.json> EXEC_CACHE=cache/exec_<tag>.json \
    python report_baseline.py

# 3. pairwise adjudication over the top-3 classes, 3 representatives each
BIRD_FILE=<candidates.json> EXEC_CACHE=cache/exec_<tag>.json \
MULTIREP_CACHE=cache/pairs_<tag>.json MULTIREP_RMAX=3 \
SCHEMA_CACHE=<rendered_schema_cache.json> \
    python run_multirep_tournament.py

# 4. Eq. (11): fuse win rate with vote share and report EX
BIRD_FILE=<candidates.json> EXEC_CACHE=cache/exec_<tag>.json \
MULTIREP_CACHE=cache/pairs_<tag>.json MULTIREP_RMAX=3 \
    python eval_multirep.py
```

Step 3 issues the LLM calls and is the slow part; the verdict cache is flushed
periodically and the script resumes, so it is safe to interrupt. Step 4 is
instant once the cache exists and sweeps the aggregation variant, the number of
adjudicated classes `K`, and the consistency-prior weight, printing the best
configuration.

Judge configuration in the paper: GLM-5.2 via an OpenAI-compatible endpoint,
temperature 0, thinking disabled, 32-way concurrency. 72% of Bird-dev questions
need no adjudication at all because all their candidates fall into a single
execution-equivalence class; adjudication averages 6.5 pairwise calls per
question.

`SCHEMA_CACHE` is optional but improves the judge: without it the prompt falls
back to the raw DDL. The key format is again `db_id|||question`.

## Notes and limitations

- The launcher scripts encode our cluster's staging steps (copying models and
  databases to local disk, unpacking a prebuilt virtualenv). Read them before
  running; the trainer invocation at the bottom of each is the part that matters.
- Many inline comments are in Chinese. Docstrings of the entry points and all
  documentation here are in English.
- Checkpoints are not released in this repository.

## Anonymous review artifact

This snapshot is provided for double-anonymous peer review. Author information
and the identifying source-repository link are omitted during review.

## License

MIT — see [LICENSE](LICENSE). Third-party components and their licenses are
listed in [NOTICE](NOTICE).
