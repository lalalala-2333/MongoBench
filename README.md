# MongoBench

A benchmark of whether large language models can predict **which documents a MongoDB
query matches**. Given a collection of documents *D* and a `find()` filter *Q*, a model
must return the exact set of matching `_id` values, scored by exact-set match. Ground
truth is computed by a real `mongod` server, so answers are authoritative and
independent of any model.

The 300 items each compose three or four documented MongoDB matching pitfalls
("footguns"), such as range comparison against `null` (which matches nothing) and
field-order-sensitive embedded-document equality.

## Data

`mongobench.jsonl` — one item per line:

| field | meaning |
|---|---|
| `id` | item identifier |
| `docs` | the document collection (JSON objects with integer `_id`) |
| `query` | the `find()` filter |
| `answer` | gold set of matching `_id` values (from a real `mongod`) |
| `families` | the footgun families the item composes |
| `n_families` | how many families (3 or 4) |
| `proposer` | the model that authored the item |
| `oracle` | the engine used for ground truth |

## Quick start

```bash
pip install pymongo anthropic openai

# 1. start a local MongoDB (ground-truth oracle)
mongod --port 27018 --dbpath /tmp/mbench --bind_ip 127.0.0.1

# 2. verify the released gold reproduces on your server
python oracle_mongod.py --items mongobench.jsonl --uri mongodb://127.0.0.1:27018

# 3. evaluate a model (baseline)
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY
python run_eval.py --model claude-opus-4-8 --provider anthropic \
    --items mongobench.jsonl --out results.jsonl --effort medium
```

## Files

| file | purpose |
|---|---|
| `mongobench.jsonl` | the 300-item benchmark |
| `oracle_mongod.py` | recompute / verify gold answers with a real `mongod` |
| `run_eval.py` | evaluate a model — `--mode base` (default), `rules`, or `decomp` |
| `run_delegate.py` | engine-delegation condition: the model runs the filter via a tool |

## Conditions

These map to the conditions E0–E3 in the paper:

- **base** (paper: **E0**) — the model reasons and answers.
- **rules** (`--mode rules`; paper: **E1**) — a one-paragraph rule sheet (one rule per footgun family) is prepended.
- **decomp** (`--mode decomp`; paper: **E2**) — the model judges each document before assembling the set.
- **delegation** (`run_delegate.py`; paper: **E3**) — the model may call a `mongo_find` tool that runs a
  filter on the collection and returns the matching `_id` values. Each item also records
  whether the executed filter preserved the intended query's semantics.

Providers: `anthropic`, `openai`, `gemini`, `deepseek` (OpenAI-compatible endpoints). Pass
`--effort medium` for models that support a reasoning-effort setting.

## License

The benchmark data and code are released under the MIT License.
