"""Evaluate a model on MongoBench.

Given each item's documents and query, the model predicts the set of matching
``_id`` values; the prediction is scored by exact-set match against the gold set.

Three prompting conditions:
  base    the model reasons and answers (default)
  rules   prepend a one-paragraph rule sheet (one rule per footgun family)
  decomp  ask the model to judge each document before assembling the set

Usage:
  export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY
  python run_eval.py --model claude-opus-4-8 --provider anthropic \\
      --items mongobench.jsonl --out results.jsonl --effort medium
"""
import json, re, os, time, argparse
from concurrent.futures import ThreadPoolExecutor

PROMPT = """You are given MongoDB documents and a query. Return the _id values of the documents that MATCH the query, per MongoDB's exact matching semantics.

Documents:
{docs}

Query:
{query}

Think step by step, then end with exactly one line:
ANSWER: [comma-separated _id values, ascending, or empty]"""

RULES = """MongoDB matching rules to apply exactly:
1. Array-implicit: {f:v} or {f:{$gt:x}} matches if f equals/satisfies it, OR f is an array with an element that does.
2. Implicit-AND on arrays: {f:{$gt:a,$lt:b}} = some element >a AND some (maybe different) element <b; $elemMatch = ONE element satisfies all.
3. null/missing: {f:null} matches f=null, f missing, OR an array containing null. {f:{$ne:null}} matches only docs where f is present, not null, and (if an array) has no null element.
4. Comparison vs null: {f:{$gt:null}} and {f:{$lt:null}} match NOTHING; but {f:{$gte:null}} and {f:{$lte:null}} match docs where f is null, missing, or an array containing null.
5. Type-bracketing: comparisons only compare within the same BSON type (number-number, string-string); cross-type never matches.
6. $ne/$nin on arrays: exclude the doc if ANY element matches.
7. Embedded-doc equality {f:{a:..,b:..}}: requires EXACT field set AND field ORDER (or an array element that is such a doc).
8. Dot notation: a.b applies to any array element's b; a.0 is positional (first element). $exists:false = field missing only. a.b does NOT reach into arrays nested inside arrays.
9. Nested arrays: array-implicit matching looks only one level deep. {f:1} does not match f=[[1,2]]; {f:[1,2]} matches f equal to [1,2] OR an array having [1,2] as an element.
10. $size:n matches arrays with exactly n top-level elements (never non-arrays). $all:[x,y] matches if the field contains every listed value (array-implicit, so a scalar can match a one-value $all); $all with $elemMatch clauses requires each clause to be satisfied by some element.

"""

DECOMP_PROMPT = """You are given MongoDB documents and a query. Decide which documents MATCH, per MongoDB's exact matching semantics.

Documents:
{docs}

Query:
{query}

Evaluate EACH document one at a time, in _id order. For each write: "_id <n>: MATCH/NO - <one-line reason>", considering field type, array-vs-scalar, missing/null, and field order for embedded documents. After evaluating every document, end with exactly one line:
ANSWER: [comma-separated _id values of MATCH documents, ascending, or empty]"""

OPENAI_BASE = {"openai": None,
               "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "deepseek": "https://api.deepseek.com"}
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}


def parse(raw):
    m = re.search(r"ANSWER:\s*\[?([0-9,\s]*)\]?\s*$", (raw or "").strip(), re.I | re.S)
    if not m:
        m = re.search(r"ANSWER:\s*(.*)$", (raw or "").strip(), re.I | re.S)
    body = m.group(1) if m else ""
    return sorted(int(x) for x in re.findall(r"\d+", body))


def call_anthropic(model, prompt, effort):
    import anthropic
    client = anthropic.Anthropic()
    kw = dict(model=model, max_tokens=16000, messages=[{"role": "user", "content": prompt}])
    if effort:
        kw.update(thinking={"type": "adaptive"}, output_config={"effort": effort})
    r = client.messages.create(**kw)
    return "".join(b.text for b in r.content if b.type == "text"), getattr(r, "model", None)


def call_openai_compatible(provider, model, prompt, effort):
    from openai import OpenAI
    base = OPENAI_BASE[provider]
    kw = {"timeout": 300, "max_retries": 2}
    if base:
        kw.update(api_key=os.environ[KEY_ENV[provider]], base_url=base)
    client = OpenAI(**kw)
    extra = {"reasoning_effort": effort} if effort else {}
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **extra)
    return r.choices[0].message.content, getattr(r, "model", None)


def call(provider, model, prompt, effort):
    if provider == "anthropic":
        return call_anthropic(model, prompt, effort)
    return call_openai_compatible(provider, model, prompt, effort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini", "deepseek"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="base", choices=["base", "rules", "decomp"])
    ap.add_argument("--effort", default=None, help="reasoning effort for models that support it (e.g. medium)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items)]
    base_prompt = DECOMP_PROMPT if a.mode == "decomp" else PROMPT
    prefix = RULES if a.mode == "rules" else ""

    def solve(it):
        prompt = prefix + base_prompt.format(docs=json.dumps(it["docs"]), query=json.dumps(it["query"]))
        for attempt in range(4):
            try:
                return it, *call(a.provider, a.model, prompt, a.effort)
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(5 * 2 ** attempt)  # back off on transient API errors

    n_ok = 0
    with open(a.out, "w") as out, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for it, raw, api_model in ex.map(solve, items):
            pred = parse(raw)
            ok = pred == it["answer"]
            n_ok += ok
            out.write(json.dumps({"id": it["id"], "pred": pred, "answer": it["answer"],
                                  "correct": ok, "api_model": api_model, "raw": raw}) + "\n")
            print(("OK " if ok else "XX "), it["id"], "pred", pred, flush=True)
    print(f"\n{a.model} ({a.mode}): {n_ok}/{len(items)} = {n_ok/len(items):.1%}")


if __name__ == "__main__":
    main()
