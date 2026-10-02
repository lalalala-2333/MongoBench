"""Engine-delegation condition (E3) for MongoBench.

The model is given a ``mongo_find`` tool that runs a filter on the item's collection
in a real MongoDB server and returns the matching ``_id`` values. We score the
model's final answer by exact-set match, and also record, for each item, whether the
filter it executed preserved the semantics of the intended query (see
``semantically_same``: a conservative check that treats order-irrelevant rewrites as
equivalent but a change to an embedded document's field order or key as altered).

Usage:
  # start a local mongod first (see oracle_mongod.py)
  python run_delegate.py --model gpt-5.5 --provider openai \\
      --items mongobench.jsonl --out results_e3.jsonl --effort medium
"""
import json, os, re, argparse, time
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient

PROMPT = """You are given MongoDB documents (already stored in a collection) and a query. Return the _id values of the documents that MATCH the query, per MongoDB's exact matching semantics.

You have a tool `mongo_find` that runs a find() filter on this collection with a real MongoDB server and returns the matching _id values.

Documents:
{docs}

Query:
{query}

When you are done, end with exactly one line:
ANSWER: [comma-separated _id values, ascending, or empty]"""

TOOL_DESC = "Run a MongoDB find() filter on the collection and return the sorted list of matching _id values."
SCHEMA = {"type": "object", "properties": {"filter": {"type": "object", "description": "The find() filter document."}},
          "required": ["filter"]}
MAX_TURNS = 10
OPENAI_BASE = {"openai": None,
               "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "deepseek": "https://api.deepseek.com"}
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}


def parse(raw):
    m = re.search(r"ANSWER:\s*\[?([0-9,\s]*)\]?\s*$", (raw or "").strip(), re.I | re.S)
    if not m:
        m = re.search(r"ANSWER:\s*(.*)$", (raw or "").strip(), re.I | re.S)
    return sorted(int(x) for x in re.findall(r"\d+", m.group(1) if m else ""))


def _canon(o, pos):
    """Canonicalize a query for semantic comparison. Order-irrelevant positions
    (top-level conjunction, operator dicts, $and/$or elements) are normalized;
    order-relevant positions (embedded-document values, array values) are preserved."""
    if isinstance(o, dict):
        if pos == "val":
            if o and all(k.startswith("$") for k in o):
                return _canon(o, "cond")
            return ["OBJ"] + [[k, _canon(v, "val")] for k, v in o.items()]
        items = []
        for k, v in o.items():
            if k in ("$and", "$or", "$nor"):
                items.append([k, sorted((_canon(e, "cond") for e in v), key=repr)])
            elif k == "$elemMatch":
                items.append([k, _canon(v, "cond")])
            elif k.startswith("$"):
                items.append([k, _canon(v, "val")])
            elif isinstance(v, dict) and v and all(x.startswith("$") for x in v):
                items.append([k, _canon(v, "cond")])
            else:
                items.append([k, _canon(v, "val")])
        return sorted(items, key=repr)
    if isinstance(o, list):
        return ["ARR"] + [_canon(e, "val") for e in o]
    return o


def semantically_same(a, b):
    """Conservative preservation check: True is reliable (the rewrites are equivalent);
    False does not prove MongoDB-inequivalence, as other equivalent rewrites are not normalized here."""
    return _canon(a, "cond") == _canon(b, "cond")


def execute(uri, docs, flt, db):
    coll = MongoClient(uri)[db].col
    try:
        coll.delete_many({})
        coll.insert_many([dict(d) for d in docs])
        return json.dumps(sorted(d["_id"] for d in coll.find(flt)))
    except Exception as e:
        return f"error: {type(e).__name__}: {str(e)[:200]}"


def run_anthropic(model, prompt, tool_fn, effort):
    import anthropic
    client = anthropic.Anthropic(timeout=300, max_retries=2)
    tools = [{"name": "mongo_find", "description": TOOL_DESC, "input_schema": SCHEMA}]
    kw = {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}} if effort else {}
    msgs = [{"role": "user", "content": prompt}]
    calls = []
    for _ in range(MAX_TURNS):
        r = client.messages.create(model=model, max_tokens=16000, tools=tools, messages=msgs, **kw)
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text"), calls
        msgs.append({"role": "assistant", "content": r.content})
        results = []
        for b in r.content:
            if b.type == "tool_use":
                flt = b.input.get("filter", {}) if isinstance(b.input, dict) else {}
                calls.append(flt)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": tool_fn(flt)})
        msgs.append({"role": "user", "content": results})
    return "", calls


def run_openai_responses(client, model, prompt, tool_fn, effort):
    # OpenAI reasoning models combine function tools with reasoning effort via the Responses API.
    tools = [{"type": "function", "name": "mongo_find", "description": TOOL_DESC, "parameters": SCHEMA}]
    kw = {"reasoning": {"effort": effort}} if effort else {}
    inp = [{"role": "user", "content": prompt}]
    calls = []
    for _ in range(MAX_TURNS):
        r = client.responses.create(model=model, input=inp, tools=tools, **kw)
        fcs = [o for o in r.output if o.type == "function_call"]
        if not fcs:
            return r.output_text or "", calls
        inp += [o.model_dump(exclude_none=True) for o in r.output]
        for fc in fcs:
            try:
                flt = json.loads(fc.arguments).get("filter", {})
            except Exception:
                flt = {}
            calls.append(flt)
            inp.append({"type": "function_call_output", "call_id": fc.call_id, "output": tool_fn(flt)})
    return "", calls


def run_openai_compat(client, model, prompt, tool_fn, effort):
    tools = [{"type": "function", "function": {"name": "mongo_find", "description": TOOL_DESC, "parameters": SCHEMA}}]
    extra = {"reasoning_effort": effort} if effort else {}
    msgs = [{"role": "user", "content": prompt}]
    calls = []
    for _ in range(MAX_TURNS):
        m = client.chat.completions.create(model=model, messages=msgs, tools=tools, **extra).choices[0].message
        if not m.tool_calls:
            return m.content or "", calls
        msgs.append(m.model_dump(exclude_none=True))  # keep provider-specific fields (e.g. thought signatures)
        for t in m.tool_calls:
            try:
                flt = json.loads(t.function.arguments).get("filter", {})
            except Exception:
                flt = {}
            calls.append(flt)
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": tool_fn(flt)})
    return "", calls


def runner(provider):
    if provider == "anthropic":
        return lambda m, p, fn, eff: run_anthropic(m, p, fn, eff)
    from openai import OpenAI
    if provider == "openai":
        client = OpenAI(timeout=300, max_retries=2)
        return lambda m, p, fn, eff: run_openai_responses(client, m, p, fn, eff)
    client = OpenAI(api_key=os.environ[KEY_ENV[provider]], timeout=300, max_retries=2, base_url=OPENAI_BASE[provider])
    return lambda m, p, fn, eff: run_openai_compat(client, m, p, fn, eff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini", "deepseek"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--uri", default="mongodb://127.0.0.1:27018")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run = runner(a.provider)
    items = [json.loads(l) for l in open(a.items)]

    def solve(it):
        db = "deleg_" + re.sub(r"\W", "_", it["id"])
        tool_fn = lambda flt: execute(a.uri, it["docs"], flt, db)
        prompt = PROMPT.format(docs=json.dumps(it["docs"]), query=json.dumps(it["query"]))
        for attempt in range(4):
            try:
                return it, *run(a.model, prompt, tool_fn, a.effort)
            except Exception as e:
                if attempt == 3:
                    return it, f"error: {type(e).__name__}: {str(e)[:200]}", []
                time.sleep(5 * 2 ** attempt)

    n_ok = 0
    with open(a.out, "w") as out, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for it, raw, calls in ex.map(solve, items):
            pred = parse(raw)
            ok = pred == it["answer"]
            n_ok += ok
            out.write(json.dumps({"id": it["id"], "pred": pred, "answer": it["answer"], "correct": ok,
                                  "used_tool": bool(calls),
                                  "semantic_preserved": any(semantically_same(c, it["query"]) for c in calls),
                                  "tool_calls": calls, "raw": raw}) + "\n")
            print(("OK " if ok else "XX "), it["id"], "calls", len(calls), flush=True)
    print(f"\n{a.model} (E3): {n_ok}/{len(items)} = {n_ok/len(items):.1%}")


if __name__ == "__main__":
    main()
