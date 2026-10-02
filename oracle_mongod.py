"""Recompute MongoBench gold answers with a real MongoDB server.

Each item's gold ``answer`` is the set of ``_id`` values a real ``mongod`` returns
for the item's query. This script inserts each item's documents into a scratch
collection, runs the query, and checks the result against the released gold set.

Usage:
  # start a local mongod, e.g.:
  #   mongod --port 27018 --dbpath /tmp/mbench --bind_ip 127.0.0.1
  python oracle_mongod.py --items mongobench.jsonl --uri mongodb://127.0.0.1:27018
"""
import json, argparse
from pymongo import MongoClient


def gold(coll, docs, query):
    coll.delete_many({})
    coll.insert_many([dict(d) for d in docs])
    return sorted(d["_id"] for d in coll.find(query))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="mongobench.jsonl")
    ap.add_argument("--uri", default="mongodb://127.0.0.1:27018")
    a = ap.parse_args()
    cli = MongoClient(a.uri)
    ver = cli.server_info()["version"]
    coll = cli.mongobench.col
    items = [json.loads(l) for l in open(a.items)]
    mismatch = []
    for it in items:
        if gold(coll, it["docs"], it["query"]) != it["answer"]:
            mismatch.append(it["id"])
    print(f"oracle = mongod {ver}; {len(items)} items checked")
    if mismatch:
        print(f"{len(mismatch)} items disagree with the released gold: {mismatch}")
    else:
        print("all released gold answers reproduced exactly.")


if __name__ == "__main__":
    main()
