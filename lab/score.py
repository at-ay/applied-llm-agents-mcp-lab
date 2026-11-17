"""Score an answers file against the answer key, or compare two answers files.

    uv run python lab/score.py answers_q3.jsonl
    uv run python lab/score.py answers_q3.jsonl --compare answers_q5.jsonl

Correct:        the verdict is the key's verdict.
Overconfident:  true or false where the key says cant_tell.
Wrong:          every other mistake (true for false, false for true, or cant_tell where the key has an answer).
"""

import argparse
import csv
import json
from math import comb
from pathlib import Path

MODULE = Path(__file__).resolve().parent.parent


def load_key():
    with open(MODULE / "data" / "expected.csv", newline="", encoding="utf-8") as f:
        return {r["claim_id"]: r for r in csv.DictReader(f)}


def load_answers(name):
    path = Path(name) if Path(name).exists() else MODULE / name
    if not path.exists():
        raise SystemExit(f"{name} not found (run the question's script first)")
    with open(path, encoding="utf-8") as f:
        return {a["claim_id"]: a for a in map(json.loads, filter(str.strip, f))}


def outcome(answer, key):
    if answer["verdict"] == key["verdict"]:
        return "correct"
    if key["verdict"] == "cant_tell":
        return "overconfident"
    return "wrong"


def score(name, key):
    answers = load_answers(name)
    missing = sorted(set(key) - set(answers))
    print(f"{name}\n")
    print(f"{'claim':6} {'key':9} {'answer':9} {'result':13} {'tools':>5} {'cost':>8}")
    counts = {"correct": 0, "wrong": 0, "overconfident": 0}
    for cid in sorted(answers):
        if cid not in key:
            continue
        a, k = answers[cid], key[cid]
        o = outcome(a, k)
        counts[o] += 1
        note = f"  (ERROR: {a['error']})" if a.get("error") else f"  ({a['check_failed']})" if a.get("check_failed") else ""
        print(f"{cid:6} {k['verdict']:9} {a['verdict']:9} {o:13} {a.get('tool_calls', 0):5} ${a.get('cost', 0):.4f}{note}")
    total = lambda f: sum(a.get(f, 0) for a in answers.values())
    print(f"\nCorrect {counts['correct']} · Wrong {counts['wrong']} · Overconfident {counts['overconfident']}"
          f"  (of {len(key)} claims{f'; missing: {missing}' if missing else ''})")
    errors = sorted(c for c, a in answers.items() if a.get("error"))
    if errors:
        print(f"\n{len(errors)} claim(s) crashed and were recorded as cant_tell: {', '.join(errors)}. "
              "Look at the error, fix it, and run the whole file again.")
    print("\nFor the Q6 table:")
    print("| Correct | Wrong | Overconfident | Tool calls | Input tokens | Output tokens | Tavily calls | Cost |")
    print(f"| {counts['correct']} | {counts['wrong']} | {counts['overconfident']} | {total('tool_calls')} | "
          f"{total('input_tokens'):,} | {total('output_tokens'):,} | {total('tavily_calls')} | ${total('cost'):.4f} |")


def compare(name_a, name_b, key):
    """Paired comparison (week 6): which claims only one version got right, and could that be chance?"""
    a, b = load_answers(name_a), load_answers(name_b)
    ids = [c for c in sorted(key) if c in a and c in b]
    right = lambda ans, c: ans[c]["verdict"] == key[c]["verdict"]
    only_a = [c for c in ids if right(a, c) and not right(b, c)]
    only_b = [c for c in ids if right(b, c) and not right(a, c)]
    both = sum(right(a, c) and right(b, c) for c in ids)
    neither = len(ids) - both - len(only_a) - len(only_b)
    n = len(only_a) + len(only_b)
    # exact two-sided sign test: if both versions were equally good, each of the n changed claims
    # would be a coin flip; p is the chance of a split at least this uneven
    k = min(len(only_a), len(only_b))
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    print(f"{len(ids)} claims in both files\n")
    w = max(len(name_a), len(name_b)) + 7
    print(f"{'':{w}} {name_b + ' right':>{w}} {name_b + ' wrong':>{w}}")
    print(f"{name_a + ' right':{w}} {both:>{w}} {len(only_a):>{w}}")
    print(f"{name_a + ' wrong':{w}} {len(only_b):>{w}} {neither:>{w}}")
    print(f"\nOnly {name_a} right: {', '.join(only_a) or '-'}")
    print(f"Only {name_b} right: {', '.join(only_b) or '-'}")
    print(f"\nSign test on the {n} claims that differ: p = {p:.2f}"
          + ("  (could easily be chance)" if p > 0.05 else "  (unlikely to be chance alone)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers")
    parser.add_argument("--compare", metavar="OTHER", help="a second answers file to compare with")
    args = parser.parse_args()
    key = load_key()
    if args.compare:
        compare(args.answers, args.compare, key)
    else:
        score(args.answers, key)


if __name__ == "__main__":
    main()
