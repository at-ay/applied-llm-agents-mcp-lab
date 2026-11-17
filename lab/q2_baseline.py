"""Lab · Q2 · The baseline: one search, the top results, one model call. No agent.  (complete: read it, run it)

    uv run python lab/q2_baseline.py                 (all claims, writes answers_baseline.jsonl)
    uv run python lab/q2_baseline.py --claim c01     (one claim, printed in full)
"""

import argparse

from common import (ClaimVerdict, claim_message, lab_tools, load_claims, make_model, pick_claims, run,
                    run_batch, run_stats, tavily_client, tool_results)
import config

PROMPT = "Check whether the claim is true, using only the search results below."


async def check_claim_baseline(claim, search, model):
    # 1. one Tavily search with the claim text as the query, called directly from Python
    output = await search.ainvoke({"query": claim["claim"], "max_results": config.BASELINE_RESULTS})

    # 2. the top results: title, URL and snippet
    results = tool_results(output)[: config.BASELINE_RESULTS]
    snippets = "\n\n".join(f"[{i}] {r['title']}\nURL: {r['url']}\n{r['text']}" for i, r in enumerate(results, 1))

    # 3. one model call with the claim and the snippets, answered in the ClaimVerdict schema
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"{claim_message(claim)}\n\nSearch results:\n\n{snippets}"},
    ]
    reply = await model.with_structured_output(ClaimVerdict, include_raw=True).ainvoke(messages)
    answer = reply["parsed"].model_dump() | {"claim_id": claim["claim_id"]}

    stats = run_stats([reply["raw"]]) | {"tool_calls": 1, "tavily_calls": 1, "searches": 1}
    return answer, stats, snippets


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", help="check one claim, e.g. c01, and print it in full")
    parser.add_argument("--limit", type=int, help="check only the first N claims")
    args = parser.parse_args()

    model = make_model()
    search = (await lab_tools(tavily_client()))[0]  # tavily_search
    claims = pick_claims(load_claims(), args.claim, args.limit)

    if args.claim:
        answer, stats, snippets = await check_claim_baseline(claims[0], search, model)
        print(f"{claim_message(claims[0])}\n\n{snippets}\n\nAnswer: {answer}\n{stats}")
        return

    async def check(claim):
        answer, stats, _ = await check_claim_baseline(claim, search, model)
        return answer, stats

    await run_batch(check, claims, "answers_baseline.jsonl")


run(main)
