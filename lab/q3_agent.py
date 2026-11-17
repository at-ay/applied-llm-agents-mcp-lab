"""Lab · Q3 · The agent, minimal version: two tools, a one-line prompt, a structured answer.

    uv run python lab/q3_agent.py --claim c01     (one claim, with its trace)
    uv run python lab/q3_agent.py                 (all claims, writes answers_q3.jsonl)

Q4 and Q5 import build_agent, check_claim and main from this file.
"""

import argparse

from langchain.agents import create_agent

from common import (ClaimVerdict, claim_message, lab_tools, load_claims, make_model, pick_claims, print_trace,
                    run, run_batch, run_stats, safety_limits, tavily_client)

SYSTEM_PROMPT = "Check whether the claim is true."


def build_agent(model, tools, system_prompt=SYSTEM_PROMPT, middleware=()):
    # YOUR TURN: return create_agent(...) with the model, the tools, the system prompt,
    #            ClaimVerdict as the response_format, and middleware=list(middleware) (Q5 uses it).
    return None


async def check_claim(agent, claim):
    """Run the agent on one claim. Returns (answer dict, stats dict, messages)."""
    if agent is None:
        raise SystemExit("Fill in the YOUR TURN lines in lab/q3_agent.py first.")
    # YOUR TURN: run the agent on one user message, claim_message(claim), with await agent.ainvoke(...).
    #            Take the messages (result["messages"]) and the structured answer
    #            (result.get("structured_response"), a ClaimVerdict or None).
    result = None
    messages = None
    verdict = None
    if messages is None:
        raise SystemExit("Fill in the YOUR TURN lines in check_claim() first.")
    if verdict is None:  # the run ended without an answer: it hit config.MAX_MODEL_CALLS
        answer = {"claim_id": claim["claim_id"], "verdict": "cant_tell", "source_url": None, "evidence": None,
                  "checked_date": None, "check_failed": "no answer (model call limit)"}
    else:
        answer = verdict.model_dump() | {"claim_id": claim["claim_id"]}
    return answer, run_stats(messages), messages


async def main(system_prompt=SYSTEM_PROMPT, out_name="answers_q3.jsonl", trusted_domains=None,
               middleware=(), check_answer=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", help="check one claim, e.g. c01, and print its trace")
    parser.add_argument("--limit", type=int, help="check only the first N claims")
    args = parser.parse_args()

    model = make_model()
    tools = await lab_tools(tavily_client(trusted_domains))
    agent = build_agent(model, tools, system_prompt, [*safety_limits(), *middleware])  # the cap is always on
    claims = pick_claims(load_claims(), args.claim, args.limit)

    async def check(claim):
        answer, stats, messages = await check_claim(agent, claim)
        if check_answer:
            answer = check_answer(answer, messages)
        return answer, stats

    if args.claim:
        answer, stats, messages = await check_claim(agent, claims[0])
        print_trace(messages)
        if check_answer:
            answer = check_answer(answer, messages)
        print(f"\nAnswer: {answer}")
        return

    await run_batch(check, claims, out_name)


if __name__ == "__main__":
    run(main)
