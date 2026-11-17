"""Lab · Q5 · Enforce the rules in code: source and quote checks, trusted domains, a search limit.

    uv run python lab/q5_checks.py --claim c01     (one claim, with its trace)
    uv run python lab/q5_checks.py                 (all claims, writes answers_q5.jsonl)
"""

import re

from langchain.agents.middleware import ToolCallLimitMiddleware

from common import retrieved, run
import config
from q3_agent import main
from q4_prompt import SYSTEM_PROMPT

QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalise(text):
    """Lower case, straight quote marks, single spaces."""
    # YOUR TURN: return the text in lower case, with QUOTES applied (text.translate(QUOTES)) and every
    #            run of whitespace replaced by one space (re.sub). Treat None as "".
    return None


def check_answer(answer, messages):
    """Keep the answer only if its source and its quote come from this run's tool results."""
    if answer["verdict"] == "cant_tell":
        return answer  # nothing to check: cant_tell needs no source
    if normalise("A  “B”") is None:
        raise SystemExit("Fill in normalise() in lab/q5_checks.py first.")
    found = retrieved(messages)  # everything the tools returned: dicts with tool, url, title and text

    # YOUR TURN: source_ok is True if answer["source_url"] is the url of one of the results in found
    #            (ignore a trailing "/"); evidence_ok is True if answer["evidence"], after normalise(),
    #            is part of the normalised text of the results. Both are False when the field is empty.
    source_ok = None
    evidence_ok = None

    if source_ok is None or evidence_ok is None:
        raise SystemExit("Fill in the YOUR TURN lines in check_answer() first.")
    failed = [name for name, ok in (("source", source_ok), ("evidence", evidence_ok)) if not ok]
    if failed:
        return answer | {"verdict": "cant_tell", "check_failed": "+".join(failed),
                         "verdict_before_check": answer["verdict"]}
    return answer


# YOUR TURN: limit the searches: a ToolCallLimitMiddleware for "tavily_search" with
#            run_limit=config.MAX_SEARCHES and exit_behavior="continue" (see the lab sheet).
MIDDLEWARE = []

if __name__ == "__main__":
    if not SYSTEM_PROMPT:
        raise SystemExit("Write SYSTEM_PROMPT in lab/q4_prompt.py first (Q5 uses it).")
    if not config.TRUSTED_DOMAINS:
        raise SystemExit("Set TRUSTED_DOMAINS in lab/config.py first.")
    if not MIDDLEWARE:
        raise SystemExit("Fill in MIDDLEWARE in lab/q5_checks.py first.")
    run(lambda: main(SYSTEM_PROMPT, "answers_q5.jsonl", config.TRUSTED_DOMAINS, MIDDLEWARE, check_answer))
