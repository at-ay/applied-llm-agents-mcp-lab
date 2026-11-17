"""Lab · Q4 · The same agent with a real system prompt.

    uv run python lab/q4_prompt.py --claim c01     (one claim, with its trace)
    uv run python lab/q4_prompt.py                 (all claims, writes answers_q4.jsonl)
"""

from common import run
from q3_agent import main

# YOUR TURN: write your system prompt as a plain string. Cover at least: the verdicts, sources,
#            no answers from memory, freshness, conflicts, and when to stop (see the lab sheet).
SYSTEM_PROMPT = None


if __name__ == "__main__":
    if not SYSTEM_PROMPT:
        raise SystemExit("Write SYSTEM_PROMPT in lab/q4_prompt.py first.")
    run(lambda: main(SYSTEM_PROMPT, "answers_q4.jsonl"))
