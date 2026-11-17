# Applied LLM Course · Agents

The week 7 lab of the Applied LLM Course: agents and MCP.

- **`lab/`:** the lab sheet and its starter files. You build an agent that checks claims about the LLM world against the live web, with tools from Tavily's MCP server. The agent is built with LangChain's `create_agent` and gets its tools through `langchain-mcp-adapters`. See [lab/README.md](lab/README.md).
- **`data/`:** the claims to check and the answer key.

**This lab uses a cloud model.** Earlier labs ran `llama3.2:3b` on your own computer. An agent that searches, reads pages and decides when to stop needs a more capable model, so this lab uses **Claude Haiku 5.5** through [OpenRouter](https://openrouter.ai), with a key the instructor gives you. Search comes from [Tavily](https://tavily.com), with your own free key. Every run costs a few cents of the instructor's key, and uses Tavily credits from your free plan.

## Files

```
applied-llm-agents-mcp-lab/
├── README.md
├── pyproject.toml        dependencies, for uv
├── uv.lock               exact versions, for uv
├── .env.example          the template for .env
├── .env                  your two API keys (you create it from .env.example; never share or commit it)
├── data/
│   ├── claims.csv             the claims to check
│   └── expected.csv           the answer key
└── lab/
    ├── README.md              the lab sheet
    ├── lab-sheet.docx         the answer sheet, to hand in
    ├── config.py              all settings: model, effort, trusted domains, limits
    ├── common.py              helpers: model, Tavily, claims, tokens and cost, traces (don't change)
    ├── score.py               scores or compares answers files (don't change)
    └── q0_check_setup.py ...  one file per question
```

Running a question writes its answers (for example `answers_q3.jsonl`) to this folder.

## Installation

You need [uv](https://docs.astral.sh/uv/), as in earlier weeks. From this folder:

```bash
uv sync
```

Then copy the template to `.env`:

```bash
cp .env.example .env
```

and put your two keys in `.env`, after the `=` signs, with no quotes or spaces:

```
OPENROUTER_API_KEY=...
TAVILY_API_KEY=...
```

`.env` is in `.gitignore`, so git won't commit it. Never share it or put it in a screenshot.

and check the setup:

```bash
uv run python lab/q0_check_setup.py
```

Run every script from this folder, as `uv run python lab/<file>.py`.
