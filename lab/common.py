"""Helpers for the lab: the model, the Tavily MCP server, the claims, tokens and cost, traces and batches.

You don't need to change this file.
"""

import asyncio
import csv
import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config

MODULE = Path(__file__).resolve().parent.parent  # the 07-agents folder
DATA = MODULE / "data"
load_dotenv(MODULE / ".env")


# ---------------------------------------------------------------- the answer schema (Q2–Q5)

class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Literal["true", "false", "cant_tell"]
    source_url: str | None
    evidence: str | None = Field(description="A short verbatim quote copied exactly from the source page")
    checked_date: str | None  # the date the source gives, if any


# ---------------------------------------------------------------- the model and the server

def make_model(effort=None):
    """Claude Haiku 5.5 through OpenRouter. OpenRouter calls the effort setting 'verbosity'."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing: put it in the .env file in the 07-agents folder.")
    return ChatOpenAI(model=config.MODEL, base_url=config.OPENROUTER_BASE_URL, api_key=key,
                      extra_body={"verbosity": effort or config.EFFORT})


def tavily_client(trusted_domains=None):
    """An MCP client for Tavily's remote server. The key goes in a header, never in the URL.

    With trusted_domains, the server limits every search to those domains (Q5),
    whatever the model asks for.
    """
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise SystemExit("TAVILY_API_KEY is missing: put it in the .env file in the 07-agents folder.")
    headers = {"Authorization": f"Bearer {key}"}
    if trusted_domains:
        headers["DEFAULT_PARAMETERS"] = json.dumps({"include_domains": list(trusted_domains)})
    return MultiServerMCPClient({"tavily": {"transport": "http", "url": config.TAVILY_MCP_URL, "headers": headers}})


async def lab_tools(client):
    """Only the lab's two tools (config.TOOLS), in that order."""
    tools = {t.name: t for t in await client.get_tools()}
    return [tools[name] for name in config.TOOLS]


def safety_limits():
    """The safety cap every agent gets (config.MAX_MODEL_CALLS model calls per claim).

    At the cap the run stops without a structured answer; check_claim then records cant_tell.
    """
    return [ModelCallLimitMiddleware(run_limit=config.MAX_MODEL_CALLS, exit_behavior="end")]


async def tool_definition_tokens(model, tools):
    """Input tokens the tool definitions add to every model call, measured by the model itself.

    Sends one short message with and without the tools and compares the input tokens
    the model reports. The difference also includes the model's fixed tool-use instructions.
    """
    message = [{"role": "user", "content": "Say OK."}]
    without = (await model.ainvoke(message)).usage_metadata["input_tokens"]
    with_tools = (await model.bind_tools(tools).ainvoke(message)).usage_metadata["input_tokens"]
    return with_tools - without


# ---------------------------------------------------------------- claims

def load_claims(path=DATA / "claims.csv"):
    """The claims as a list of dicts with claim_id and claim."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{"claim_id": r["claim_id"], "claim": r["claim"]} for r in csv.DictReader(f)]


def pick_claims(claims, claim_id=None, limit=None):
    """One claim by its id, or the first `limit` claims, or all of them."""
    if claim_id:
        claims = [c for c in claims if c["claim_id"] == claim_id]
        if not claims:
            raise SystemExit(f"no claim with id {claim_id} in data/claims.csv")
        return claims
    return claims[:limit] if limit else claims


def claim_message(claim):
    """The user message for one claim."""
    return f"Claim {claim['claim_id']}: {claim['claim']}"


# ---------------------------------------------------------------- what a run used

def _ai(messages):
    return [m for m in messages if isinstance(m, AIMessage)]


def run_cost(messages):
    """The run's cost in US dollars. OpenRouter reports each model call's cost, which arrives
    in response_metadata["token_usage"]["cost"] (in response_metadata["cost"] with ChatOpenRouter)."""
    return sum(m.response_metadata.get("cost") or (m.response_metadata.get("token_usage") or {}).get("cost") or 0
               for m in _ai(messages))


def run_stats(messages):
    """Tool calls, Tavily calls, tokens and cost of one run.

    The agent returns its structured answer as an extra tool call named ClaimVerdict:
    it isn't counted as a tool call.
    """
    calls = [tc["name"] for m in _ai(messages) for tc in m.tool_calls if tc["name"] != "ClaimVerdict"]
    return {
        "tool_calls": len(calls),
        "tavily_calls": sum(name in config.TOOLS for name in calls),
        "searches": calls.count("tavily_search"),
        "extracts": calls.count("tavily_extract"),
        "input_tokens": sum((m.usage_metadata or {}).get("input_tokens", 0) for m in _ai(messages)),
        "output_tokens": sum((m.usage_metadata or {}).get("output_tokens", 0) for m in _ai(messages)),
        "cost": round(run_cost(messages), 6),
    }


# ---------------------------------------------------------------- what the tools returned

def _blocks_text(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def tool_results(content):
    """The results of one Tavily call (a ToolMessage's content or a tool's output) as a list of dicts.

    Each dict has url, title and text: the snippet for a search, the whole page for an extract.
    """
    text = _blocks_text(content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [{"url": "", "title": "", "text": text}]
    return [{"url": r.get("url", ""), "title": r.get("title", ""),
             "text": r.get("raw_content") or r.get("content") or ""}
            for r in data.get("results", [])]


def retrieved(messages):
    """Everything the Tavily tools returned in this run: a list of dicts with tool, url, title and text."""
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name in config.TOOLS:
            out += [{"tool": m.name, **r} for r in tool_results(m.content)]
    return out


# ---------------------------------------------------------------- traces

def _short(text, n):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


def print_trace(messages):
    """Print a run step by step: the model's tool calls, what came back, and the answer."""
    step = 0
    for m in messages:
        if isinstance(m, AIMessage):
            step += 1
            u = m.usage_metadata or {}
            print(f"\n[{step}] model  ({u.get('input_tokens', 0)} in, {u.get('output_tokens', 0)} out)")
            if m.text.strip():
                print("    says:", _short(m.text, 300))
            for tc in m.tool_calls:
                print(f"    calls {tc['name']}({_short(json.dumps(tc['args']), 200)})")
        elif isinstance(m, ToolMessage):
            if m.name in config.TOOLS:
                results = tool_results(m.content)
                print(f"    ← {m.name}: {len(results)} result(s), {len(_blocks_text(m.content)):,} characters")
                for r in results:
                    print(f"        {r['url']}  {_short(r['text'], 120)!r}")
            else:
                print(f"    ← {m.name}: {_short(m.content, 200)}")
        elif getattr(m, "type", "") == "human":
            print("user:", _short(m.content, 300))
    stats = run_stats(messages)
    print(f"\n{stats['tool_calls']} tool calls ({stats['searches']} searches, {stats['extracts']} extracts), "
          f"{stats['input_tokens']:,} input and {stats['output_tokens']:,} output tokens, ${stats['cost']:.4f}")


# ---------------------------------------------------------------- batches

async def run_batch(check, claims, out_path):
    """Run check(claim) on every claim, a few at a time, and write one JSON line per claim.

    check is an async function that returns (answer, stats): the answer a dict with the
    ClaimVerdict fields (and any extra fields you add), stats a dict from run_stats.
    """
    sem = asyncio.Semaphore(config.CONCURRENCY)
    rows = [None] * len(claims)

    async def one(i, claim):
        async with sem:
            t = time.time()
            try:
                answer, stats = await check(claim)
            except Exception as e:  # a failed claim is recorded, not lost
                answer, stats = {"claim_id": claim["claim_id"], "verdict": "cant_tell", "source_url": None,
                                 "evidence": None, "checked_date": None, "error": f"{type(e).__name__}: {e}"[:300]}, {}
            row = {**answer, **stats, "claim_id": claim["claim_id"], "seconds": round(time.time() - t, 1)}
            rows[i] = row
            print(f"{row['claim_id']}  {row['verdict']:9}  {stats.get('tool_calls', 0):2} tool calls  "
                  f"${stats.get('cost', 0):.4f}  {row['seconds']:5.1f}s" + (f"  ERROR {row['error']}" if "error" in row else ""))

    start = time.time()
    await asyncio.gather(*(one(i, c) for i, c in enumerate(claims)))
    out = MODULE / out_path
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cost = sum(r.get("cost", 0) for r in rows)
    print(f"\n{len(rows)} claims in {time.time() - start:.0f}s, ${cost:.4f}. Wrote {out.name}; "
          f"score it with: uv run python lab/score.py {out.name}")
    return rows


def run(main):
    """Run an async main() and stop with a short message on Ctrl+C."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit("\nStopped.")
