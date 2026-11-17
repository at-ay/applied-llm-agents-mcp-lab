# Lab: A fact-checking agent with a real MCP server

**Applied LLM Course · Week 7 lab: agents and MCP** · Duration: about 2 hours · Submit by: \[date\], via \[platform\]

In this lab you build an agent that checks claims about the LLM world against the live web. For each claim, it answers **true**, **false** or **can't tell**, and backs the answer with a source and a quote.

You don't write the tools yourself. They come from **Tavily**, a search service for AI agents, through its official MCP server. You also use a more capable model than before, through **OpenRouter**.

You will find out:
- what an agent adds over a fixed search-then-answer pipeline, and what it costs;
- what a system prompt can and can't make an agent do;
- which rules have to be enforced in code instead;
- why web content is untrusted input.

---

## 1. What you need

| Item | Where it comes from |
|---|---|
| Python, uv, the lab repository | as in earlier labs |
| OpenRouter API key | given to you by the instructor. It has a spending limit; don't share it |
| Tavily API key | create a free account at tavily.com. No credit card needed. The free plan gives 1,000 credits a month |

You don't need Node.js: the lab uses Tavily's **remote** MCP server, so nothing runs on your laptop except Python.

### Setup

Run everything from the top folder of the repository: the one with `pyproject.toml` in it.

1. Install the dependencies:
   ```
   uv sync
   ```
2. Copy `.env.example` to `.env` in the same folder:
   ```
   cp .env.example .env
   ```
   Then open `.env` and put your two keys after the `=` signs, with no quotes or spaces. `.env` is already in `.gitignore`, so it won't be committed:
   ```
   OPENROUTER_API_KEY=...
   TAVILY_API_KEY=...
   ```
3. Check the setup:
   ```
   uv run python lab/q0_check_setup.py
   ```
   It sends one message to the model and lists the Tavily tools. If both work, you're ready.

**Never** put keys in code, prompts, claims or screenshots. Keys belong in `.env` only.

### Files

| File | Contents |
|---|---|
| `lab/config.py` | model name, effort level, trusted domains, limits. All settings live here |
| `lab/common.py` | helpers: connect to Tavily, load claims, measure tool-definition tokens, sum tokens and cost, print a trace |
| `data/claims.csv` | 14 claims to check (`claim_id`, `claim`) |
| `data/expected.csv` | the answer key (`claim_id`, `verdict`, `source`, `checked_date`) |
| `lab/score.py` | scores an `answers.jsonl` file against the answer key, or compares two files |
| `lab/q1_inspect.py` … `lab/q5_checks.py` | one file per question: a starter with `YOUR TURN` parts, except `q2_baseline.py`, which is complete |
| `lab/lab-sheet.docx` | the Word answer sheet you hand in |

The starter files stop with a "fill in" message until you complete each `YOUR TURN` part.

---

## 2. Background

### The claims

`claims.csv` holds short statements about LLM models, services and tools: prices, limits, parameters, features. Some are true, some are false, and some can't be settled from reliable sources. Several were true once and aren't any more, which makes them traps for a model's out-of-date knowledge.

The three verdicts mean:

| Verdict | Meaning |
|---|---|
| `true` | a reliable, current source states it |
| `false` | a reliable, current source states something that contradicts it |
| `cant_tell` | no reliable, current source settles it. *Not finding a source is not the same as false* |

### The Tavily MCP server

The server offers five tools. You use two of them:

| Tool | Input | Returns |
|---|---|---|
| `tavily_search` | a query, plus options such as domains to include | results: title, URL, short snippet |
| `tavily_extract` | one or more URLs | the full text of each page, as clean markdown or text |

A snippet is a short piece of a page, often without its context or date. Extract gives the whole page, but costs many more tokens: one long page can be 20,000 tokens.

Tavily is not a search engine with its own index. It collects results from other sources, so "the agent searched the web" really means "the agent searched what Tavily returned".

### The model

You use **Claude Haiku 5.5** through OpenRouter, set in `config.py`. Three things differ from Ollama:
- There is no `temperature`. The model doesn't support it, and OpenRouter **silently ignores** it if you send one, so setting it changes nothing. Instead, the model has an **effort** setting (`low`, `medium`, `high`, `max`) that controls how much it thinks and writes. The lab uses `medium`, set in `config.py`; keep it as it is.
- Every run costs real money, usually a few cents for all claims. Your key has a limit, so don't leave runs in a loop.
- OpenRouter reports the cost of every model call, in `response_metadata["token_usage"]["cost"]`, so you don't need a price table.

---

## 3. Tasks

Each question builds on the previous one. Save the output of every scored run: you need the numbers in Q6.

### Q1: Connect and inspect (`lab/q1_inspect.py`)

Connect to the Tavily MCP server with the MCP client, as in week 5's MCP example, but now with someone else's server:

```python
client = MultiServerMCPClient({
    "tavily": {
        "transport": "http",
        "url": "https://mcp.tavily.com/mcp/",
        "headers": {"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
    }
})
tools = await client.get_tools()
```

The key goes in a header, not in the URL: URLs end up in logs and error messages.

**YOUR TURN**
1. Print the name, description and argument schema of every tool.
2. Measure how many input tokens the tool definitions cost on every model call. `common.py` has `tool_definition_tokens(model, tools)`: it sends one short message with and without the tools and compares the input tokens the model reports.
3. Keep only `tavily_search` and `tavily_extract`. Measure the token cost again.

**Think about**
- Why are crawl, map and research not needed for this task? Answer in one sentence.
- The tool descriptions were written by Tavily, not by you. What does that mean for how well the model will use them?
- `tavily_search` lets the model choose `include_domains` itself. Is that a good idea for a fact checker?

### Q2: The baseline (`lab/q2_baseline.py`)

The baseline is given complete. It makes no decisions. For each claim it:
1. calls the `tavily_search` tool **directly from Python** (`await search_tool.ainvoke({"query": claim})`, not through an agent), with the claim text as the query;
2. takes the top 3 results;
3. makes **one** model call with the claim and the 3 snippets, asking for a verdict, a source URL and a quote.

It uses the same Pydantic schema as the agent (Q3), with `model.with_structured_output(ClaimVerdict)`.

**YOUR TURN**
1. Read `check_claim_baseline(claim)` and find the three steps above in the code.
2. Run all claims, write `answers_baseline.jsonl`, and score it:
   ```
   uv run python lab/score.py answers_baseline.jsonl
   ```

**Think about**
- On which kinds of claims does the baseline do well? Where does it fail?

### Q3: The agent, minimal version (`lab/q3_agent.py`)

Build the agent with the two tools, a one-line system prompt and a structured output schema:

```python
class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Literal["true", "false", "cant_tell"]
    source_url: str | None
    evidence: str | None = Field(description="A short verbatim quote copied exactly from the source page")
    checked_date: str | None  # the date the source gives, if any

agent = create_agent(
    model,
    tools,
    system_prompt="Check whether the claim is true.",
    response_format=ClaimVerdict,
)
```

The field description matters: without it, the model writes a summary ("The page states that…") instead of a quote, and Q5's check can't work.

Every agent run (Q3 to Q5) has a safety cap: at most `MAX_MODEL_CALLS` model calls per claim (12, set in `config.py`). A run that hits the cap stops without an answer and is recorded as `cant_tell`, with `check_failed` set to `no answer (model call limit)`. Don't raise the cap to make a run finish: look at its trace instead.

**YOUR TURN**
1. Run the agent on **one** claim of your choice and print the full trace: `uv run python lab/q3_agent.py --claim c01` prints it with `print_trace(result["messages"])`. Which searches did it run? Which pages did it extract? Did it search at all?
2. Run the agent on all claims and write `answers_q3.jsonl`. For each claim, record the number of tool calls, input and output tokens, and the number of Tavily calls. The agent returns its structured answer as one extra tool call, named `ClaimVerdict`: don't count it.
3. Score the file. Print the total cost of the run: the sum of `response_metadata["token_usage"]["cost"]` over the model's messages (`common.py` has `run_cost(messages)`).

**Think about**
- Find one claim the agent answered without searching, or from a snippet alone. Was the answer right?
- Which decisions did the agent make that the baseline couldn't?
- Run your one claim a second time. Did the agent take the same steps?

### Q4: Write the system prompt (`lab/q4_prompt.py`)

Replace the one-line prompt with a real system prompt. Write it yourself as a plain string; don't use prompt templates. It should cover at least:
- **the verdicts:** what counts as true, false and can't tell (see the table in section 2);
- **sources:** which kinds of sources to prefer, and which not to trust;
- **no answers from memory:** every verdict must rest on a page retrieved in this run;
- **freshness:** what to do with an old page, or a page with no date;
- **conflicts:** what to do when sources disagree;
- **when to stop:** after how many searches without a reliable source to answer `cant_tell`.

**YOUR TURN**
1. Write the prompt in `SYSTEM_PROMPT`.
2. Run all claims, write `answers_q4.jsonl`, and score it.
3. Compare with Q3: how many answers were **overconfident** (true or false where the key says `cant_tell`)?

**Think about**
- Which rule in your prompt made the biggest difference? How do you know?
- Find one rule the agent broke anyway. What does that tell you about prompts?

### Q5: Enforce the rules in code (`lab/q5_checks.py`)

Some rules must never be broken: a verdict must not cite a page that was never retrieved, or quote text that isn't on it. A prompt can only ask for this. Code can enforce it.

**YOUR TURN**
1. Write `check_answer(answer, messages)`. It returns the answer unchanged if both checks pass:
   - `source_url` appears in the output of a tool call in this run;
   - `evidence` appears in the output of a tool call in this run, a search snippet or an extracted page (compare after normalising whitespace, case and quote marks).

   If either check fails, it returns the answer with `verdict="cant_tell"` and records which check failed.
2. Limit searches to a list of trusted domains. Set the list in `config.py` and pass it to the server, not to the model. The Tavily server reads default search options from a `DEFAULT_PARAMETERS` header, and these win over anything the model sends:
   ```python
   "headers": {
       "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
       "DEFAULT_PARAMETERS": json.dumps({"include_domains": TRUSTED_DOMAINS}),
   },
   ```
3. Limit the agent's searches with LangChain's call-limit middleware:
   ```python
   from langchain.agents.middleware import ToolCallLimitMiddleware

   agent = create_agent(
       model, tools, system_prompt=SYSTEM_PROMPT, response_format=ClaimVerdict,
       middleware=[ToolCallLimitMiddleware(tool_name="tavily_search", run_limit=MAX_SEARCHES,
                                           exit_behavior="continue")],
   )
   ```
   With `"continue"`, searches over the limit are blocked with an error message and the agent still has to give a verdict. (With `"end"`, the run stops and `result["structured_response"]` is `None`. Your code would then have to record `cant_tell`.)
4. Run all claims with your Q4 prompt plus these checks, write `answers_q5.jsonl`, and score it.

**Think about**
- How many answers did the checks change? Were those answers wrong before?
- Which domains did you trust, and what did the domain list cost you (claims that became `cant_tell`)?
- The domain list applies to `tavily_search` only. Can the agent still read a page from a domain you don't trust? How?
- Find one answer whose quote came from a search snippet, not an extracted page. Was the snippet enough to settle the claim?

### Q6: Analyse and report

Fill in this table from your runs. `lab/score.py` prints each row; Wrong doesn't include the overconfident answers.

| Version | Correct | Wrong | Overconfident | Tool calls | Input tokens | Output tokens | Tavily calls | Cost |
|---|---|---|---|---|---|---|---|---|
| Q2 baseline | | | | | | | | |
| Q3 minimal agent | | | | | | | | |
| Q4 system prompt | | | | | | | | |
| Q5 prompt + checks | | | | | | | | |

Then write a short report (max. 1 page):
1. **Where the agent helps.** Name the claims where the agent beat the baseline, and say why: a two-part claim, an outdated source, conflicting sources, a misleading snippet.
2. **Where it only costs.** Name the claims the baseline already got right, and the extra cost of the agent on them.
3. **Failures.** Show two or three failures from your final version. For each, point to the step in the trace where it went wrong, and explain why.
4. **Prompt or code?** Which improvements came from the prompt, and which from the code checks? Give one rule you'd always put in code, and one you'd leave to the prompt.
5. **Is the difference real?** Runs vary, and 14 claims is a small set. Take the two versions you most want to compare, count the claims that only one of them got right, and say whether the difference could be chance (the paired comparison from week 6). `uv run python lab/score.py answers_q3.jsonl --compare answers_q5.jsonl` does the counting.

---

## 4. What to submit

Fill in the Word answer sheet, [lab-sheet.docx](lab-sheet.docx). It has a box for every answer: your answers to the *Think about* questions, the results table and the report (Q6), and one annotated trace (the trace of one interesting claim, with a short note on each step). Hand it in together with these files:

- your completed `.py` files
- `answers_baseline.jsonl`, `answers_q3.jsonl`, `answers_q4.jsonl`, `answers_q5.jsonl`

## 5. Rules

- Use the current LangChain API (1.x). Build messages as lists of dicts and f-strings; don't use prompt templates.
- Build the agent with LangChain's `create_agent` (from `langchain.agents`). Don't write your own tool loop: the call limits in Q5 and the safety cap are LangChain middleware and work only with `create_agent`.
- Use only `tavily_search` and `tavily_extract`.
- Keep all settings (model, effort, domains, limits) in `config.py`.
- Keys stay in `.env`. Don't commit, print or share them.
- Watch your spending: run one claim before running the whole file, and stop runs that loop. Check your remaining Tavily credits on the Tavily dashboard.

## 6. Troubleshooting

| Problem | Likely cause |
|---|---|
| 401 / "invalid API key" | wrong key in `.env`, or `.env` not loaded (it must be in the repository's top folder, and you must run from there) |
| 402 from OpenRouter | your key's spending limit is used up; ask the instructor |
| Tavily "rate limited" or out of credits | wait, or check your usage on the Tavily dashboard |
| "Tool call limit reached" in the trace | the agent hit `MAX_SEARCHES`; check the trace for a search loop |
| `GraphRecursionError` | you passed `recursion_limit` instead of using the middleware in Q5 |
| One claim costs far more than the others | the agent extracted several long pages; look at the extract calls in the trace |
| "Model call limits exceeded" in the trace | the run hit MAX_MODEL_CALLS and was recorded as cant_tell; check the trace for a loop |
