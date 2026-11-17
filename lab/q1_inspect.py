"""Lab · Q1 · Connect to the Tavily MCP server and inspect its tools.

    uv run python lab/q1_inspect.py
"""

from common import make_model, run, tavily_client, tool_definition_tokens
import config


async def main():
    model = make_model()
    client = tavily_client()
    tools = await client.get_tools()

    # YOUR TURN 1: for every tool, print its name (tool.name), its description (tool.description)
    #              and its arguments. tool.args_schema is a JSON schema (a dict): the arguments are
    #              in its "properties", each with a "type" and sometimes an "enum" of allowed values.

    # YOUR TURN 2: measure what the definitions of all the tools cost on every model call,
    #              with await tool_definition_tokens(model, tools).
    all_tokens = None

    # YOUR TURN 3: keep only the tools whose names are in config.TOOLS, and measure again.
    lab = None
    lab_tokens = None

    if all_tokens is None or lab_tokens is None:
        raise SystemExit("Fill in the YOUR TURN lines in lab/q1_inspect.py first.")
    print(f"\nAll {len(tools)} tools: {all_tokens:,} input tokens on every model call")
    print(f"{' and '.join(t.name for t in lab)}: {lab_tokens:,} input tokens on every model call")


run(main)
