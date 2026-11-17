"""Lab · Q0 · Check the setup: one message to the model, and the Tavily tools.

    uv run python lab/q0_check_setup.py
"""

from common import lab_tools, make_model, run, run_cost, tavily_client
import config


async def main():
    model = make_model()
    reply = await model.ainvoke("Reply with the single word: ready")
    print(f"Model {config.MODEL} (effort {config.EFFORT}) says: {reply.text.strip()!r}"
          f"  (cost ${run_cost([reply]):.6f})")

    tools = await (tavily_client()).get_tools()
    print(f"Tavily MCP server: {len(tools)} tools: {', '.join(t.name for t in tools)}")
    await lab_tools(tavily_client())  # fails if one of the lab's two tools is missing
    print("\nBoth work: you're ready.")


run(main)
