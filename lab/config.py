"""All settings for the lab. Change them here, never in the other files."""

# The model, through OpenRouter
MODEL = "anthropic/claude-haiku-5.5"
EFFORT = "medium"  # low, medium, high or max: how much the model thinks and writes
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The Tavily MCP server
TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"
TOOLS = ["tavily_search", "tavily_extract"]  # the two tools the lab uses

# Q2: how many search results the baseline gives the model
BASELINE_RESULTS = 3

# Q5: the domains the search may return (sent to the server, not to the model)
# YOUR TURN (Q5): list the domains you trust, for example "openrouter.ai".
TRUSTED_DOMAINS = []

# Q5: how many searches the agent may run for one claim
MAX_SEARCHES = 4

# Safety cap for every agent run (Q3–Q5): at most this many model calls per claim.
# A run that hits it stops and is recorded as cant_tell. Don't raise it to make a run finish.
MAX_MODEL_CALLS = 12

# How many claims run at the same time in a batch
CONCURRENCY = 4
