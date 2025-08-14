# main.py  — Firecrawl MCP + Kimi-K2 (Windows-friendly)

import os, sys, shutil, asyncio
from typing import List, Any
from dotenv import load_dotenv
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


path_for_env_file = Path(__file__).resolve().parent          # folder containing this .py file
ENV_FILE = path_for_env_file / "keys.env"                    # .../your_script_dir/keys.env

# Load it (returns True if found and loaded)
load_dotenv(dotenv_path=ENV_FILE, override=False)


# ------------------------------------------------------------
# Utility to retrieve environment variables with fallback names
# and optional "required" enforcement.
# ------------------------------------------------------------
def _get_env(*names: str, required: bool = False) -> str | None:
    """Return the first non-empty env var among 'names'.

    Args:
        *names: Candidate environment variable names to check, in order.
                The first one that exists and is non-empty will be returned.
        required: If True, raise a RuntimeError when none of the names are set.

    Returns:
        The string value of the first found environment variable, or None
        if not found and 'required' is False.

    Raises:
        RuntimeError: If 'required' is True and none of the provided names
                      are set in the environment or are empty.
    """
    for n in names:
        v = os.getenv(n)  # Look up env var 'n'; returns None if missing
        if v:             # Treat empty string as "missing"
            return v
    if required:
        # Compose a clear message listing all names we tried
        raise RuntimeError(f"Missing required env var (tried: {', '.join(names)})")
    return None

# ------------------------------------------------------------
# Normalize / read configuration from environment.
# This supports both NEW_SNAKE_CASE and older Mixed_Case names,
# so you can migrate without breaking current setups.
# ------------------------------------------------------------
FIRECRAWL_API_KEY = _get_env("FIRECRAWL_API_KEY", "FIRECRAWL_API_Key", required=True)
KIMI_BASE         = _get_env("KIMI_K2_HF_BASE", "Kimi_K2_HF_Base", required=True)
KIMI_TOKEN        = _get_env("KIMI_K2_HF_TOKEN", "Kimi_K2_HF_Token", required=True)
KIMI_MODEL        = _get_env("KIMI_K2_HF_MODEL", "Kimi_K2_HF_Model") or \
                    "moonshotai/Kimi-K2-Instruct:fireworks-ai"

# verify if everything was loaded properly
# print(FIRECRAWL_API_KEY)
# print(KIMI_BASE)
# print(KIMI_TOKEN)
# print(KIMI_MODEL)


# ------------------------------------------------------------
# Minimal helper to find an executable on Windows, trying
# a list of candidate names (e.g., 'npx.cmd' and 'npx').
# On Unix, shutil.which('npx') is usually enough.
# ------------------------------------------------------------
def _which_windows(cmds: List[str]) -> str | None:
    for c in cmds:
        p = shutil.which(c)  # Returns absolute path if found on PATH
        if p:
            return p
    return None

class FirecrawlAgent:
    """High-level wrapper that:
       1) Validates prerequisites (env & 'npx').
       2) Configures and launches the Firecrawl MCP server (over stdio).
       3) Initializes an LLM (Kimi-K2) via OpenAI-compatible client.
       4) Runs an interactive REPL loop that delegates to a ReAct agent
          with MCP tools loaded from the Firecrawl server.
    """

    def __init__(self):
        # Will hold the ChatOpenAI model instance; set in _init_kimi_k2_model
        self.model: ChatOpenAI | None = None

        # Will hold StdioServerParameters to spawn Firecrawl MCP via npx
        self.server_params: StdioServerParameters | None = None

        # Validate environment + dependencies early to fail fast with a clear error
        self._Loading_verifying_requirements()

        # Prepare the MCP stdio server params for Firecrawl (does NOT run yet)
        self._init_firecrawl_mcp_server()

        # Initialize the OpenAI-compatible client (Kimi-K2) for the agent
        self._init_kimi_k2_model()

    # ---------- Preflight ----------
    def _Loading_verifying_requirements(self) -> None:
        """Collect and report any misconfigurations before doing real work."""
        problems: List[str] = []

        # Ensure required env vars are available (already "normalized" above)
        if not FIRECRAWL_API_KEY: problems.append("Missing FIRECRAWL_API_KEY")
        if not KIMI_BASE:         problems.append("Missing KIMI_K2_HF_BASE")
        if not KIMI_TOKEN:        problems.append("Missing KIMI_K2_HF_TOKEN")

        # Verify 'npx' exists on PATH. Firecrawl MCP is distributed via npx.
        if os.name == "nt":
            # On Windows, npx may appear as 'npx.cmd' or 'npx'
            if not _which_windows(["npx.cmd", "npx"]):
                problems.append("npx not found on PATH (install Node.js LTS).")
        else:
            # On POSIX systems, a single check is sufficient
            if not shutil.which("npx"):
                problems.append("npx not found on PATH (install Node.js LTS).")

        # If we discovered any issues, fail early with a consolidated message
        if problems:
            raise RuntimeError("Preflight failed:\n - " + "\n - ".join(problems))

    # ---------- MCP server (Firecrawl) ----------
    def _init_firecrawl_mcp_server(self) -> None:
        """Define how to spawn the Firecrawl MCP server via npx (stdio transport).

        Notes:
            - We *do not* start the process here; we only prepare parameters.
            - The MCP protocol runs over stdio in this setup.
            - '-y' auto-confirms npm prompts (important for non-interactive shells,
              especially on Windows where 'npx' may prompt on first run).
        """
        if os.name == "nt":
            # Resolve the actual, working path to npx on Windows
            npx_path = _which_windows(["npx.cmd", "npx"])
            if not npx_path:
                raise RuntimeError("Cannot find 'npx' on PATH.")
            command = npx_path
        else:
            # Use the resolved absolute path if available, else fallback to 'npx'
            command = shutil.which("npx") or "npx"

        # Configure the stdio server parameters.
        # Firecrawl MCP is an npm package runnable via 'npx firecrawl-mcp'.
        self.server_params = StdioServerParameters(
            command=command,
            args=["-y", "firecrawl-mcp"],          # '-y' avoids interactive confirmations
            env={"FIRECRAWL_API_KEY": FIRECRAWL_API_KEY},  # provide API key to the server
        )

    # ---------- LLM (Kimi-K2) ----------
    def _init_kimi_k2_model(self) -> None:
        """Instantiate the OpenAI-compatible chat model client.

        Important:
            - 'base_url' points to your OpenAI-compatible endpoint that proxies Kimi-K2.
            - 'api_key' authenticates with that endpoint.
            - 'temperature=0' for deterministic, tool-friendly outputs (good for agents).
        """
        self.model = ChatOpenAI(
            model=KIMI_MODEL,    # e.g., "moonshotai/Kimi-K2-Instruct:fireworks-ai"
            temperature=0,       # low randomness; better for reliable tool use
            api_key=KIMI_TOKEN,  # auth for the OpenAI-compatible server
            base_url=KIMI_BASE,  # host where the OpenAI-compatible API lives
        )

    # ---------- Main loop ----------
    async def main(self) -> None:
        """Run an interactive REPL that uses a ReAct agent with MCP tools.

        Flow:
            1) Launch/connect to the Firecrawl MCP server over stdio.
            2) Create a ClientSession for the MCP protocol and initialize it.
            3) Load available MCP tools (Firecrawl provides web scraping / crawling).
            4) Build a ReAct-style agent with the LLM and these tools.
            5) Enter a user REPL that streams user messages into the agent and prints replies.

        Exiting:
            - Type 'quit' or 'exit' to break the loop cleanly.
        """
        # stdio_client(...) manages the lifecycle of the stdio process (npx firecrawl-mcp)
        async with stdio_client(self.server_params) as (read, write):
            # ClientSession wires up the MCP protocol over the stdio pipes
            async with ClientSession(read, write) as session:
                await session.initialize()  # Handshake + capabilities exchange

                # Dynamically discover tools exposed by the Firecrawl MCP server
                tools = await load_mcp_tools(session)

                # Build a LangGraph/LangChain ReAct agent that can call those tools
                agent = create_react_agent(self.model, tools)

                # Informational: show which tool names are available to the agent
                print("Available Tools:", ", ".join(getattr(t, "name", "tool") for t in tools))
                print("-" * 60)

                # Seed the conversation with a system instruction guiding tool usage
                messages: List[Any] = [
                    SystemMessage(
                        content=(
                            "You can scrape/crawl/extract data using Firecrawl MCP tools. "
                            "Decide when to use tools; think step by step."
                        )
                    )
                ]

                # ---------------------------
                # Simple terminal-based REPL
                # ---------------------------
                while True:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() in {"quit", "exit"}:
                        print("Goodbye!")
                        break

                    # Cap the message length to avoid overly long payloads
                    messages.append(HumanMessage(content=user_input[:175000]))

                    try:
                        # Invoke the agent asynchronously with the running conversation
                        result = await agent.ainvoke({"messages": messages})

                        # The agent's return shape can vary a bit; try common paths
                        ai_text = None
                        if isinstance(result, dict) and "messages" in result:
                            final_msgs = result["messages"]
                            if final_msgs:
                                ai_text = getattr(final_msgs[-1], "content", None)
                        if not ai_text and isinstance(result, dict):
                            # Fallback fields some agent executors use
                            ai_text = result.get("output") or result.get("final")

                        print("\nAgent:", ai_text or "[No content returned]")
                    except Exception as e:
                        # Catch and print any runtime errors (tool errors, network issues, etc.)
                        print("Error:", repr(e))


# ---------- Entrypoint ----------
if __name__ == "__main__":
    try:
        # Run the async REPL loop; asyncio.run handles event loop creation/teardown
        asyncio.run(FirecrawlAgent().main())
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        print("\nInterrupted by user.")
    except Exception as e:
        # Any unhandled fatal error bubbles up here; exit with non-zero code
        print("\nFATAL:", e)
        sys.exit(1)

