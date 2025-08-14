# WebSearch_LLM_AiAgent_Using_FireCrawl_LangGraph_Kimi_K2_HF

# README — Firecrawl MCP + Kimi‑K2 Agent and FastAPI LLM Service

This project gives you two complementary pieces:

1) A **FastAPI microservice** that exposes a simple `/chat` endpoint backed by a Kimi‑K2 LLM via an OpenAI‑compatible API.  
2) A **terminal REPL agent** that launches the **Firecrawl MCP** server (via `npx`) and uses a ReAct‑style agent 
(LangGraph + LangChain) to decide when to call Firecrawl tools for crawling/extraction before answering with the Kimi‑K2 model.

Both parts share the same credentials model (via `keys.env` / environment variables) and are designed to run on Windows and macOS/Linux.

---

## Table of Contents

- [Architecture & Flow](#architecture--flow)
- [Why These Technologies](#why-these-technologies)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Installation](#installation)
- [Running the FastAPI Service](#running-the-fastapi-service)
- [Using the REST API](#using-the-rest-api)
- [Running the Firecrawl Agent (REPL)](#running-the-firecrawl-agent-repl)
- [How the REPL Agent Works (Step-by-Step)](#how-the-repl-agent-works-step-by-step)
- [Extending the System](#extending-the-system)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [Appendix — Key Code References](#appendix--key-code-references)

---

## Architecture & Flow

### 1) FastAPI LLM Service (stateless HTTP)

- **Request**: `/chat` with a `prompt` (GET query or JSON body).
- **Processing**: The app creates a chat completion using your **OpenAI‑compatible** base URL, API key, and Kimi‑K2 model name.
- **Response**: Returns the model’s text back to the client.

### 2) Firecrawl MCP + ReAct Agent (interactive terminal)

- **Start**: Program validates env vars and locates `npx` (Windows: `npx.cmd`/`npx`).  
- **Tooling**: It spawns **Firecrawl MCP** over stdio via `npx -y firecrawl-mcp` and loads those MCP tools into LangGraph/LangChain.  
- **Reasoning**: A ReAct agent (`create_react_agent`) uses Kimi‑K2 (OpenAI‑compatible) to plan → decide → call Firecrawl tools when web crawling is helpful → synthesize a final answer.  
- **REPL**: You type a query, the agent may call tools, then replies. Type `exit`/`quit` to leave.

---

## Why These Technologies

- **FastAPI + Pydantic**: Lightweight, fast, type‑safe HTTP API to expose LLMs quickly.
- **OpenAI Python client**: Works against **OpenAI‑compatible** endpoints; you only need `base_url` + `api_key` for Kimi‑K2.
- **Kimi‑K2 LLM**: General‑purpose chat model; OpenAI‑compatible surface makes it drop‑in for LangChain & the client.
- **Model Context Protocol (MCP)**: Standard interface for tools (here: Firecrawl) so the model can call them reliably.
- **Firecrawl MCP**: Production‑grade crawling/extraction pipeline exposed as MCP tools.
- **LangGraph + LangChain**: Orchestrates the **ReAct** loop—chain thoughts, pick tools, observe results, answer.

The net effect: **stateless API for simple use cases** + **agent with web tools for complex, retrieval‑heavy questions**.

---

## Project Layout

A suggested structure based on the provided files:

```
project-root/
├─ keys.env
├─ requirements              # Python deps (pip) — install with `pip install -r requirements`
├─ LLM_Model_HuggingFace.py  # FastAPI service (module name used by uvicorn)
├─ Web_SearchAgent.py        # Firecrawl MCP + ReAct Agent (entry script)
└─ README.md
```

> The FastAPI service initializes an OpenAI‑compatible client using env vars and exposes `/chat` via FastAPI.  
> The agent script preflights env, finds `npx`, starts Firecrawl MCP, loads tools, builds a ReAct agent, and runs a terminal REPL.

---

## Prerequisites

- **Python** ≥ 3.10
- **Node.js LTS** (which provides **`npx`**)  
  - Windows will typically use `npx.cmd`; the agent code handles both.
- Network access to your **OpenAI‑compatible** Kimi‑K2 endpoint.

---

## Configuration

Create a `keys.env` at the project root with:

```env
# Kimi‑K2 (OpenAI‑compatible) endpoint & credentials
KIMI_K2_HF_BASE=https://YOUR-OPENAI-COMPATIBLE-HOST/v1
KIMI_K2_HF_TOKEN=sk-xxxxxxxxxxxxxxxxxxxxxxxx
# Optional: override; else default in code is fine
KIMI_K2_HF_MODEL=moonshotai/Kimi-K2-Instruct:fireworks-ai

# Firecrawl MCP
FIRECRAWL_API_KEY=fc_live_xxxxxxxxxxxxxxxxxxxxxxxxx
```

**Notes**

- Make sure your Python files point to the right `.env` location. A simple and robust pattern near the top of each file is:
  ```python
  import os
  from pathlib import Path
  from dotenv import load_dotenv

  HERE = Path(__file__).resolve().parent
  load_dotenv(HERE / "keys.env")  # falls back to process env if not found
  ```
- You can always override with shell env variables in CI/production.

---

## Installation

```bash
# from project-root
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements
```

> If you don’t have a complete `requirements` file, install the visible deps directly:
>
> ```bash
> pip install fastapi uvicorn python-dotenv openai >            langchain langgraph langchain-openai >            mcp langchain-mcp-adapters
> ```
>
> Also install **Node.js LTS** from nodejs.org (ensures `npx` works).

---

## Running the FastAPI Service

Assuming the file is named `LLM_Model_HuggingFace.py` and defines `app`:

```bash
uvicorn LLM_Model_HuggingFace:app --host 0.0.0.0 --port 8000 --reload
```

At startup it will:

- Load env (per the path in the file).  
- Build `OpenAI(base_url=..., api_key=...)`.  
- Serve `POST /chat` and `GET /chat?prompt=...`.

---

## Using the REST API

### Quick via `curl`

```bash
# GET
curl "http://127.0.0.1:8000/chat?prompt=Tell%20me%20a%20joke"

# POST (JSON)
curl -X POST "http://127.0.0.1:8000/chat"   -H "Content-Type: application/json"   -d '{"prompt": "Summarize the advantages of MCP."}'
```

### Python snippet

```python
import requests
resp = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"prompt": "What is the capital of France?"}
)
print(resp.json())
```

Under the hood, the service calls `client.chat.completions.create(model=..., messages=[...])` and returns the text.

---

## Running the Firecrawl Agent (REPL)

```bash
# Windows/macOS/Linux
python Web_SearchAgent.py
```

What happens:

1) **Preflight**: Checks required env vars and ensures `npx` is available on your PATH (Windows: `npx` or `npx.cmd`).  
2) **Spawn MCP**: Prepares `StdioServerParameters` to run `npx -y firecrawl-mcp` with `FIRECRAWL_API_KEY` in env.  
3) **Load Tools**: `load_mcp_tools(session)` discovers Firecrawl tools.  
4) **Create Agent**: `create_react_agent(model, tools)` wires Kimi‑K2 to tools.  
5) **REPL**: Type a prompt; the agent may call tools; get an answer. Type `quit`/`exit` to leave.

---

## How the REPL Agent Works (Step-by-Step)

1. **Env load & normalization**  
   - Accepts standard names like `KIMI_K2_HF_BASE`, `KIMI_K2_HF_TOKEN`, `KIMI_K2_HF_MODEL`; requires `FIRECRAWL_API_KEY` as well.  
   - Fails early with a consolidated error list if anything is missing.

2. **Find `npx`**  
   - On Windows tries `npx.cmd` then `npx`; on POSIX uses `shutil.which("npx")`.

3. **Prepare MCP server**  
   - `StdioServerParameters(command=<npx>, args=["-y","firecrawl-mcp"], env={"FIRECRAWL_API_KEY": ...})`  
   - `-y` suppresses interactive npm prompts (important on first run & CI).

4. **Initialize LLM**  
   - `ChatOpenAI(model=..., temperature=0, api_key=..., base_url=...)` — deterministic outputs help tool use.

5. **MCP session & tools**  
   - `stdio_client(...)` ↔ `ClientSession(...).initialize()` handshakes MCP.  
   - `load_mcp_tools(session)` exposes Firecrawl endpoints (crawl/extract).

6. **ReAct agent loop**  
   - Seeds a system message (“use Firecrawl when helpful”).  
   - For each user turn: decide → (optionally) call tool → observe → answer.  
   - Prints final text; catches exceptions to avoid crashing the REPL.

---

## Extending the System

- **Add more MCP tools**: Any MCP‑compatible server can be launched similarly; load its tools alongside Firecrawl.  
- **Swap the LLM**: Keep the OpenAI‑compatible client but change `base_url`/`model` (and token). Works for both the API and the agent.  
- **Expose more endpoints**: On the FastAPI side, add routes for streaming, tool‑augmented answers, or RAG pipelines.

---

## Troubleshooting

**`npx not found on PATH` (Windows/macOS/Linux)**  
- Install Node.js LTS; reopen terminal so PATH updates.  
- On Windows, verify `where npx` finds `npx.cmd`.

**`Preflight failed: Missing <ENV>`**  
- Ensure `keys.env` has all required entries and is being loaded from the correct path.  
- You can also export env vars in your shell to override.

**npm error `ETARGET` / “No matching version found for firecrawl-mcp@X.Y.Z”**  
- Remove version pins; run plain `npx firecrawl-mcp` (the agent already does this).  
- Clear cache (`npm cache clean --force`) and retry; ensure an open network (no restrictive proxy).  
- Update Node.js/npm.

**FastAPI returns 500**  
- Check `KIMI_K2_HF_BASE`, `KIMI_K2_HF_TOKEN`, and `KIMI_K2_HF_MODEL`.  
- Verify your OpenAI‑compatible endpoint is reachable and supports `chat.completions`.

**Agent prints “No content returned”**  
- Depending on agent library versions, the output field name might differ. Add logging and inspect the result dict if needed.

---

## Security Notes

- **Never commit `keys.env`** or tokens to version control.  
- Prefer **per‑host env vars** in production.  
- Rotate keys periodically.  
- Consider rate limiting, request validation, and auth in front of FastAPI for public deployments.

---

## Appendix — Key Code References

- **FastAPI LLM service**: `LLM_Model_HuggingFace.py` — loads env, builds `OpenAI(base_url, api_key)`, exposes `POST/GET /chat`, and returns `completion.choices[0].message.content`.  
- **Agent entry**: `Web_SearchAgent.py` — preflight env + `npx`, start MCP (`npx -y firecrawl-mcp`), load MCP tools, create `ChatOpenAI(...)`, build ReAct agent, run REPL.

---


Thank you 
If you find this interesting or see if there are any issues or improvements you see please let me know.