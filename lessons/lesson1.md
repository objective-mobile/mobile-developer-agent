# Task: Research Agent

Build an agent that receives a question from the user, independently searches for information using a set of tools, collects findings, and generates a structured Mark

> **Note:** The project contains example code as a starting point. Feel free to modify, extend, and adapt it to your needs.

**Example interaction:**
```
User: "Compare three approaches to building RAG: naive, sentence-window, and parent-child retr"

Agent:
  Thought: Need to find information about each approach separ
  → web_search("naive RAG pipeline approach")
  → web_search("sentence window retrieval RAG")
  → web_search("parent child retrieval RAG")
  → read_url("https://...article comparing approaches...")
  → web_search("RAG approaches comparison tradeoffs 2024")

Final Answer: [structured Markdown report comparing the three approaches]

Output: → research_report.md
```

---

### Secrets and Environment Vari

**NEVER commit API keys to the repository!**

- Store all secrets (API keys, tokens) in a `.env` file
- The `.env` file is already added to `.gitignore` — it won't endup in git
- Configuration is managed via Pydantic Settings (`config.py`), which automatically reads variables from the `.env` file
- Environment variable template is in `.env.example`

---

### What to Implement

#### 1. Agent Tools

Define and implement at least **3 tools** with tool calling:

##### `web_search(query: str) -> list[dict]`

Internet search. Use the [`ddgs`](https://pypi.org/project/ddgs/) library. It's free and doesn't require an API key.

What DuckDuckGo returns: **a list of results**, where each contains `title` (page title), `href` (URL), and `body` (snippet — 1-2 sentences from the page). This is **not the full page text**, just a short fragment like on a Google results page. This is enough for the agent to understand which pages are relevant, but not enough for deep analysis — that's what `read_url` ir.

Example of what DuckDuckGo returns:
```python
from ddgs import DDGS

results = DDGS().text("LangChain vs LlamaIndex RAG", max_results=5)
# [
#   {
#     "title": "LangChain vs LlamaIndex: A Detailed Comparison",
#     "href": "https://example.com/article",
#     "body": "LangChain focuses on composable chains while LlamaIndex specializes in..."
#   },
#   ...
# ]
```

**What's expected from your implementat*
- A wrapper around `DDGS().text()` formatted as a tool with JSON Schema (description, parametes)
- The `max_results` parameter can be fixed (e.g., 5) or made a tool parameter
- Return results in a format convenient for the LLM — e.g., a list with `title`, `url`, `snippet`

##### `read_url(url: str) -> str`

Fetching the **full text** from a page by URL. This is needed because `web_search` only returns snippets — short fragments. When the agent finds a relevant page through search, it can read it fully via `read_url` to get details.

Recommended libraries (your choice):
- [`trafilatura`](https://pypi.org/project/trafilatura/) — extracts main text from a page, ignoring menus, ads, footers
- `httpx` + `readability-lxml` — a more manual approach, but also works
- Simple `httpx.get()` + `BeautifulSoup` — minimal option

Example with `trafilatura:
```python
import trafilatura

downloaded = trafilatura.fetch_url("https://example.com/article")
text = trafilatura.extract(downloaded)
# "LangChain is a framework for building LLM-powered applications..."
# (full article text, without HTML, menus, ads)
```

**What's expected from your implementat:**
- A tool that accepts a URL and returns the pageext
- **Result truncation** — full page text can be 20,000+ characters, which will fill the context window. Truncate to a reasonable limit (e.g., first 5,000–10,000 characters). This is context engineering in practice
- Error handling: invalid URL, timeout, page unavailable — return a clear error message, not a crash

##### `write_report(filename: str, content: str) -> str`

Saves the final Markdown report to a file.

**What's expected from your implementation:**
- Accepts a filename and report text (Markdown)
- Saves to a file in the `output/` directory (or another of your choi)
- Returns a confirmation with the full path to th

This is the simplest tool — essentially a wrapper around `open(path, 'w').write(content)`. But it's needed so the agent can **itself** save the result, rather than relying on external code.

##### Additional tools (optional)

You can add other tools if you find them useful — for example, `calculate`, `read_file`, `list_files`. Additional tools are not required, but can improve your grade if they are meaningful and integrated into the agent loop.

#### 2. Agent Loop

Implement the agent using **LangChain**.

Use `create_react_agent` (or `create_agent`) with the `@tool` decorator to define tools. LangChain implements the ReAct cycle itself — you need to correctly describe the tools, configure the agent, and connect the model.

Model — any of your choice: `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, etc. LangChain allows changing the provider in one line — take advantage of this.

**Agent requirements:**
- The agent runs from the terminal (`python main.py`) and works in interactive mode — the user enters a question, gets an answer, and can continue the dialogue
- The agent **supports coherent dialogue** — remembers previous messages within a session. For example, if the user first asked to research a topic, then says "now compare this with X", the agent understands the context. Use `MemorySaver` (checkpointer) or a similar mechanism
- The agent **decides itself** which tools to call and in what order — you don't hardcode the order
- Support for **multi-step**: minimum 3–5 tool calls per request
- **Step limit** (max_iterations) — so the agent doesn't loop
- **Error handling**: if a tool returns an error — the agent receives it in context and reacts (retries with different parameters or continues without thatsult)

#### 3. Context Engineering

Implement **tool result truncation** — if the result of `web_search` or `read_url` is too large, truncate to N characters before returning to context. For example, full page text from `read_url` can be 20,000+ characters — return only the first 5,000–10,000.

#### 4. Prompts and Configuration

- System prompt and all prompt templates must be **extracted to a separate file** (`config.py` or `prompts.py`), not hardcoded in the agent logic
- System prompt must clearly describe the agent's role, available tools, and research strategy

#### 5. Environment and Documentation

- **Dependencies**: the project must contain a dependency file — `requirements.txt`, `pyproject.toml`, `Pipfile`, or equivalent — so the environment can be reproduced with one command (e.g., `pip install -r requirements.txt`). Specify exact library versions
- **Minimum library versions:**
  - `langchain >= 1.2.0` — https://pypi.org/project/langchain/
  - `ddgs >= 7.0` — https://pypi.org/project/ddgs/
  - `trafilatura >= 2.0.0` — https://pypi.org/project/trafilatura/
- **README.md**: how to run, what dependencies to install, which API key is needed, brief architecture descon
- **Example output**: save one generated report in `example_output/` so the result is visible

---

### Project Structure

```
research-agent/
├── main.py              # Entry point — interactive REPL loop
├── agent.py             # Agent setup (LLM, tools, memory, create_react_agent)
├── tools.py             # Tool definitions and implementations
├── config.py            # System prompt, settings, constants
├── requirements.txt
├── example_output/
│   └── report.md        # Example generated report
└── README.md            # Setup instructions, architecture overview
```
