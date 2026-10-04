# MCP Filesystem Agent

Extends the "Agentic Profile Matching" agent by replacing its
direct filesystem calls with a standalone **MCP (Model Context Protocol)
server**. The LangGraph agent now acts as an MCP client, talking to the
filesystem server over JSON-RPC 2.0 instead of calling file functions
directly.

## Architecture

```
matching_agent.py (LangGraph agent)
        |
        |  MultiServerMCPClient (JSON-RPC 2.0 over stdio)
        v
filesystem_mcp_server.py (MCP server, subprocess)
        |
        v
resumes/   <-- sandboxed root directory
```

- **`filesystem_mcp_server.py`** — MCP server exposing file tools
  (`read_file`, `write_file`, `list_files`, `search_in_file`,
  `watch_directory`, `batch_process`) plus resource discovery
  (`resumes://list`, `resumes://{filename}`). Sandboxed to a configurable
  root directory, with extension and max-file-size limits.
- **`matching_agent.py`** — LangGraph agent (Parse JD -> Extract
  Requirements -> Search Resumes -> Generate Report -> Final Recommendation
  -> Human Feedback loop). The two nodes that touch the filesystem
  (`parse_jd`, `search_resumes`) now call the MCP server through an MCP
  client instead of importing file functions directly.
- **`rag_engine/`** — Milestone 2 RAG pipeline (chunking, embeddings,
  ChromaDB, hybrid search). Unchanged; not part of the MCP refactor.
- **`file_assistant/`** — Milestone 1 plain-Python file tools, used only by
  `resume_rag.py` at indexing time (a local script, not part of the agent's
  runtime path).

## Prerequisites

- Python 3.10+
- Node.js (only needed to run the MCP Inspector for manual testing)
- API keys: OpenRouter (agent LLM calls) and Cohere (embeddings)

## Setup

```bash
git clone <this-repo>
cd mcp-filesystem-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENROUTER_API_KEY=your_key
export COHERE_API_KEY=your_key
```

## 1. Build the resume index

The agent's RAG search needs resumes embedded in ChromaDB. Put `.txt`
resumes in `resumes/`, then run:

```bash
python3 -m rag_engine.resume_rag
```

This creates the local ChromaDB store and prints the number of chunks
embedded.

## 2. Add a job description

Job descriptions must live inside `resumes/` as well, since that is the MCP
server's sandboxed root (see Configuration below). Example:

```bash
cp job_descriptions/jd_01.txt resumes/jd_01.txt
```

## 3. Run the MCP server standalone (optional, for manual testing)

Using the MCP Inspector:

```bash
npx -y @modelcontextprotocol/inspector python3 filesystem_mcp_server.py
```

Opens a browser UI to list tools/resources and call them directly, and to
inspect the raw JSON-RPC traffic.

Using the included test client:

```bash
python3 test_client.py
```

## 4. Run the agent

The agent launches the MCP server itself as a subprocess — you do not need
to start it separately.

```bash
python3 matching_agent.py
```

Expected flow: the agent loads the JD via MCP, extracts requirements,
searches resumes, generates per-candidate reports and final
hire/no-hire recommendations, then drops into an interactive loop where you
can type things like `compare the top 2`, `why did X rank higher`,
`more years of experience required`, or `done`.

## Configuration

The MCP server reads these environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `FS_MCP_ROOT` | `./resumes` | Sandboxed root directory — all file paths are resolved relative to this and rejected if they fall outside it |
| `FS_MCP_MAX_FILE_SIZE` | `5242880` (5 MB) | Maximum file size `read_file` will accept |
| `FS_MCP_ALLOWED_EXTENSIONS` | `.txt,.pdf,.docx` | Comma-separated list of extensions `read_file` will accept |

## Project layout

```
.
├── filesystem_mcp_server.py   # Part A: MCP server
├── matching_agent.py          # Part B: LangGraph agent, MCP client
├── test_client.py             # standalone MCP server test
├── test_scenarios.md          # test matrix for Parts A and B
├── diagrams/
│   └── mcp_workflow.mmd       # agent <-> MCP sequence diagram
├── rag_engine/
│   ├── resume_rag.py
│   └── job_matcher.py
├── file_assistant/            # Milestone 1 tools, used only by resume_rag.py
├── resumes/                   # MCP server's sandboxed root (resumes + JDs)
├── requirements.txt
└── README.md
```

## Known issues

- One candidate occasionally appears as "Unknown" in results — a
  pre-existing name-parsing issue in the Milestone 2/3 RAG pipeline,
  unrelated to the MCP refactor.
- `watch_directory` uses a snapshot-diff approach (compares the file list
  between calls) rather than a live filesystem watcher, since MCP over
  stdio is request/response rather than a persistent push channel.
