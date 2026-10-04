# MCP Filesystem Server — Test Scenarios

## Setup
- Server: `filesystem_mcp_server.py`
- Transport: stdio
- Tested with: MCP Inspector (`npx @modelcontextprotocol/inspector python3 filesystem_mcp_server.py`)
- Config: FS_MCP_ROOT=./resumes, FS_MCP_MAX_FILE_SIZE=5MB (default), FS_MCP_ALLOWED_EXTENSIONS=.txt,.pdf,.docx (default)

## Tools

### 1. list_files
| Scenario | Input | Expected | Result |
|---|---|---|---|
| List all files | `{"directory": ".", "extension": null}` | Returns resumes with metadata | ✅ Pass |
| Filter by extension | `{"directory": ".", "extension": ".txt"}` | Only .txt files returned | ✅ Pass |
| Nonexistent directory | `{"directory": "nope"}` | `{"success": false, "error": "Directory not found"}` | ✅ Pass |

### 2. read_file
| Scenario | Input | Expected | Result |
|---|---|---|---|
| Read valid .txt | `{"text_file": "resume_01.txt"}` | Full content + metadata | ✅ Pass |
| Nonexistent file | `{"text_file": "does_not_exist.txt"}` | `{"success": false, "error": "File not found"}` | ✅ Pass |
| Path traversal attempt | `{"text_file": "../../etc/passwd"}` | `{"success": false, "error": "Access denied..."}` | ✅ Pass |
| Disallowed extension (config test) | `{"text_file": "resume.pdf"}` with FS_MCP_ALLOWED_EXTENSIONS=.txt | Rejected by config | ✅ Pass |

### 3. write_file
| Scenario | Input | Expected | Result |
|---|---|---|---|
| Write new file | `{"filepath": "test_output.txt", "content": "hello from mcp"}` | File created, confirmed via list_files/read_file | ✅ Pass |

### 4. search_in_file
| Scenario | Input | Expected | Result |
|---|---|---|---|
| Keyword found | `{"filepath": "resume_01.txt", "keyword": "Python"}` | Match positions + context | ✅ Pass |

### 5. watch_directory
| Scenario | Input | Expected | Result |
|---|---|---|---|
| First call (baseline) | `{"directory": "."}` | `new_files: []`, baseline recorded | ✅ Pass |
| After adding a file | `{"directory": "."}` (2nd call) | New file listed in `new_files` | ✅ Pass |

### 6. batch_process
| Scenario | Input | Expected | Result |
|---|---|---|---|
| Mixed valid/invalid files | `{"filenames": ["resume_01.txt", "resume_02.txt", "does_not_exist.txt"]}` | `succeeded: 2, failed: 1`, no crash | ✅ Pass |

## Resources
| Scenario | URI | Expected | Result |
|---|---|---|---|
| List resources | `resumes://list` | Newline-separated filenames | ✅ Pass |
| Read a resource | `resumes://resume_01.txt` | File content | ✅ Pass |

## Bugs found & fixed during testing
1. **Path traversal not blocked** — initial `validate_path` using `.parents` didn't catch `../../etc/passwd`. Fixed using `os.path.commonpath`.
2. **Crash on missing file in batch_process** — `read_file` called `os.path.getsize`/`getmtime` before checking file existence, raising an uncaught `FileNotFoundError`. Fixed by adding an explicit `os.path.isfile` check before metadata calls.
3. **datetime not JSON-serializable** — original fs_tools returned raw `datetime` objects; converted to `.isoformat()` strings for MCP/JSON-RPC compatibility.
4. **Misnamed test file** — `resume_05txt` (missing dot) silently excluded by extension filtering; renamed.


## Part B: Agent ↔ MCP Integration

| Scenario | Description | Result |
|---|---|---|
| JD loaded via MCP | parse_jd reads jd_01.txt through read_file tool | ✅ Pass — 725 chars loaded |
| Requirements extracted | LLM parses must-have/nice-to-have/years from real JD | ✅ Pass |
| Temp JD write via MCP | search_resumes writes temp_jd.txt through write_file tool | ✅ Pass |
| RAG search on real data | match_resumes_to_job returns candidates from ChromaDB | ✅ Pass — 2 candidates found |
| Full pipeline run | parse_jd → extract_requirements → search_resumes → generate_report → final_recommendation | ✅ Pass, end to end |
| human_feedback loop | "finish" classified correctly, graph routes to end | ✅ Pass |

## Bugs found during Part B integration
1. Tool result shape varied (string vs list of content blocks) depending on adapter version — handled with `parse_mcp_result()` normalizer.
2. `final_recommendation` retry block was unconditionally re-running — fixed indentation bug carried over from Milestone 3.
3. MCP server's `ALLOWED_ROOT` resolves relative to the subprocess's cwd — JD and resume files must live inside `resumes/`, not `job_descriptions/`, since the server only has one configured root.