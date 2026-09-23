# LoRA-Harvester MCP Server

Lets Claude (Code / Desktop) drive LoRA-Harvester directly: scan datasets,
validate captions, export to kohya, run video processing / character sorting
as background jobs, and **diagnose real runtime errors**.

## Install

```bash
pip install mcp
```

Launch the server with the **Python env that has the project's heavy deps**
(cv2, torch, PyQt5) so the background jobs work. Light tools and diagnostics
work even without them.

## Register

### Option A — project scope (auto-loads in this repo)
`.mcp.json` is already in the project root. When you run `claude` from the
project directory it is picked up automatically. Approve it once when prompted.

### Option B — global (works from anywhere)
```
claude mcp add lora-harvester -- python "C:/Users/cavas/Desktop/Programlar/devam/LoRA-Harvester-main/LoRA-Harvester-main/mcp_server.py"
```

If you use a venv, point at its python, e.g.:
```
claude mcp add lora-harvester -- "C:/.../venv/Scripts/python.exe" "C:/.../mcp_server.py"
```

Verify with `/mcp` — `lora-harvester` should be connected with 17 tools.

## Tools

### Dataset (light, instant, no GPU)
| Tool | Purpose |
|------|---------|
| `scan_dataset(folder)` | Image/caption counts by concept |
| `validate_captions(folder, trigger_word, min_tags)` | Caption quality audit |
| `list_concepts(folder)` | Concept → count |
| `dataset_stats(folder)` | Resolution buckets + warnings |
| `list_upscale_models()` | Upscale model registry |
| `export_kohya(source, dest, repeats, copy, gen_toml)` | Kohya export |

### Processing (background jobs, GPU)
| Tool | Purpose |
|------|---------|
| `process_video(video_path, ...)` | Video → frames + captions → job_id |
| `sort_characters(input_folder, references, ...)` | Face-based sort → job_id |
| `download_models(upscale, florence2, ...)` | Model download → job_id |

### Job control
| Tool | Purpose |
|------|---------|
| `get_job_status(job_id)` | running/done/failed/cancelled |
| `get_job_log(job_id, tail)` | Live stdout/stderr (incl. tracebacks) |
| `list_jobs()` | All jobs |
| `cancel_job(job_id)` | Terminate a job |

### Diagnostics (detect real runtime errors)
| Tool | Purpose |
|------|---------|
| `health_check()` | Import every module, report broken imports + missing deps |
| `get_crash_log(tail)` | App UI-crash tracebacks (crash_log.txt) |
| `run_self_test()` | Run the test suite, return pass/fail summary |
| `gpu_check()` | CUDA / torch diagnostics |

## How error detection works

- **Background jobs** write combined stdout+stderr to `.mcp_jobs/<job_id>.log`.
  `get_job_log` surfaces live progress and any Python traceback from a failed run.
- **`health_check`** catches broken imports / missing dependencies instantly.
- **`get_crash_log`** surfaces UI-thread crashes the app records to `crash_log.txt`.
- **`run_self_test`** catches regressions.

Job logs persist under `.mcp_jobs/` (gitignore-able).
