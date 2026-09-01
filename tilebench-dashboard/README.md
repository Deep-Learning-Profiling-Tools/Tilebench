# TileBench Dashboard

TileLang against Triton, cuTile and torch across 42 GPU kernels on a B200 (sm_100),
with NCU profiles, per-kernel source, engineer notes and a constrained performance query agent.

Built API-first: the page renders from the same routes an external client would call,
so the compiler-source, NCU and agent surfaces grow without reshaping the app.

See [AGENTS.md](AGENTS.md) for deployment state, data provenance, NCU report
quality and staleness, and the agent-pipeline design notes.

## Run

```bash
npm install
npm run dev          # http://localhost:3000
```

## API

| route | purpose |
|---|---|
| `GET /api/health` | dataset counts, notes-store kind, whether the agent is configured |
| `GET /api/operators` | all rows + pooled headline. `?tier=`, `?target=1`, `?sort=autotune\|default\|op` |
| `GET /api/operators/:op` | one kernel: results, notes, profile presence, source listing |
| `GET /api/notes` | every note, grouped by kernel |
| `GET/POST/DELETE /api/notes/:op` | read, append (`{body}`), remove (`?ts=`) |
| `GET /api/profile` | NCU run manifest across all profiled kernels |
| `GET /api/profile/:op` | the NCU write-up plus that kernel's `.ncu-rep` index |
| `GET /api/source/:op` | implementation files available for a kernel |
| `GET /api/source/:op/:file` | one file as `text/plain` |
| `POST /api/agent` | `{messages?, question?, contextOp?, apiKey?}` → SSE stream (`meta`, `tool`, `delta`, `done`, `error`) |

## Data

`data/` is generated from the TileBench repo and read at request time.

- `operators.json` — 42 kernels, default and autotune held apart. Backend figures are
  geomean speedup vs torch; `tl_over_triton` below 1.0 means TileLang is slower.
- `profiles/*.md` — the NCU write-ups (~100KB, shipped).
- `profiles/manifest.json` — indexes the 96 `.ncu-rep` binaries by name and size.
  The binaries themselves (85MB) stay in the TileBench repo; `repo_path` locates them.
- `source/` — all four backends' implementations per kernel (~1.3MB, shipped).
- `tilelang_compiler/` — pinned TileLang compiler source snapshot from `tile-ai/tilelang`
  commit `056ccfc267f03498d8dcc79f6efd2aba2bae8abd`. The agent can list, read and grep
  this read-only snapshot for lowering/scheduling/codegen/autotune questions.
- `ncu_json/` — optional exported Nsight Compute reports produced offline from
  `tilebench_run/ncu_export_json.py`; the agent queries this instead of `.ncu-rep`.
  Exported JSON keeps hash/size provenance, not raw `.ncu-rep` paths.
- `ncu_source/` — optional captured generated CUDA/source files referenced by NCU JSON.

### Pooled numbers

Per-operator geomeans are pooled by case count:

```
exp( sum(n_i * ln g_i) / sum(n_i) )
```

This is exact — a case-weighted geomean of geomeans equals the geomean over the pooled
population — so headline figures never need the 4,100 individual cases in memory.

Note the two modes have different populations: default covers 2,080 cases across all 42
kernels, autotune covers 2,020 across 41 (`batched_matmul` has no autotune CSV). They are
never averaged together.

## Configuration

Copy `.env.example` to `.env.local`.

- `OPENROUTER_API_KEY` — optional server-side key for `/api/agent`. Without it, the drawer
  chat tab can still run the agent by sending a pasted OpenRouter key with the request.
  Chat history is browser-held per kernel tab and sent as `messages`; the server does not
  persist conversations.
- `TILEBENCH_AGENT_MODEL` — defaults to `openai/gpt-5.6-luna`.
- `OPENROUTER_SITE_URL` / `OPENROUTER_APP_TITLE` — optional OpenRouter attribution
  headers for the dashboard.
- `KV_REST_API_URL` / `KV_REST_API_TOKEN` — attach Vercel KV (Upstash Redis) to make
  notes durable. **Without these, notes are per-instance and reset when the serverless
  instance recycles**; the UI says so on save. Everything else is read-only and fine.

## Extending

- **More NCU runs** — drop `REPORT.md` into `data/profiles/<op>.md` and add a manifest
  entry for the human profile tab. For queryable agent metrics, export `.ncu-rep` files with
  `python tilebench_run/ncu_export_json.py` in the TileBench repo, then ship the generated
  `ncu_json/` and `ncu_source/` directories under `data/`.
- **Compiler source** — `data/source/<op>/` accepts any file; add it to
  `source/index.json` and it appears in the source tab. Generated CUDA and PTX slot in
  the same way as the Python implementations.
- **Notes durability** — implement `NotesStore` in `lib/store.ts`; no route changes.
