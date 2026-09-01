# AGENTS.md

Handoff notes for whoever picks this up next. The README explains how the app works;
this file records state, provenance and findings that aren't recoverable from the code.

Last updated: 2026-08-30.

## What we're building

**Goal: a dashboard where someone can ask an agent why a kernel performs the way it does,
and get an answer grounded in real evidence** — the benchmark numbers, the NCU profile of
that exact kernel, and the compiler source that produced it.

The board itself is the starting point, not the product. The three things being grown into
it, in the user's own framing:

1. **Compiler source code** — the implementations behind each kernel, eventually including
   TileLang's generated CUDA/PTX, not just the Python DSL.
2. **NCU reports per kernel** — exported JSON the agent can query, rather than prose write-ups
   a human has to read.
3. **An agent users query about performance** — with tool access to all of the above,
   able to grep and search the corpus rather than being handed a fixed context blob.

That is why the app is API-first: every surface the agent will need is already an HTTP
route, so adding a tool means wiring a tool to a route, not reshaping the app.

### Where it stands

| | state |
|---|---|
| Board + 42 kernels | **done**, live |
| API routes (11) | **done**, live |
| Source browsing (225 files, 4 backends) | **done**, live |
| NCU write-ups (7 kernels, prose) | **done**, live |
| Notes | done, but non-durable |
| Agent | OpenRouter tool loop built; drawer chat tab accepts pasted OpenRouter keys or prod `OPENROUTER_API_KEY` |
| NCU as queryable data | RTX 4060 smoke payload deployed; B200 replacement still needed |
| TileLang compiler source | pinned `tile-ai/tilelang` snapshot deployed for agent list/read/grep |
| Compiler-generated source (CUDA/PTX) | partial: RTX 4060 smoke generated CUDA deployed |

### Research context

The underlying data is **unpublished** B200 benchmark results and NCU findings tied to an
EMNLP paper draft. This is why access control kept coming up, and why the agent's system
prompt is strict about not inventing numbers and preserving the write-ups'
established / not-established distinction. Treat the numbers as pre-publication.

### Known gap driving current work

NCU coverage is the weak point: **7 of 42 kernels profiled**, and only 2 `(kernel, backend)`
pairs have current TileLang captures. An agent asked about the other 35 kernels has no
profile to reason from. Options discussed: re-capture on B200, or capture locally on an
RTX 4060 — noting 4060 is cc 8.9 (Ada, no TMEM, no `wgmma`), so local captures can support
tooling development but cannot back B200 claims in the paper.

Current queryable NCU payload is a **local RTX 4060 smoke corpus** for
`vector_add/fp32` with Triton and TileLang reports. It exists to validate the
agent/tool/export loop and should be replaced by B200 captures for paper claims.

## What this is

A Next.js 16 dashboard over the TileBench results: TileLang vs Triton, cuTile and torch
across 42 GPU kernels on a B200 (sm_100). Built API-first so NCU reports, compiler source
and a query agent can grow into it without reshaping the app.

Sibling repo, referenced throughout: `../Tilebench` (branch
`aaroosh/feature/tilelang-all-implemented`).

## Live deployment

| URL | state |
|---|---|
| https://tilebench-dashboard.vercel.app | production, **public**, serves the real board |

Vercel project `prj_NeT2he9Dz9xI2i1SGbyomgG5W9kq`, team `team_bwydauL1Uxl0TcElxKML9LRZ`
(`volleydoodle's projects`, **Hobby plan**). Deployed via CLI, not git-linked.

**Access is deliberately public.** SSO was disabled by explicit user decision on
2026-08-30. This publishes unpublished B200 benchmark data, 7 NCU write-ups, 225 source
files and the engineer notes. Do not change this either direction without asking.

Plan constraint, established empirically — on Hobby, `ssoProtection` maxes out at
`all_except_custom_domains`, so the production alias **cannot** be protected. Attempts
returned `428 invalid_sso_protection` and `428 invalid_password_protection`; `pause_project`
returned 400. Protecting the nice URL requires Pro.

### Deploy gotcha

`vercel.json` pins `framework: "nextjs"`. **Do not remove it.** A plain-HTML deploy to
this project once reset the detected framework to `null`, after which every Next build
succeeded and then failed on `No Output Directory named "public" found`. The pin overrides
the project setting.

## Unconfigured

- `OPENROUTER_API_KEY` is **not set in production**. The agent still works when the browser
  sends a pasted OpenRouter key with the request.
- Notes are **non-durable**: `notes_store: {kind: "memory", durable: false}`. A note
  survives the tab, not a cold start. Set `KV_REST_API_URL` + `KV_REST_API_TOKEN` to fix.

## Data provenance

`data/operators.json` was reconstructed from the published artifact's `rowsdata` array
(`[triton, cutile, tilelang, tl_over_triton, tl_over_cutile, cases]`) plus two later
published updates, applied algebraically. Pooled results reproduce the board exactly:
default 2.084 / 1.717 / 1.919, tl/tri 0.920; autotune 2.385 / 1.984 / 2.279, tl/tri 0.956.

**Known error in the published artifact, not fixed:** its default panel shows 2,020 cases.
The true figure is 2,080 — `batched_matmul` contributes a default with no autotune. Ratios
are unaffected; only the case-count label is wrong. `data/` and README carry 2,080.

## NCU reports — read this before building on them

96 `.ncu-rep` files, all added in one commit (`f120be3d`, 2026-07-27), living in
`../Tilebench/profile/b200_autotuned_postmerge/`. Not shipped with the app (88.8 MB);
`data/profiles/manifest.json` indexes them by `repo_path`.

**Quality: metrics are good, attribution is absent.**

- 2,481 metric columns on `full_` captures, 821 on standard, 748 on the matmul `f_` set.
  Full-section capture, not `--set basic`. Real B200 (cc 10.0).
- NCU **rule engine output is embedded** (`SOLBottleneck`, `CPIStall`,
  `IssueSlotUtilization`, with estimated speedups). Highest-value content for an agent —
  it is already diagnosis, not raw counters.
- **0 of 96 reports have SASS→source attribution.** Verified across all 96: every
  instruction row is `???`, PTX view empty. Captured without `-lineinfo`, so the
  address→line map does not exist. `SourceCounters` sections are present and carry
  per-address stall counts, but nothing maps them to a line.

The `source_` / `src_` / `full_` prefixes reflect *intent* to capture source, not success.
Filenames use 16 different prefixes and are per-investigation — **use the manifest, never
parse filenames**.

To get line attribution you would need `-lineinfo` at build **and** `--import-source yes`
at capture. Note the map would land on TileLang's *generated CUDA*, not the Python DSL;
whether TileLang preserves line provenance through lowering is unverified.

Multi-kernel captures are useful compensation: `radix_sort` decomposes into `main_kernel`,
`count_ones_per_block`, `compute_prefix_sums_per_block`, `compute_prefix_sums_bb`, so
per-phase attribution is available even without line granularity.

### Staleness — reports vs current source

All 7 profiled operators changed after capture, but per `(op, backend)`, not per operator.
Checked against impl files, `config.yaml`, and the autotune logs.

| operator | tilelang | triton | cutile |
|---|---|---|---|
| `matrix_transpose` | **current** | current | stale |
| `weight_dequant` | **current** | current | stale |
| `matmul_fp32_fp16_fp8` | stale | current | current |
| `sigmoid` | stale | current | current |
| `batched_matmul` | stale | current | stale |
| `radix_sort` | stale | autotune log dirty | stale |
| `top_k_selection` | stale | stale | stale |

`top_k_selection` is fully invalidated — all four impls plus `config.yaml` changed, with
uncommitted edits still in the tree. Its 4 `variant_v*` reports are self-ablations with no
backend in the name.

Only `matrix_transpose` and `weight_dequant` have current TileLang-on-B200 captures.

## Agent pipeline

Built 2026-08-30 against OpenRouter's OpenAI-compatible chat completions API. The default
model is `openai/gpt-5.6-luna`; override with `TILEBENCH_AGENT_MODEL`.

- `/api/agent` is a multi-turn tool loop. It forces a tool call on the first round, then
  allows up to 10 tool rounds before asking for a final answer.
- Conversation state is browser-held per selected kernel and sent as `messages`. The
  Vercel route is stateless and does not store chats.
- The server emits SSE events: `meta`, `tool`, `delta`, `done`, `error`. The client shows
  the `tool` trace so model/tool behavior can be inspected from the page.
- Runtime tools live in `lib/agent/tools.ts`: board (`list_operators`, `get_operator`),
  TileLang benchmark source/files (`list_files`, `read_file`, `read_source`,
  `grep_files`), TileLang compiler snapshot (`list_tilelang_compiler_files`,
  `read_tilelang_compiler_file`, `grep_tilelang_compiler`), and NCU JSON
  (`find_ncu_reports`, `list_ncu_metrics`, `get_ncu_metric`,
  `get_ncu_rule_results`, `get_ncu_source`, `compare_ncu_metrics`).
- Tools are Vercel-safe: no shell, no `ncu`, no `.ncu-rep` import at runtime. Agent file
  access is constrained to exported `data/ncu_json`, exported `data/ncu_source`,
  `operators.json`, `source/<op>/impl_tilelang.py`, and the pinned
  `data/tilelang_compiler` source snapshot; large reads are pageable through `offset` /
  `next_offset`.
- The TileLang compiler snapshot is from `tile-ai/tilelang` main commit
  `056ccfc267f03498d8dcc79f6efd2aba2bae8abd`, clean worktree when captured. It ships
  root build metadata plus `tilelang/` and `src/`, not examples/tests/docs.
- **The agent must never touch a `.ncu-rep` at runtime.** Offline ETL produces per-report
  JSON with `tilebench_run/ncu_export_json.py`; ship that as `data/ncu_json`. Ship captured
  generated CUDA/source as `data/ncu_source` when available.

## Conventions

- Default and autotune are **separate columns** and are never blended into one aggregate.
  This is an explicit user requirement.
- `lib/metrics.ts` holds the pooling math; page and API both use it so they cannot disagree.
- Engineer notes and NCU write-ups are human UI surfaces. Do not expose them through the
  active agent tool loop unless the access contract is deliberately widened.
- Ratios below 1.0 mean TileLang is **slower**.

<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
