# S_llm — frozen shared framework for the formal B200 Base campaign

| item | value |
|---|---|
| S_llm commit | `81001678aa901402009d5706074ccef1e6984ea0` (branch `exp/llm`, tag `S_llm-B200-2026-10-06`) |
| frozen hashes | `tilebench/llm/v2/manifests/frozen/B200_base_2026-10-06.json` (study config hash, manifests, profile file sha + seal, scoring table sha, 45 contract hashes, skill hashes, templates, execution policy) |
| campaign branch | `exp/llm-b200` cut from S_llm, worktree `/projects/kzhou6/bcui2/research/tilebench/llm_b200_wt` |
| campaign identity | `formal_b200_base_2026-10-06` (Triton and cuTile x gpt and claude, Base, formal) |
| environment | dgx003, `LD_LIBRARY_PATH` unset, bwrap sandbox, worker wall-clock 1800 s |

The shared branch `exp/llm` may move on (this file is a follow-up commit); the campaign keeps running from S_llm. A framework
change after formal data exists requires a new, versioned campaign identity, never an in-place edit.
