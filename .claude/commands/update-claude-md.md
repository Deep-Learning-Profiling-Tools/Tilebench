---
name: update-claude-md
description: Re-analyze the repository and update CLAUDE.md to reflect current state
---

Re-analyze the current state of this repository and update CLAUDE.md to reflect any changes.

Specifically check for:
1. New operators added to `benchmarks/operators/` (including any in-progress ones)
2. New scripts or utilities in `scripts/`
3. Changes to core modules (`core/`) that affect architecture or usage
4. New config options or patterns introduced in operator `config.yaml` files
5. Any new test patterns in `tests/`
6. Changes to `data/tensors.py` (new generators or case expansion logic)

Update CLAUDE.md in-place: keep the existing structure but revise outdated sections and add new information. Do not add sections for things that haven't changed. Keep the file concise.
