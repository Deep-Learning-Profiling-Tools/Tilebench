# ARTIFACT_POLICY — what the LLM v2 campaigns produce, where it lives, what is tracked

Decided by the study owner on 2026-10-05 (NEXT_STEP_CLAUDE.md §五): the
iteration artifacts of LLM campaigns enter `main` with the final PR. This
replaces the earlier v2 convention under which raw trajectories were kept
only on an archive branch / Hugging Face. `.gitignore` is not used to hide
them silently; the rule below says exactly what is tracked and what is not.

## 1. Three locations

| Location | Tracked | Content |
|---|---|---|
| `tilebench/llm/v2/**`, `tests/llm_v2/**`, `skills/**`, `docs/llm_v2/**` | yes | source, templates, manifests, contracts, publishable Skills, documentation |
| `outputs/llm_v2/<campaign>/...` | no (`outputs/` is git-ignored) | the live run cache: every trajectory as the runner writes it, the per-campaign `_sandbox/` (compile caches of the isolated worker, deleted per evaluation), device locks, provider probes |
| `artifacts/llm_v2/<campaign>/...` | **yes** | the publication export of a campaign, produced by `python -m tilebench.llm.v2 export-publication` from the run cache |
| `artifacts/llm_v2/calibration/<device>/<calibration_id>/` | **yes** | one empirical calibration run: protocol, environment, raw per-batch samples, sealed `profile.json`, `SHA256SUMS`; written once, never edited (a new calibration is a new directory) |
| `artifacts/llm_v2/scoring/<device>/` | **yes** | scoring tables (T_emp / status / hashes of every eligible task) derived from a registered profile and the declaration file |

`llm_wt` is a worktree of the same repository on branch `exp/llm`; tracked
paths reach `main` through the branch's PR. Nothing is copied by hand
between checkouts.

## 2. What the publication export contains (per trajectory, every round)

```
artifacts/llm_v2/<campaign>/<condition>/<device>/<dsl>/<model>/<operator>/<dtype>/<case_id>/
  trajectory.json                 state: task identity, content hashes, every round/attempt record,
                                  verdicts, costs and cost_status, timing samples, notes, stop reason
  usage.jsonl                     one row per archived response (normalized + raw-linked usage)
  transport.jsonl                 every transport event (sending / succeeded / failed / refused / orphaned)
  reviews.jsonl                   human compliance decisions, when any
  round_NN/attempt_M/request.json      the exact prompt (system + user), model id, settings, request hash
  round_NN/attempt_M/response.json     the provider result: text, response id, echoed model id, terminal
                                       status, truncation flag, raw usage, raw response dump, transport list
  round_NN/attempt_M/impl_<dsl>.py     the parsed candidate
  round_NN/attempt_M/compliance.json   static + contract evidence, verdict, review items
  round_NN/attempt_M/compliance_recheck_NNNN.json   append-only rechecks (checker version, supersedes)
  round_NN/attempt_M/eval_NNNN/META.json            one directory per evaluation revision: reason (initial /
                                                    retry_incomplete / independent re-evaluation), supersedes,
                                                    executor, evaluator fingerprint, candidate sha256
  round_NN/attempt_M/eval_NNNN/evaluation.json      worker result (stage records, three samples, mean, timing record,
                                                    config reads, isolation report, seed)
  round_NN/attempt_M/eval_NNNN/evaluation/          archived sandbox evidence: job.json, result.json, progress.json,
                                                    worker stdout/stderr, the candidate file, the retained Proton
                                                    profile (*.hatchet), ARCHIVE_INDEX.json
  (campaign of 2026-10-05, before revisions existed: round_NN/attempt_M/evaluation.json + evaluation/ are that
   attempt's first and only revision; later re-evaluations of those candidates start at eval_0002 and say
   `supersedes: evaluation.json (legacy revision 1)`)
  best_valid/impl_<dsl>.py             the fastest valid candidate
<campaign>/campaign.json               run type, generator spec (model id, settings, key *variable name*), config
                                       hash, template hash, host, git SHA, isolation report
<campaign>/summary_*.json, INDEX.json (sha256 of every file), SUMMARY.md, REDACTIONS.json
```

Not only the final round or the winner: every attempt of every round,
including violations, format errors and failed evaluations, is exported.
Files are copied byte for byte; nothing is rewritten. Evidence files
(requests, responses, candidates, compliance records, evaluation revisions,
archives, ledgers) are immutable once written: a later export of the same
campaign ADDS entries to INDEX.json (new revisions, new rechecks) and
never changes their hashes. The only files whose hash legitimately changes
between exports are the living records `trajectory.json` (notes and the
revision list are appended; earlier rounds/attempts are never edited) and
the ledgers (`usage.jsonl`, `transport.jsonl`, `reviews.jsonl`, append-only).
The export of 2026-10-05 after the two `eval_0002` re-evaluations
demonstrates this: 20 entries added, the two re-evaluated trajectories'
`trajectory.json` changed (additive: `evaluation_revisions` + one note
each), the other 1,612 entries unchanged.

## 3. What never enters Git

- credentials: keys are read from environment variables named in
  `models.yaml` (`api_key_env`); they are not written to any file, and the
  evaluation worker runs without them (environment scrubbed, home hidden);
- private or non-publishable Skill bodies (`skills/manifest.json`:
  `permission: private`, `publishable: false`). A trajectory whose injected
  context includes such a component is **withheld entirely** from the
  export and listed in `REDACTIONS.json` with its content hashes (prompts
  embed the component, responses may quote it);
- compile caches (Triton / cuTile / TileLang), NEFF files, large profiler
  captures (NCU, rocprof): they live in the per-evaluation sandbox and are
  deleted with it; only the small Proton hatchet JSON of the three timed
  launches is archived. `.gitignore` re-includes exactly
  `artifacts/llm_v2/*/{base,enhanced}/**/evaluation/*.hatchet` (profiler
  output anywhere else stays ignored), so a clean clone verifies INDEX.json
  (`tests/llm_v2/test_freeze_fixes.py::test_published_index_verifies_in_a_clean_git_archive`);
- the run cache itself (`outputs/`) and any temporary ZIP.

## 4. Size and limits

A five-round trajectory (revision 3) exports roughly 0.5–1.5 MB of text (the prompt repeats
the Reference and Device Skills in every request). A 45-operator campaign
per (device, DSL, model, condition) is therefore in the order of a few
hundred MB of highly redundant text, which Git delta-compresses well but
which still exceeds what a single PR review comfortably holds. Options to
be chosen by the owner when that point is reached (not decided here):
Git LFS for `artifacts/llm_v2/**/request.json`, one PR per campaign, or
per-campaign tarballs tracked with LFS. The validation campaign of
2026-10-05 is small enough to be tracked as plain files.

## 5. Validation versus formal campaigns

Both run types are exported the same way. `campaign.json.spec.run_type`
and every `trajectory.json.run_type` say which one a record belongs to.
Validation campaigns are engineering acceptance of the execution chain:
unscored, never a distillation source (`distillation.access` refuses them
by code), never merged into formal E(B) curves (`metrics` reports them under
their own key with `scoring_note`).
