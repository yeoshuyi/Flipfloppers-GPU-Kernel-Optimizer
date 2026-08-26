# Agent Roles, Limits, and Best Practices

**Current configuration: 1 main agent + 1 subagent + 3 scripts.**

This document covers the full six-role design as well, because you may want to
scale up if you move off Pro, and because knowing *why* a role was cut is what
stops you re-adding it by accident.

---

## 1. Why only one subagent

The test: **does the agent save more tokens than it costs?**

| Role | Verdict | Reasoning |
|---|---|---|
| **profiler** | **KEEP** | Raw `ncu --set full` is 25k–100k tokens. Isolating it in a separate context and returning ~20 lines of JSON is a net token **saver**. The only role that pays for itself. |
| strategist | cut → main agent | Same model, same knowledge, different prompt. Pure duplication of context. |
| implementer | cut → main agent | This is "write the code". The main agent is already doing it. |
| verifier | cut → `run_accuracy()` | Returns pass/fail JSON against a fixed threshold. No judgement content. |
| adversary | cut → `tools/check_validity.py` | ~90% mechanically detectable. A script catches it deterministically, every time, free. |
| archivist | cut → `tools/archive.py` | File I/O. Zero reasoning content. |

**Overhead: ~45k tokens/iteration → ~4k worst case.**

> The adversary cut is the one to understand properly. The *role* is still
> mandatory — an agent optimising a scalar latency objective will find the
> harness exploit. What changed is the mechanism: a static gate is strictly
> better than an LLM here because it is deterministic, free, and cannot be
> talked out of a verdict. The residual ~10% (unchecked preconditions, thresholds
> fitted to disclosed shapes) is a one-time human review at solidification, not
> a per-iteration LLM call.

---

## 2. Role definitions

### 2.1 MAIN AGENT (your session) — Orchestrator + Strategist + Implementer

**Responsibilities**
- Own the budget, the tier ladder, and the gates
- Read `docs/DIAGNOSIS.md` after profiling; map facts → action
- Read `docs/CATALOGUE.md`; select **one** optimisation per iteration
- Write the diff
- Run the scripts in order and interpret their output
- Decide when a regime cell is exhausted

**Limits — hard**
- **Never** runs `python` on the GPU directly. Only `sbatch`.
- **Never** pastes raw `ncu`/`nsys` output into its own context.
- **Never** bundles two optimisations into one candidate.
- **Never** benchmarks before accuracy passes.
- **Never** touches `nvidia-smi` clock state — that is the Slurm prolog's job.
- Must cite a specific profiler fact for every proposal.
- Must name the target regime for every proposal.

### 2.2 PROFILER (subagent) — the one that stays

**Responsibilities**
- Submit profiling jobs via `sbatch`, poll for the result
- Parse and return a compact JSON fact block, nothing else
- Normalise all throughput to `pct_of_peak` (peak by dtype: 82.6 / 165 / 330)

**Limits — hard**
- **Returns JSON only.** No prose, no preamble, no summary.
- **Never suggests an optimisation.** Facts only. The moment it starts advising,
  it duplicates the strategist and the token saving evaporates.
- **Never edits a file.** Read + Bash(sbatch/sacct) only.
- Top 3 hot kernels maximum — a full kernel list defeats the purpose.
- If the correct peak for a dtype is ambiguous, returns `null` and says which.
  Never guesses — a wrong `pct_of_peak` misdirects the entire next iteration.

**Why it survives the cut:** context isolation. It is the only role whose *input*
is enormous and whose *output* is tiny. Every other role had comparable input and
output sizes, which is what made them pure overhead.

### 2.3 Scripts (replace three former agents)

| Script | Replaces | Runs | Cost |
|---|---|---|---|
| `tools/check_validity.py` | adversary | before every accuracy run | 0 tokens |
| `run_accuracy()` in the job harness | verifier | every candidate | 0 tokens |
| `tools/archive.py` | archivist | after every verdict | 0 tokens |

`check_validity.py` catches: `data_ptr()` outside the mask-cache whitelist,
`Parameter`/`Buffer` registration, explicit `attn_mask` on SDPA, module-level
mutable caches, and any speedup implying a latency below the FP8 theoretical
floor.

---

## 3. The full six-role design (reference — do not deploy on Pro)

If you move to a plan with headroom, or run this on API credits, the expanded
roster is below. **Do not deploy it as-is on Pro** — it is roughly 11× the token
cost per iteration.

| Agent | Model | Responsibility | Hard limit |
|---|---|---|---|
| orchestrator | (lead session) | Budget, gates, tier ladder | Writes no code |
| profiler | haiku | ncu/nsys → JSON facts | No advice, no edits |
| strategist | opus | Facts + catalogue → ranked strategies | **Prose only, no code** |
| implementer | sonnet | One strategy → one diff | One per invocation, no benchmarking |
| verifier | haiku | Full sweep accuracy | **Gate, not advisor.** Cannot approve. |
| adversary | opus | Validity test | **Veto authority.** When uncertain, veto. |
| archivist | haiku | MAP-Elites + lineage | Records every outcome incl. failures |

**Enabling coordinated teams** (experimental, off by default):
```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```
Without it you still get plain subagents — separate contexts and scoped tools —
which is the part that actually matters here.

---

## 4. Platform limits you must design around

| Limit | Consequence |
|---|---|
| **Clean context per invocation** | A subagent remembers nothing between calls. Everything it needs must be in the invocation. Do not assume it saw the conversation. |
| **Cannot ask follow-up questions** | A subagent that needs clarification will guess. Prompts must be self-contained. |
| **Returns a summary, not its transcript** | The parent sees only the final message. Intermediate reasoning is lost — design the return format deliberately. |
| **Nesting capped at depth 5** | Subagents spawning subagents hits a wall. Keep the tree flat. |
| **Tool restrictions are hard** | A subagent without `Edit` cannot edit, and will not tell you it wanted to. Scope generously enough to do the job, narrowly enough to prevent scope creep. |
| **~7× token cost** | Each subagent maintains its own context window. This is the number that killed five of the six roles. |
| **Latency per invocation** | Each call is a fresh session with startup cost. Batch related work into one call. |

---

## 5. Best practices

### Design
1. **One responsibility per agent.** A "profiler and optimiser" agent will drift
   into advising, and you lose the context isolation that justified it.
2. **Write the `description` field for auto-delegation.** It is what the main
   agent matches against. Action-oriented and specific: *"Use at the start of
   every optimisation iteration"* beats *"Profiles kernels"*.
3. **Scope tools to the minimum.** The profiler gets `Read` and
   `Bash(sbatch:*)`. Not `Write`. Not `Bash(*)`.
4. **Pick the cheapest model that does the job.** Mechanical extraction → haiku.
   Judgement under ambiguity → opus. Do not default everything to the largest
   model.
5. **Specify the return format exactly**, ideally as a literal JSON schema in
   the system prompt. Free-form returns re-inflate the context you were trying
   to protect.
6. **Version-control `.claude/agents/`.** These are project artefacts and their
   evolution is part of the report.

### Operation
7. **Never let raw profiler output reach the main context.** This single rule is
   worth more than every other item here.
8. **`/clear` between archive cells.** Each regime is an independent problem.
   Carrying tiny-regime context into long-seq work is dead weight.
9. **One optimisation per iteration.** Bundling makes failures undiagnosable and
   costs a full re-run.
10. **Run free gates before expensive ones.** `check_validity.py` (0 tokens, 0
    GPU) → `compile_check` (0 tokens, CPU) → `run_accuracy` (GPU job) →
    `run_bench` (GPU job).
11. **`Grep` before `Read`.** Never read a whole file to locate one function.
12. **Let the agent stop.** The JSON result block is the deliverable. Prose
    summarising a diff you can already see is pure cost.

### Headless
13. **Always wrap in `timeout`.** A stuck agent runs until killed.
14. **Check exit codes**, do not merely `tee`. Non-zero on tool failure and
    rate-limit exhaustion.
15. **Per-run `HOME`.** Concurrent runs sharing `~/.claude` corrupt session state.
16. **Log `total_cost_usd` per iteration.** A retry loop on a bad prompt is the
    most expensive failure mode there is.
17. **Prompts must be self-contained.** There is no follow-up turn. A prompt
    ending *"let me know if you also want…"* is a hang, not a clarification.

---

## 6. Bootstrap validation — do this before anything hard

Give the loop **G3.2 (fused LayerNorm + residual)** — a task with a known-good
answer. Confirm it can autonomously: profile → diagnose → implement → validate →
verify → benchmark → commit.

**If the loop cannot do that, fix the loop before pointing it at G4.**
Discovering a broken loop against the megakernel costs days.

---

## 7. Headless driver

```bash
#!/bin/bash
# loop.sh <cell> [iters]
set -euo pipefail
CELL="$1"; ITERS="${2:-8}"

for i in $(seq 1 "$ITERS"); do
  echo "=== $CELL iteration $i ==="
  HOME="/scratch/techjam2/runs/home_${CELL}_${i}" \
  timeout 45m claude -p "$(sed "s/{{CELL}}/$CELL/" prompts/optimise.md)" \
      --output-format json \
      --max-turns 30 \
      --allowedTools "Read,Edit,Write,Glob,Grep,Bash(sbatch:*),Bash(sacct:*),Bash(python3 tools/*)" \
      > "runs/${CELL}_${i}.json" || { echo "iter $i failed"; continue; }

  jq -r '.total_cost_usd' "runs/${CELL}_${i}.json" >> runs/cost.log
  [[ "$(jq -r '.structured_output.improved' "runs/${CELL}_${i}.json")" == "false" ]] \
      && { echo "no improvement; stopping"; break; }
done
```

## 8. Permissions

```json
// .claude/settings.json
{
  "permissions": {
    "allow": [
      "Read", "Glob", "Grep", "Edit", "Write",
      "Bash(sbatch:*)", "Bash(squeue:*)", "Bash(sacct:*)", "Bash(scancel:*)",
      "Bash(git diff:*)", "Bash(git log:*)", "Bash(git add:*)",
      "Bash(git commit:*)", "Bash(python3 tools/*)",
      "Bash(nvidia-smi --query-gpu*)"
    ],
    "deny": [
      "Bash(rm -rf:*)", "Bash(sudo:*)",
      "Bash(nvidia-smi -r*)", "Bash(nvidia-smi -lgc*)", "Bash(nvidia-smi -pl*)",
      "Read(./.env)", "Read(./**/*.key)"
    ]
  }
}
```

`nvidia-smi` clock commands are **denied deliberately**. Clocks belong to the
Slurm prolog; an agent resetting them mid-sweep silently invalidates every
measurement taken afterward, and nothing in the output will indicate it happened.
