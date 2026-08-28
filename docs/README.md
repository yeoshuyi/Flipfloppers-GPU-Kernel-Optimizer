# Lean setup — 1 main agent + 1 subagent

## Why only one subagent

| Role | Verdict | Reason |
|---|---|---|
| **profiler** | **KEEP as subagent** | Raw `ncu --set full` is 25k–100k tokens. Isolating it in a separate context and returning ~20 lines is a net token **saver**. The only role that pays for itself. |
| strategist | cut → main agent | Same model, same knowledge, different prompt. Pure duplication. |
| implementer | cut → main agent | This is just "write the code". |
| verifier | cut → `run_accuracy()` | Returns pass/fail JSON. No judgement needed. |
| adversary | cut → `tools/check_validity.py` | ~90% mechanically detectable. Script catches it free, every time, deterministically. Residual 10% is a one-time human review at solidification. |
| archivist | cut → `tools/archive.py` | File I/O. Zero reasoning content. |

**6 agents → 1 subagent + 2 scripts + your main session.**

## Context budget per session

```
CLAUDE.md          ~1,900 tok   always loaded
docs/CATALOGUE.md  ~1,345 tok   on demand, once per session
docs/DIAGNOSIS.md    ~418 tok   on demand, once per session
profiler subagent    ~322 tok   isolated context, ~20 lines returned
                    ---------
worst case          ~4,000 tok of instruction overhead
```

Previous 6-agent design: ~7,600 tok × 6 contexts ≈ 45k tok of duplication
**per iteration**, before any actual work.

## Rules that actually save tokens

1. **Never let raw `ncu` output into the main context.** Use the profiler
   subagent. This is worth more than every other rule combined.
2. **One optimisation per iteration.** Bundling makes failures undiagnosable
   and costs a full re-run.
3. **Run `check_validity.py` before `run_accuracy`.** A failed accuracy run
   costs a GPU job *and* a diagnosis round trip; the static gate costs nothing.
4. **`Grep` before `Read`.** Never read a whole file to find one function.
5. **Don't re-read `docs/*.md`** already read this session.
6. **`/clear` between archive cells.** Each regime is an independent problem;
   carrying tiny-regime context into long-seq work is dead weight.
7. **Let the model stop.** The JSON result block is the deliverable — prose
   summarising a diff you can already see is pure cost.

## Files

```
CLAUDE.md                    always loaded — invariants, ground truth, loop
docs/CATALOGUE.md            read before proposing (G0–G4)
docs/DIAGNOSIS.md            read after profiling (fact → action)
docs/MEGAKERNEL.md           read only when working on G4
.claude/agents/profiler.md   the one subagent
tools/check_validity.py      static gate — replaces the adversary
tools/archive.py             MAP-Elites — replaces the archivist
```

## Loop

```bash
# 1. profile (subagent, isolated context)
#    "Use the profiler subagent on the current default elite."
# 2. read docs/DIAGNOSIS.md, pick a row
# 3. read docs/CATALOGUE.md, pick ONE optimisation
# 4. implement
python3 tools/check_validity.py benchmark.py        # free gate
sbatch infra/slurm/<candidate>.sbatch              # accuracy + matched before/after in one job
python3 tools/archive.py commit --cell default/fp8 --id cand_0042 \
        --speedup 12.4 --applied G0.1,G1.1
```

## If you want it even cheaper

Drop the profiler subagent too and have `tools/` parse `ncu` to JSON directly —
`ncu --csv --page raw` piped through a parser. Then it is a **single agent with
three scripts**, and no subagent context at all. Do this if you find yourself
rate-limited; the only loss is the subagent's judgement about which counters
matter for an unfamiliar kernel shape.
