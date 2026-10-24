"""Small local checkpoint writer; no scheduler, generation or scoring actions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / '.agent/notes/strengthen'


def checkpoint(title, details, status, next_action, **updates):
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    path = HERE / 'live.json'
    live = json.loads(path.read_text()) if path.exists() else {}
    live.update(updates)
    live.update(timestamp=now, status=status, next_action=next_action)
    path.write_text(json.dumps(live, indent=2) + '\n')
    with (ROOT / '.agent/notes/results.md').open('a') as f:
        f.write(f'\n### {now}: {title}\n{details}\n')
    state = f'''## Goal
Strengthen findings paper with factorial, checkpoint/guidance, temperature, fresh150, seeds and Qwen.
Report all outcomes, revise claims, rebuild/inspect PDF and refresh standalone source zip.

## Status and next action
Updated {now}.
{status}
NEXT: {next_action}
Live machine-readable progress: .agent/notes/strengthen/live.json.

## Map
| File | Read when |
| .agent/notes/prereg_strengthen.md | Exact frozen hypotheses, selection, statistics and amendments |
| .agent/notes/strengthen/live.json | Current queue counts, receipts, next steps and scheduler observation |
| .agent/notes/strengthen/*_queue.json | Frozen task identities, scripts and source hashes |
| .agent/notes/strengthen/seed1_reuse.json | Reused 110 held-out seed-1 images for the retained two arms |
| .agent/notes/strengthen/fresh_selection* | Dataset selection, exclusions and reference provenance |
| .agent/notes/strengthen/ANALYSIS_AND_EVALUATION.md | One-shot completion gates and analysis commands |
| .agent/notes/strengthen/qwen_protocol.md | Qwen settings, smoke acceptance and source/model pins |
| .agent/notes/strengthen/diagnostics_protocol.md | MUSIQ preprocessing and schedule audit |
| scripts/strengthen_status.py | Enforced ten-minute scheduler polls and spool identity counts |
| .agent/notes/state_before_strengthen_20261019.md | Historical state before this task |
| paper/PLAN.md | Paper honesty, formatting and evidence requirements |
| .agent/notes/main2/REPORT.md | Existing negative results and raw evidence |
| .agent/notes/results.md | Append-only campaign progress |
| slurm/fleet_CR22.sbatch | Existing 8-GH200, 24-hour fleet; no submissions authorized here |

## Decisions
- Amended H1 pools 337 prompts, IR margin 0.081977, Pick margin 0.10; H2 fresh MUSIQ; Holm family of 3.
- H3 excluded from new work; retain historical negatives. P1-P5 queued before Qwen.
- E2 averages seeds 0/1/2 within prompt before bootstrap; n=187; seeds 1/2 sensitivity separately.
- No selection/tuning on new outputs. Incomplete blocks cannot establish hypotheses.
- Preserve prior seed 1 and all historical results; verify before reuse.
- Shared CPU host: OMP_NUM_THREADS=8 and own processes under 20. No local GPU work.
- Never ssh, sbatch, scancel, pkill or killall; no scratch deletion; worker scripts unchanged.
- .git read-only: leave edits, no commits. No em dashes or section symbol in prose.
- While CR22 is pending, scheduler polls stay at least 10 minutes apart; sleep when no useful work.

## Open questions
- GPU completion and final hypothesis outcomes remain pending unless recorded above.
- Qwen settings and CPU parity checks are frozen; full transfer still needs a successful GPU smoke.
- Metrics are surrogate evidence; no human-preference, universal-transfer or SOTA claims.
'''
    if len(state.splitlines()) >= 120:
        raise ValueError('State exceeds line limit')
    (ROOT / '.agent/STATE.md').write_text(state)
    handover = ROOT / 'HANDOVER.md'
    text = handover.read_text()
    start = text.index('## Evidence strengthening (current, 2026-10-19 UTC)')
    end = text.index('## Paper status (2026-10-18, camera-ready pass)', start)
    current = f'''## Evidence strengthening (current, 2026-10-19 UTC)

Last update: {now}. Active task confirms the finished findings paper.
Read `.agent/STATE.md`, immutable `.agent/notes/prereg_strengthen.md` and
`.agent/notes/strengthen/live.json` for current state and resume details.
{status}
Next: {next_action}
Current session forbids SSH, submissions/cancellations, worker edits, scratch deletion and commits.
All older entries below are historical and do not override the current task.

'''
    handover.write_text(text[:start] + current + text[end:])
    print(now, title)


if __name__ == '__main__':
    checkpoint(**json.loads(sys.stdin.read()))
