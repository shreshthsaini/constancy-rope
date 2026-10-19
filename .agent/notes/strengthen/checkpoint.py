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
Strengthen finished findings paper fdee66d with E1 fresh150, E2 seeds1/2 and E3 Qwen.
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
| .agent/notes/strengthen/seed1_reuse.json | Reused existing165 held-out seed1 images |
| .agent/notes/strengthen/fresh_selection* | Dataset selection, exclusions and reference provenance |
| .agent/notes/state_before_strengthen_20261019.md | Historical state before this task |
| paper/PLAN.md | Paper honesty, formatting and evidence requirements |
| .agent/notes/main2/REPORT.md | Existing negative results and raw evidence |
| .agent/notes/results.md | Append-only campaign progress |
| slurm/fleet_CR22.sbatch | Existing8-GH200/24h fleet; no submissions authorized here |

## Decisions
- H1 joint IR/Pick equivalence; H2 full-MUSIQ superiority; H3 IR improvement ceiling0.03.
- E2 averages seeds0/1/2 within prompt before bootstrap; n187, not561; seeds1/2 sensitivity.
- No selection/tuning on new outputs. Incomplete blocks cannot establish hypotheses.
- Preserve prior seed1 and all historical results; verify before reuse.
- Shared CPU host: OMP8 and own processes<20. No local GPU work.
- Never ssh,sbatch,scancel,pkill,killall; no scratch deletion; worker scripts unchanged.
- .git read-only: leave edits, no commits. No em dashes or section symbol in prose.
- CR22 pending: scheduler polls at least10minutes apart; sleep when no useful work.

## Open questions
- GPU completion and final hypothesis outcomes remain pending unless recorded above.
- Qwen transfer needs official settings, tiny CPU tests and a successful GPU smoke first.
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
