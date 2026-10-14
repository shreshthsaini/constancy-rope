## Goal
Validate Constancy RoPE (per-octave RoPE q/k gain = tau_N * exp(eta * Wiener gate * native-profile deficit)) against SEGA, DyPE, YaRN, NTK on FLUX.1-dev at 4096^2 with real generations; write a full paper ONLY if results are good.

## Status and next action
2026-10-14 ~04:45. Setup phase. Fleet job 1071114 (CR1, gh, 4 nodes, 24h) PENDING. Env build (uv env crope) + FLUX.1-dev download running in background on i618-122 (logs: /scratch/09032/saini_2/constancy_rope/logs/{setup_env,download}.log).
Codex implementation dispatch: see notes/codex.md.
NEXT: review Codex output, CPU tests, then enqueue GPU smoke + calibration tasks into taskq.

## Map
| note | read it when |
| notes/design.md | implementing or changing the method, baselines, metrics |
| notes/codex.md | checking/redispatching Codex jobs |
| notes/results.md | any result exists or a paper decision is needed |
| ../research_reviews/sega_2605.22668_2026-10-14.md | background: SEGA critique + CPU mechanism tests |

## Decisions
- Model FLUX.1-dev (Krea variant not downloaded; SEGA used Krea). Why: available + DyPE uses dev.
- All scaled methods share the strong NTK base b*s^(2D/(D-2)) (SEGA's choice). Why: isolates the q/k gain law.
- Paper only if Constancy beats SEGA on 4K-sensitive metrics with CIs (user instruction).
- Fleet = one multi-node job draining /scratch/09032/saini_2/constancy_rope/taskq. Why: job-shape doctrine.

## Open questions
- T2 (attention MTF probe): does raising an octave's gain raise that octave's power? If not, method fails.
