#!/usr/bin/env python3
"""Prepare stage-specific tasks for the existing GPU fleet worker.

Sorted filenames are priorities, not dependency barriers. Submit one stage at a
time after its required artifacts finish. --stage all is a preview only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys


REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
from crope.utils import image_stem, read_prompts, resolution  # noqa: E402


SCRATCH = Path("/scratch/09032/saini_2/constancy_rope")
PYTHON = Path("/work/09032/saini_2/vista/library/uv-envs/crope/bin/python")
METHODS = ("base", "pi", "ntk", "ntk2", "yarn", "yarn_temp", "dype", "sega", "sega_static",
           "constancy", "temp_only", "constancy_nogate", "constancy_sref", "sega_official", "sega_official_static")
DEFAULT_METHODS = ("ntk2", "yarn", "dype", "sega_official", "sega", "temp_only", "constancy")
PILOT_METHODS = DEFAULT_METHODS
STATIC_PROFILES = {"sega_static": "sega_static_profile", "sega_official_static": "sega_official_static_profile"}
STAGES = ("smoke", "calibration", "probe", "pilot", "pilot-static", "main", "eval")


@dataclass
class Task:
    label: str
    commands: list[list[str]]
    requirements: list[Path] = field(default_factory=list)
    estimated_seconds: float | None = None
    notes: list[str] = field(default_factory=list)
    timed: bool = False
    input_fingerprint: str = ""

    def body(self, python: Path) -> str:
        lines = [
            "#!/usr/bin/env bash", "set -euo pipefail",
            "# Fleet slurm/fleet.sbatch provides per-host GPU telemetry.",
            f"# Input fingerprint: {self.input_fingerprint}",
            f"cd {shlex.quote(str(REPO))}",
            f"export PATH={shlex.quote(str(python.parent))}:\"$PATH\"",
            "export HF_HOME=/scratch/09032/saini_2/tmp/hf",
            "export CROPE_TRANSFORMER=/scratch/09032/saini_2/tmp/hf/hub/models--black-forest-labs--FLUX.1-Krea-dev/snapshots/8162a9c7b05a641be098422bf2fcf335615c2f28",
            "export TORCH_HOME=/scratch/09032/saini_2/tmp/cache",
            "export XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache",
            "export WANDB_DIR=/scratch/09032/saini_2/constancy_rope/wandb",
            "export WANDB_PROJECT=constancy-rope",
            'if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then export WANDB_MODE=offline; fi',
            "export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8",
            "export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1",
        ]
        for requirement in self.requirements:
            quoted = shlex.quote(str(requirement))
            message = shlex.quote(f"Missing prerequisite: {requirement}. Generate this stage after its dependencies finish.")
            lines.append(f"[[ -s {quoted} ]] || {{ printf '%s\\n' {message} >&2; exit 2; }}")
        for index, command in enumerate(self.commands):
            if self.timed:
                lines.append("SECONDS=0")
            lines.append(shlex.join([str(python), *command]))
            if self.timed:
                lines.append(f"printf 'smoke_command_{index}_seconds=%s\\n' \"$SECONDS\"")
        return "\n".join(lines) + "\n"

    def filename(self, python: Path) -> str:
        digest = hashlib.sha256(self.body(python).encode()).hexdigest()[:12]
        return f"{self.label}_{digest}.sh"


def sample_command(args, method: str, res: str, start: int, end: int, root: Path) -> list[str]:
    command = ["scripts/sample.py", "--method", method, "--res", res,
               "--prompts", str(args.prompts), "--start", str(start), "--end", str(end),
               "--seed", str(args.seed), "--steps", str(args.steps),
               "--guidance", str(args.guidance), "--out", str(root / method / res)]
    if method.startswith("constancy"):
        command += ["--profile", str(args.profile)]
    if method in STATIC_PROFILES:
        command += ["--sega-static-profile", str(getattr(args, STATIC_PROFILES[method]))]
    return command


def method_requirements(args, method: str) -> list[Path]:
    paths = [args.prompts]
    if method.startswith("constancy"):
        paths.append(args.profile)
    if method in STATIC_PROFILES:
        paths.append(getattr(args, STATIC_PROFILES[method]))
    return paths


def finished_outputs(args, root: Path, methods: tuple | list, rows: list[dict], start: int, end: int) -> list[Path]:
    required = []
    for method in methods:
        for row in rows[start:end]:
            stem = root / method / args.res / image_stem(row["id"], args.seed)
            required.extend([Path(f"{stem}.png"), Path(f"{stem}.json")])
    return required


def build_plan(args: argparse.Namespace, rows: list[dict]) -> list[Task]:
    """Build a plan without writing files, starting processes, or importing torch."""
    if args.stage == "all" and not args.dry_run:
        raise ValueError("--stage all requires --dry-run; enqueue stages separately after prerequisites finish")
    start = args.start
    end = min(args.end if args.end is not None else len(rows), len(rows))
    if start < 0 or end <= start:
        raise ValueError("Selected prompt range is empty or invalid")
    if args.seconds_per_image <= 0 or not math.isfinite(args.seconds_per_image):
        raise ValueError("--seconds-per-image must be finite and positive")
    if not 30 <= args.minutes_per_task <= 45:
        raise ValueError("--minutes-per-task must be between 30 and 45")
    if args.startup_seconds < 0 or not math.isfinite(args.startup_seconds):
        raise ValueError("--startup-seconds must be finite and nonnegative")
    if args.chunk_size is not None and args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if args.pilot_count < 1 or args.native_count < 1:
        raise ValueError("Calibration and pilot counts must be positive")
    chunk = args.chunk_size or max(1, math.floor((args.minutes_per_task * 60 - args.startup_seconds) / args.seconds_per_image))
    stages = STAGES if args.stage == "all" else (args.stage,)
    pilot_end = min(start + args.pilot_count, end)
    pilot_root = args.out / "pilot"
    tasks = []
    for stage in stages:
        if stage == "smoke":
            commands = [sample_command(args, "base", "1024x1024", start, start + 1, args.out / "smoke"),
                        sample_command(args, "ntk2", "4096x4096", start, start + 1, args.out / "smoke")]
            tasks.append(Task("00_smoke", commands, [args.prompts], timed=True,
                              notes=["One native and one 4K image; separate wall times are printed."]))
        elif stage == "calibration":
            commands = [["scripts/calibrate.py", "--mode", "native", "--prompts", str(args.calibration_prompts),
                         "--count", str(args.native_count), "--seed", str(args.seed), "--steps", str(args.steps),
                         "--guidance", str(args.guidance), "--out", str(args.profile)]]
            notes = ["Use --calibration-prompts with disjoint IDs for a held-out native profile."]
            tasks.append(Task("10_calibration_native", commands, [args.calibration_prompts], notes=notes))
        elif stage == "probe":
            command = ["scripts/probe_mtf.py", "--prompts", str(args.prompts), "--count", "1",
                       "--method", "ntk2", "--res", args.res, "--seed", str(args.seed),
                       "--steps", str(args.steps), "--guidance", str(args.guidance),
                       "--out", str(args.probe_out / args.res)]
            tasks.append(Task(f"20_probe_mtf_{args.res}", [command], [args.prompts]))
        elif stage == "pilot":
            for method in PILOT_METHODS:
                commands = [sample_command(args, method, args.res, start, pilot_end, pilot_root)]
                notes = ["T3 retains matched NTK2 image spectra for evaluation."] if method == "ntk2" else []
                if method in ("sega", "sega_official"):
                    commands.append(["scripts/calibrate.py", "--mode", "sega-static", "--method", method, "--runs",
                                     str(pilot_root / method / args.res), "--res", args.res,
                                     "--out", str(getattr(args, STATIC_PROFILES[f"{method}_static"]))])
                    notes.append("Aggregate SEGA logs in this task; enqueue pilot-static after this finishes.")
                estimate = args.startup_seconds + (pilot_end - start) * args.seconds_per_image
                tasks.append(Task(f"30_pilot_{method}_{args.res}_{start:05d}-{pilot_end:05d}", commands,
                                  method_requirements(args, method), estimate, notes))
        elif stage == "pilot-static":
            for method in STATIC_PROFILES:
                required = method_requirements(args, method)
                required += finished_outputs(args, pilot_root, (method.removesuffix("_static"),), rows, start, pilot_end)
                for lower in range(start, pilot_end, chunk):
                    upper = min(lower + chunk, pilot_end)
                    command = sample_command(args, method, args.res, lower, upper, pilot_root)
                    estimate = args.startup_seconds + (upper - lower) * args.seconds_per_image
                    tasks.append(Task(f"31_pilot_{method}_{args.res}_{lower:05d}-{upper:05d}", [command],
                                      required, estimate, ["T1 uses the completed pilot's target-resolution SEGA average."]))
        elif stage == "main":
            for method in args.methods:
                for lower in range(start, end, chunk):
                    upper = min(lower + chunk, end)
                    command = sample_command(args, method, args.res, lower, upper, args.out / "main")
                    estimate = args.startup_seconds + (upper - lower) * args.seconds_per_image
                    tasks.append(Task(f"50_main_{method}_{args.res}_{lower:05d}-{upper:05d}", [command],
                                      method_requirements(args, method), estimate))
        elif stage == "eval":
            eval_root = pilot_root if args.eval_scope == "pilot" else args.out / "main"
            methods = (*PILOT_METHODS, *STATIC_PROFILES) if args.eval_scope == "pilot" else args.methods
            selected_end = pilot_end if args.eval_scope == "pilot" else end
            requirements = [args.prompts, args.references]
            requirements += finished_outputs(args, eval_root, methods, rows, start, selected_end)
            command = ["scripts/eval.py", "--root", str(eval_root), "--prompts", str(args.prompts),
                       "--references", str(args.references), "--reference-method", args.reference_method,
                       "--res", args.res, "--out", str(args.eval_out / args.eval_scope / args.res)]
            tasks.append(Task(f"90_eval_{args.eval_scope}_{args.res}", [command], requirements,
                              notes=["Generation prerequisites must be complete before enqueueing evaluation."]))
    for task in tasks:
        fingerprint = hashlib.sha256()
        for path in task.requirements:
            fingerprint.update(str(path).encode())
            if path.is_file():
                stat = path.stat()
                fingerprint.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
                if path in (args.prompts, args.calibration_prompts, args.profile,
                            args.sega_static_profile, args.sega_official_static_profile):
                    fingerprint.update(path.read_bytes())
            else:
                fingerprint.update(b"missing")
        task.input_fingerprint = fingerprint.hexdigest()
    return tasks


def task_status(queue: Path, name: str) -> str | None:
    for state in ("pending", "running", "done", "failed"):
        directory = queue / state
        if (directory / name).exists() or (state == "running" and any(directory.glob(name + ".*"))):
            return state
    return None


def write_tasks(args: argparse.Namespace, tasks: list[Task]) -> list[dict]:
    """Validate the whole stage before publishing atomically to the spool."""
    missing = sorted({str(path) for task in tasks for path in task.requirements
                      if not path.is_file() or path.stat().st_size == 0})
    if missing:
        display = "\n".join(missing[:12])
        raise ValueError(f"Missing {len(missing)} prerequisites; nothing enqueued:\n{display}")
    records = []
    for task in tasks:
        name = task.filename(args.python)
        state = task_status(args.queue, name)
        records.append({"name": name, "state": state or "new"})
    for directory in ("pending", "running", "done", "failed"):
        (args.queue / directory).mkdir(parents=True, exist_ok=True)
    # Worker redirects stdout before entering each script.
    (SCRATCH / "logs").mkdir(parents=True, exist_ok=True)
    staging = args.queue / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    for task, record in zip(tasks, records):
        if record["state"] != "new":
            continue
        temporary = staging / f"{record['name']}.tmp.{os.getpid()}"
        temporary.write_text(task.body(args.python), encoding="utf-8")
        temporary.chmod(0o750)
        os.replace(temporary, args.queue / "pending" / record["name"])
    return records


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    result.add_argument("--prompts", type=Path, required=True)
    result.add_argument("--calibration-prompts", type=Path, help="Separate native-calibration captions, preferably disjoint")
    result.add_argument("--references", type=Path, default=SCRATCH / "prompts/aesthetic4k_refs.jsonl")
    result.add_argument("--out", type=Path, default=SCRATCH / "out", help="Campaign root containing {main,pilot,smoke}/METHOD/RES")
    result.add_argument("--queue", type=Path, default=SCRATCH / "taskq")
    result.add_argument("--profile", type=Path, default=SCRATCH / "calib/flux_dev_profile.npz")
    result.add_argument("--sega-static-profile", type=Path, help="Defaults to calib/sega_static_RES.npz")
    result.add_argument("--sega-official-static-profile", type=Path, help="Defaults to calib/sega_official_static_RES.npz")
    result.add_argument("--probe-out", type=Path, default=SCRATCH / "probe")
    result.add_argument("--eval-out", type=Path, default=SCRATCH / "eval")
    result.add_argument("--python", type=Path, default=PYTHON)
    result.add_argument("--res", default="4096x4096")
    result.add_argument("--methods", choices=METHODS, nargs="+", default=list(DEFAULT_METHODS))
    result.add_argument("--start", type=int, default=0)
    result.add_argument("--end", type=int, help="Exclusive prompt row; defaults to all available captions")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--steps", type=int, default=28)
    result.add_argument("--guidance", type=float, default=4.5)
    result.add_argument("--native-count", type=int, default=16)
    result.add_argument("--pilot-count", type=int, default=8)
    result.add_argument("--seconds-per-image", type=float, default=300.0,
                        help="Planning estimate, not measured throughput; update after the GPU smoke")
    result.add_argument("--startup-seconds", type=float, default=120.0)
    result.add_argument("--minutes-per-task", type=float, default=40.0)
    result.add_argument("--chunk-size", type=int, help="Override automatic per-method main/pilot-static chunks")
    result.add_argument("--eval-scope", choices=("main", "pilot"), default="main")
    result.add_argument("--reference-method", choices=METHODS, default="sega_official")
    result.add_argument("--dry-run", action="store_true", help="Print a plan without creating any directories or files")
    result.add_argument("--show-scripts", action="store_true", help="Print complete shell scripts with the plan")
    result.add_argument("--json", action="store_true", help="Print the plan as JSON")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.res = "x".join(map(str, resolution(args.res)))
        for name in ("prompts", "references", "out", "queue", "profile", "probe_out", "eval_out", "python"):
            setattr(args, name, getattr(args, name).absolute())
        args.calibration_prompts = (args.calibration_prompts or args.prompts).absolute()
        for method, name in STATIC_PROFILES.items():
            setattr(args, name, (getattr(args, name) or SCRATCH / f"calib/{method}_{args.res}.npz").absolute())
        rows = read_prompts(args.prompts)
        tasks = build_plan(args, rows)
        plan = []
        for task in tasks:
            missing = [str(path) for path in task.requirements if not path.is_file() or path.stat().st_size == 0]
            entry = {"name": task.filename(args.python), "commands": task.commands,
                     "estimated_seconds": task.estimated_seconds, "estimate_kind": "planning_only",
                     "missing_prerequisites": missing, "notes": task.notes,
                     "spool_state": task_status(args.queue, task.filename(args.python))}
            if task.estimated_seconds is not None and task.estimated_seconds > 45 * 60:
                entry["notes"] = [*entry["notes"], "Estimated runtime exceeds 45 minutes; revise timing or pilot size."]
            if args.show_scripts:
                entry["script"] = task.body(args.python)
            plan.append(entry)
        if not args.dry_run:
            write_tasks(args, tasks)
    except (OSError, ValueError) as error:
        parser().error(str(error))
    if args.json:
        print(json.dumps({"dry_run": args.dry_run, "stage": args.stage, "tasks": plan}, indent=2))
    else:
        print(f"{'DRY RUN: no files written' if args.dry_run else 'Stage enqueued'}; {len(tasks)} tasks")
        print("Runtime values are planning estimates. Replace them after measuring each method and resolution.")
        for entry in plan:
            seconds = entry["estimated_seconds"]
            estimate = "unmeasured" if seconds is None else f"estimated {seconds / 60:.1f} min"
            print(f"{entry['name']}: {estimate}; existing={entry['spool_state'] or 'none'}")
            if entry["missing_prerequisites"]:
                print(f"  Missing {len(entry['missing_prerequisites'])} prerequisites; real enqueue is blocked.")
            for note in entry["notes"]:
                print(f"  {note}")
            if "script" in entry:
                print(entry["script"], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
