"""CPU coverage for prompt provenance and dependency-safe task planning."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"crope_script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prompts = load_script("prompts")
enqueue = load_script("enqueue")


def write_prompts(path, count=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"example.{index}", "prompt": f"A detailed photograph {index}."} for index in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def plan_args(tmp_path, *extra):
    source = tmp_path / "prompts.jsonl"
    if not source.exists():
        write_prompts(source)
    args = enqueue.parser().parse_args([
        "--prompts", str(source), "--queue", str(tmp_path / "spool"),
        "--out", str(tmp_path / "out"), "--references", str(tmp_path / "refs.jsonl"),
        "--profile", str(tmp_path / "profile.npz"),
        "--sega-static-profile", str(tmp_path / "static.npz"), *extra,
    ])
    args.calibration_prompts = args.calibration_prompts or source
    return args


def test_prompt_export_preserves_order_revision_and_matched_references(tmp_path):
    revision = "a" * 40
    folder = tmp_path / "snapshots" / revision / "eval" / "size_4096"
    (folder / "images").mkdir(parents=True)
    for name in ("minus27.jpg", "39.jpeg"):
        (folder / "images" / name).write_bytes(b"test reference")
    metadata = folder / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"file_name": "images/minus27.jpg", "text": "First caption"}) + "\n" +
        json.dumps({"file_name": "images/39.jpeg", "text": "Second caption"}) + "\n"
    )
    args = prompts.parser().parse_args([
        "--metadata", str(metadata), "--offline", "--out", str(tmp_path / "prompts.jsonl"),
        "--references", str(tmp_path / "references.jsonl"),
    ])
    report = prompts.export(args)
    exported = [json.loads(line) for line in args.out.read_text().splitlines()]
    references = [json.loads(line) for line in args.references.read_text().splitlines()]
    assert report["count"] == 2
    assert report["revision"] == revision
    assert [row["id"] for row in exported] == ["a4k_eval_4096_minus27", "a4k_eval_4096_39"]
    assert [row["prompt"] for row in exported] == ["First caption", "Second caption"]
    assert [row["reference_image"] for row in exported] == [row["path"] for row in references]
    before = args.out.read_bytes(), args.references.read_bytes()
    prompts.export(args)
    assert before == (args.out.read_bytes(), args.references.read_bytes())


@pytest.mark.parametrize("file_name", ["../escape.jpg", "/absolute.jpg", "images/../../escape.jpg", "images\\escape.jpg"])
def test_prompt_metadata_rejects_path_traversal(tmp_path, file_name):
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps({"file_name": file_name, "text": "A caption"}) + "\n")
    with pytest.raises(ValueError, match="unsafe file_name"):
        prompts.parse_metadata(metadata)


def test_all_stage_dry_run_has_no_filesystem_mutations(tmp_path, capsys):
    source = tmp_path / "prompts.jsonl"
    write_prompts(source)
    before = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")}
    enqueue.main([
        "--prompts", str(source), "--queue", str(tmp_path / "spool"),
        "--out", str(tmp_path / "out"), "--profile", str(tmp_path / "absent.npz"),
        "--sega-static-profile", str(tmp_path / "absent-static.npz"),
        "--stage", "all", "--dry-run", "--json",
    ])
    after = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")}
    assert after == before
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert {task["name"].split("_", 1)[0] for task in plan["tasks"]} >= {"00", "10", "20", "30", "31", "50", "90"}
    assert any(task["missing_prerequisites"] for task in plan["tasks"])
    smoke = next(task for task in plan["tasks"] if task["name"].startswith("00_"))
    assert [command[command.index("--res") + 1] for command in smoke["commands"]] == ["1024x1024", "4096x4096"]


def test_missing_profile_blocks_entire_stage_before_spool_creation(tmp_path):
    args = plan_args(tmp_path, "--stage", "main", "--methods", "ntk2", "constancy")
    tasks = enqueue.build_plan(args, enqueue.read_prompts(args.prompts))
    with pytest.raises(ValueError, match="nothing enqueued"):
        enqueue.write_tasks(args, tasks)
    assert not args.queue.exists()


def test_all_real_stage_rejected(tmp_path):
    args = plan_args(tmp_path)
    with pytest.raises(ValueError, match="requires --dry-run"):
        enqueue.build_plan(args, enqueue.read_prompts(args.prompts))


def test_main_chunk_size_and_input_identity(tmp_path):
    args = plan_args(tmp_path, "--stage", "main", "--methods", "ntk2", "--dry-run")
    rows = enqueue.read_prompts(args.prompts)
    tasks = enqueue.build_plan(args, rows)
    ranges = [(task.commands[0][task.commands[0].index("--start") + 1],
               task.commands[0][task.commands[0].index("--end") + 1]) for task in tasks]
    assert ranges == [("0", "7"), ("7", "12")]
    assert tasks[0].estimated_seconds == 120 + 7 * 300
    name_before = tasks[0].filename(args.python)
    rows[0]["prompt"] = "A revised, different prompt."
    args.prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert enqueue.build_plan(args, rows)[0].filename(args.python) != name_before


def test_shell_arguments_are_quoted_and_smoke_prints_timing(tmp_path):
    source = tmp_path / "captions $(touch SHOULD_NOT_EXIST); 'quoted'.jsonl"
    write_prompts(source, 1)
    task = enqueue.Task("00_test", [["scripts/sample.py", "--prompts", str(source),
                                    "--out", str(tmp_path / "with spaces")]], [source], timed=True)
    body = task.body(Path("/bin/echo"))
    subprocess.run(["bash", "-n"], input=body, text=True, check=True, capture_output=True)
    result = subprocess.run(["bash"], input=body, text=True, check=True, capture_output=True)
    assert str(source) in result.stdout
    assert "smoke_command_0_seconds=" in result.stdout
    assert not (ROOT / "SHOULD_NOT_EXIST").exists()
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_spool_resume_preserves_running_and_done_tasks(tmp_path, monkeypatch):
    args = plan_args(tmp_path, "--stage", "main", "--methods", "ntk2")
    monkeypatch.setattr(enqueue, "SCRATCH", tmp_path)
    tasks = enqueue.build_plan(args, enqueue.read_prompts(args.prompts))
    first = enqueue.write_tasks(args, tasks)
    names = [row["name"] for row in first]
    (args.queue / "pending" / names[0]).rename(args.queue / "running" / f"{names[0]}.testhost")
    (args.queue / "pending" / names[1]).rename(args.queue / "done" / names[1])
    second = enqueue.write_tasks(args, tasks)
    assert [row["state"] for row in second] == ["running", "done"]
    assert list((args.queue / "pending").iterdir()) == []


def test_pilot_static_is_separate_and_requires_sega_outputs(tmp_path):
    args = plan_args(tmp_path, "--stage", "pilot", "--dry-run")
    rows = enqueue.read_prompts(args.prompts)
    tasks = enqueue.build_plan(args, rows)
    sega = next(task for task in tasks if task.label.startswith("30_pilot_sega_"))
    assert len(sega.commands) == 2
    assert sega.commands[1][:3] == ["scripts/calibrate.py", "--mode", "sega-static"]
    assert not any(task.commands[0][task.commands[0].index("--method") + 1] == "sega_static" for task in tasks)
    args.stage = "pilot-static"
    static_tasks = enqueue.build_plan(args, rows)
    requirements = static_tasks[0].requirements
    assert args.sega_static_profile in requirements
    assert sum(path.suffix == ".png" for path in requirements) == 8
    expected = enqueue.image_stem("example.0", 0)
    assert any(path.name == f"{expected}.json" for path in requirements)


def test_main_eval_uses_isolated_main_root(tmp_path):
    args = plan_args(tmp_path, "--stage", "eval", "--dry-run")
    task, = enqueue.build_plan(args, enqueue.read_prompts(args.prompts))
    command = task.commands[0]
    assert command[command.index("--root") + 1] == str(args.out / "main")
    assert "--res" in command
