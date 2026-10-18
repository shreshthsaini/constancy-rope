#!/usr/bin/env python3
# PROVENANCE (re-derived from data on every run; no number is copied from HANDOVER.md or results.md).
# Each 4096^2 image is a JSON sidecar under out_krea/<root>/<method>/4096x4096. sweeproot_* are symlinks into
# sweep/ and pilot/, so images are deduplicated by real path; stale_main_dype holds 4 superseded main/dype images,
# kept out of the table. GPU type: a sidecar is matched to the task-log line with the same prompt, method and
# seconds, then to the fleet_<job>.out interval of that task that contains the sidecar mtime. Six images whose task
# logs were truncated by a re-run are matched through the prompt chunk and seed of the task script plus the same
# time window, and dev3, whose spool logs no host, through the telemetry series busy over each task's span. The
# host's nvidia-smi memory.total names the GPU (97871 MiB: GH200, 189471 MiB: GB200), and every attributed host
# shows a busy GPU when its image was saved. Attention backend: config.attention is a constant string and is
# ignored. The sidecar's implementation_sha256 is recomputed from git at every commit, which pins the
# gained_attention source: commits up to 09e07c2 force sdpa_kernel(FLASH_ATTENTION), and 7405462 onward request
# cuDNN, then FlashAttention, then memory-efficient. c05ebdc sidecars carry the uncommitted tree later committed
# as a1ede1a; 7 main/sega_official images carry an uncommitted tree between d8793d1 and 13d81d8, both cuDNN first,
# and they run at their node's cuDNN-first speed. Outcome: pilot ran forced FA2 on GH200 nodes c610-072, c619-102,
# c637-021 and c637-022 (fleet 1072630), with yarn and dype only on c619-102, except pilot/sega_static and one
# sega_official_static image (cuDNN first). main, mainsb, seed1, sweep, devtm_eta1, devtm_eta2 and devsb ran cuDNN
# first on GH200 (fleets 1072630, 1073495, 1073496, 1073851, 1073853). dev3_* ran cuDNN first on GH200 job 1076857
# (c609-092 and c609-151, not separable per task). dev2_* and main2 ran cuDNN first on the 4-GPU GB200 node
# c672-016 (fleet 1076092). Native 1024^2 Krea: the calibration traces (code 09e07c2, forced FA2, c610-072) and the
# native comparator (06f0c87, cuDNN first, job 1077304 on c610-042 and c610-051). The official DyPE diagnostics
# ran on c637-021 under the default SDPA dispatch of diffusers. The script re-derives and checks all of this.
"""Appendix compute table (tab:compute) and the timing macros cited in prose (num_compute.tex).

Run from the project root:
    OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/tables/compute_table.py
Add --verbose for the per-node and per-root audit. The script only reads data: JSON sidecars, task and fleet
logs, telemetry CSVs, calibration traces and git objects. It writes paper/tables/tab_compute.tex and
paper/tables/num_compute.tex and prints a short summary.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SCRATCH = Path("/scratch/09032/saini_2/constancy_rope")
OUT_KREA = SCRATCH / "out_krea"
LOGS = SCRATCH / "logs"
DEV3_LOGS = SCRATCH / "taskq_dev3" / "logs"
TELEMETRY = SCRATCH / "telemetry"
TASKQ_DONE = SCRATCH / "taskq" / "done"
KREA_CALIB_TRACES = SCRATCH / "calib" / "krea_profile_traces"
NATIVE_COMPARATOR = SCRATCH / "native_comparator"
NATIVE_REPORT = PROJECT / ".agent" / "notes" / "native_cr20" / "report.json"
DIAG_LOGS = {"krea": LOGS / "task_05_diag_official_dype_krea.log", "dev": LOGS / "task_05_diag_official_dype_dev.log"}
TZ = ZoneInfo("America/Chicago")  # fleet logs and telemetry use naive local (Vista) time

# nvidia-smi memory.total identifies the GPU: GH200 96 GB HBM3 versus GB200 186 GB HBM3e.
GPU_BY_MEMORY_MIB = {97871: "GH200", 189471: "GB200"}
BUSY_MIB = 20000  # a FLUX worker holds 36 to 41 GB; idle GPUs show < 1 GB
# Files hashed by crope.runtime.implementation_hash (one definition across the whole history).
IMPL_FILES = [f"crope/{n}" for n in ("__init__.py", "runtime.py", "gain.py", "rope.py", "sega_official.py",
                                     "spectral.py", "utils.py")] + \
             [f"scripts/{n}" for n in ("sample.py", "calibrate.py", "probe_mtf.py")]
FAMILY = {  # config.method -> (order, label used in the table)
    "ntk2": (0, r"\ntk"),
    "yarn": (1, r"\yarn"),
    "dype": (2, r"\dype"),
    "sega": (3, r"\sega{} family"), "sega_official": (3, r"\sega{} family"),
    "sega_static": (3, r"\sega{} family"), "sega_official_static": (3, r"\sega{} family"),
    "sega_ctm": (4, r"\segactm{} family"), "sega_ctm_lag": (4, r"\segactm{} family"),
    "sega_rms_uniform": (4, r"\segactm{} family"), "sega_rms_normalized": (4, r"\segactm{} family"),
    "constancy": (5, r"\constancy{} family"), "constancy_nogate": (5, r"\constancy{} family"),
    "constancy_tm": (5, r"\constancy{} family"),
    "temp_only": (5, r"\constancy{} family"), "alias_temp": (5, r"\constancy{} family"),
    "alias_temp_sref": (5, r"\constancy{} family"),
}
BACKEND_LABEL = {"flash_only": "FA2, forced", "cudnn_first": "cuDNN first", "default": "default dispatch"}
GPU_ORDER = {"GH200": 0, "GB200": 1}
BACKEND_ORDER = {"flash_only": 0, "cudnn_first": 1, "default": 2}


# ---------------------------------------------------------------- small helpers
def quartiles(xs):
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0], xs[0], xs[0]
    q1, q2, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return q1, q2, q3


def local_ts(text, fmt):
    return dt.datetime.strptime(text, fmt).replace(tzinfo=TZ).timestamp()


def clock(t):
    return dt.datetime.fromtimestamp(t, TZ).strftime("%m-%d %H:%M")


def git(*args, data=None):
    return subprocess.run(["git", "-C", str(PROJECT), *args], input=data, capture_output=True, check=True).stdout


def cat_file_batch(specs):
    """Return {spec: bytes or None} for 'commit:path' specs in one git process."""
    out = git("cat-file", "--batch", data="".join(s + "\n" for s in specs).encode())
    result, pos = {}, 0
    for spec in specs:
        end = out.index(b"\n", pos)
        header = out[pos:end].decode()
        pos = end + 1
        if header.endswith(" missing"):
            result[spec] = None
            continue
        size = int(header.split()[2])
        result[spec] = out[pos:pos + size]
        pos += size + 1
    return result


def sdpa_policy(gain_source):
    """Classify gained_attention: forced FlashAttention, or cuDNN-first priority with fallbacks."""
    body = gain_source.split("def gained_attention", 1)[1].split("\ndef ", 1)[0]
    if "CUDNN_ATTENTION" in body and "set_priority=True" in body:
        return "cudnn_first"
    if "sdpa_kernel(SDPBackend.FLASH_ATTENTION)" in body:
        return "flash_only"
    return "other"


# ---------------------------------------------------------------- code provenance
def code_history():
    """Per commit: implementation hash (as crope.runtime computes it), SDPA policy, parent, commit time."""
    commits = []
    for line in git("log", "--all", "--format=%H %P %ct").decode().split("\n"):
        if line.strip():
            parts = line.split()
            commits.append((parts[0], parts[1:-1], int(parts[-1])))
    blobs = cat_file_batch([f"{c}:{p}" for c, _, _ in commits for p in IMPL_FILES])
    history = {}
    for sha, parents, ctime in commits:
        digest = hashlib.sha256()
        for path in IMPL_FILES:
            content = blobs[f"{sha}:{path}"]
            if content is not None:
                digest.update(path.encode())
                digest.update(content)
        gain = blobs[f"{sha}:crope/gain.py"]
        history[sha] = dict(impl=digest.hexdigest(), policy=sdpa_policy(gain.decode()) if gain else "other",
                            parents=parents, time=ctime)
    return history


def resolve_policy(revision, impl, history):
    """SDPA policy of the code that produced a sidecar, with how it was established."""
    by_impl = defaultdict(set)
    for sha, h in history.items():
        by_impl[h["impl"]].add(sha)
    if revision in history and history[revision]["impl"] == impl:
        return history[revision]["policy"], "clean tree at recorded revision"
    if impl in by_impl:
        policies = {history[s]["policy"] for s in by_impl[impl]}
        if len(policies) == 1:
            return policies.pop(), "uncommitted tree identical to a later commit"
    if revision in history:
        children = [s for s, h in history.items() if revision in h["parents"]]
        policies = {history[revision]["policy"]} | {history[c]["policy"] for c in children}
        if len(policies) == 1:
            return policies.pop(), "uncommitted tree; recorded revision and its child agree"
    return "other", "unresolved"


# ---------------------------------------------------------------- runtime provenance
def load_telemetry():
    series = {}
    for path in sorted(TELEMETRY.glob("*.csv")):
        m = re.match(r"(c\d{3}-\d{3})_job(\d+)_", path.name)
        if not m:
            continue
        samples = []
        with open(path) as handle:
            for row in csv.DictReader(handle):
                samples.append((local_ts(row["timestamp"].split(".")[0], "%Y/%m/%d %H:%M:%S"),
                                float(row["mem_used_mib"]), int(float(row["mem_total_mib"])),
                                float(row["util_pct"]), float(row["power_w"])))
        if samples:
            series[(m.group(1), m.group(2))] = samples
    return series


def gpu_by_host(telemetry):
    seen = defaultdict(set)
    for (host, _), samples in telemetry.items():
        seen[host] |= {GPU_BY_MEMORY_MIB.get(s[2], f"unknown({s[2]} MiB)") for s in samples}
    return {h: (v.pop() if len(v) == 1 else "ambiguous") for h, v in seen.items()}


def parse_fleets(telemetry):
    """Task intervals from fleet logs: task -> [(host, start, end, job, rc)]."""
    pattern = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) (c\d{3}-\d{3})(?:-g\d+)? (start|end) (\S+?)\.sh"
                         r"(?: rc=(-?\d+))?")
    intervals = defaultdict(list)
    for path in sorted(LOGS.glob("fleet_*.out")):
        job = re.search(r"fleet_(\d+)", path.name).group(1)
        running = {}
        for line in open(path, errors="replace"):
            m = pattern.match(line)
            if not m:
                continue
            t = local_ts(m.group(1), "%Y-%m-%dT%H:%M:%S")
            key = (m.group(2), m.group(4), line.split()[1])
            if m.group(3) == "start":
                running[key] = t
            elif key in running:
                intervals[m.group(4)].append((m.group(2), running.pop(key), t, job, m.group(5)))
        for (host, task, _), start in running.items():  # killed with the job: cap at the job's telemetry end
            end = telemetry.get((host, job), [(float("inf"),)])[-1][0]
            intervals[task].append((host, start, end, job, "open"))
    return intervals


def parse_saved_lines():
    """(prompt_id, method) -> [(task, seconds)] from every per-task log."""
    pattern = re.compile(r"^saved (\S+) (\S+) (\S+) seconds=([\d.]+)")
    saved = defaultdict(list)
    sources = [(p.name[len("task_"):-len(".log")], p) for p in LOGS.glob("task_*.log")]
    sources += [("dev3/" + p.name[:-len(".sh.log")], p) for p in DEV3_LOGS.glob("*.sh.log")]
    for task, path in sources:
        for line in open(path, errors="replace"):
            m = pattern.match(line)
            if m:
                saved[(m.group(1), m.group(2))].append((task, float(m.group(4))))
    return saved


def parse_task_scripts():
    """Task name -> (output directory, method, seed, set of prompt ids) from each spooled sample.py call."""
    prompt_ids = {}
    targets = {}
    for path in list(SCRATCH.glob("taskq/*/*.sh")) + list(SCRATCH.glob("taskq_dev3/done/*.sh")):
        text = path.read_text(errors="replace")
        flags = {k: re.search(rf"--{k} (\S+)", text) for k in ("out", "method", "prompts", "start", "end", "seed")}
        if "sample.py" not in text or not all(flags.values()):
            continue
        source = flags["prompts"].group(1)
        if source not in prompt_ids:
            prompt_ids[source] = [json.loads(line)["id"] for line in open(source) if line.strip()]
        chunk = prompt_ids[source][int(flags["start"].group(1)):int(flags["end"].group(1))]
        name = ("dev3/" if "taskq_dev3" in str(path) else "") + path.stem
        targets[name] = (str(Path(flags["out"].group(1))), flags["method"].group(1), int(flags["seed"].group(1)),
                         frozenset(chunk))
    return targets


def telemetry_hosts(t0, t1, telemetry):
    """(host, job) series that cover at least 90% of [t0, t1] with a busy GPU in at least 80% of samples."""
    hits = []
    for (host, job), samples in telemetry.items():
        overlap = min(t1, samples[-1][0]) - max(t0, samples[0][0])
        if overlap < 0.9 * (t1 - t0):
            continue
        window = [s for s in samples if t0 <= s[0] <= t1]
        if window and sum(s[1] > BUSY_MIB for s in window) >= 0.8 * len(window):
            hits.append((host, job))
    return hits


def busy_at(host, job, t, telemetry, slack=90):
    samples = telemetry.get((host, job), [])
    return any(abs(s[0] - t) <= slack and s[1] > BUSY_MIB for s in samples)


# ---------------------------------------------------------------- data
def load_sidecars():
    """Unique 4096^2 sidecars under out_krea, deduplicated by real path (sweeproot_* are symlink farms)."""
    paths = sorted(OUT_KREA.glob("*/*/4096x4096/*.json")) + sorted(OUT_KREA.glob("stale_main_dype/*.json"))
    rows, seen = [], set()
    for path in paths:
        real = path.resolve()
        if real in seen:
            continue
        seen.add(real)
        data = json.loads(real.read_text())
        cfg = data["config"]
        rel = real.relative_to(OUT_KREA).parts
        # sample.py --out names the resolution directory; stale_main_dype holds images moved out of main/dype.
        out_dir = str(OUT_KREA / "main" / "dype" / "4096x4096") if rel[0] == "stale_main_dype" else str(real.parent)
        rows.append(dict(root=rel[0], stale=rel[0] == "stale_main_dype", method=cfg["method"], out_dir=out_dir,
                         prompt_id=cfg["prompt_id"], seed=cfg["seed"], res=cfg["res"], steps=cfg["steps"],
                         batch=cfg["batch_size"], torch=cfg["torch"], revision=cfg["code_revision"],
                         impl=cfg["implementation_sha256"], seconds=float(data["seconds"]),
                         peak=data["peak_memory_bytes"], complete=data["complete"],
                         scope=data["seconds_scope"], png=data["png_sha256"], mtime=real.stat().st_mtime))
    return rows


def attribute(rows, intervals, saved, scripts, telemetry, gpu):
    """Host and GPU per image.

    Primary: sidecar -> the task-log line with the same prompt, method and seconds -> the fleet interval of that
    task containing the sidecar mtime. Fallback when a re-run truncated the task log: tasks whose script writes
    the same directory and method, again by fleet interval. dev3 (its spool logs no host): the telemetry series
    that were busy over the whole span of the task's images.
    """
    by_target = defaultdict(set)
    for task, (out, method, seed, chunk) in scripts.items():
        for prompt in chunk:
            by_target[(out, method, seed, prompt)].add(task)
    unresolved_tasks = defaultdict(list)
    for r in rows:
        r["pairs"], r["via"] = [], None
        tasks = {task for task, sec in saved.get((r["prompt_id"], r["method"]), [])
                 if abs(sec - r["seconds"]) < 6e-4}
        via = "fleet log"
        if not tasks:
            tasks = by_target.get((r["out_dir"], r["method"], r["seed"], r["prompt_id"]), set())
            via = "fleet log, chunk and time window"
        r["task"] = sorted(tasks)[0] if len(tasks) == 1 else ("ambiguous" if tasks else None)
        hits = [iv for task in tasks for iv in intervals.get(task, [])
                if iv[1] - 5 <= r["mtime"] <= iv[2] + 120]
        if len(hits) > 1:
            hits = [iv for iv in hits if iv[4] == "0"] or hits
        if len({(iv[0], iv[3]) for iv in hits}) == 1:
            r["pairs"], r["via"] = [(hits[0][0], hits[0][3])], via
        elif len(tasks) == 1 and not any(t in intervals for t in tasks):
            unresolved_tasks[r["task"]].append(r)
    for task, members in unresolved_tasks.items():
        span = (min(m["mtime"] for m in members) - 60, max(m["mtime"] for m in members))
        hits = telemetry_hosts(*span, telemetry)
        if hits and len({gpu.get(h) for h, _ in hits}) == 1:
            for m in members:
                m["pairs"], m["via"] = sorted(hits), "telemetry span"
    for r in rows:
        types = {gpu.get(h, "unknown") for h, _ in r["pairs"]}
        r["gpu"] = types.pop() if len(types) == 1 else "unknown"
        r["host"] = "|".join(h for h, _ in r["pairs"]) or "unknown"
        r["telemetry_ok"] = bool(r["pairs"]) and any(busy_at(h, j, r["mtime"], telemetry) for h, j in r["pairs"])


def native_runs(history):
    """Native 1024^2 Krea timings: forced-flash calibration traces and the cuDNN-first native comparator."""
    import numpy as np
    runs = []
    for path in sorted(KREA_CALIB_TRACES.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["config_json"]))
        policy, how = resolve_policy(cfg["code_revision"], cfg["implementation_sha256"], history)
        runs.append(dict(kind="calibration", seconds=float(z["seconds"]), res=cfg["res"], policy=policy, how=how,
                         peak=None, process="calibration", mtime=path.stat().st_mtime))
    for path in sorted(NATIVE_COMPARATOR.glob("job*/chunk*/*/1024x1024/*.json")):
        data = json.loads(path.read_text())
        cfg = data["config"]
        policy, how = resolve_policy(cfg["code_revision"], cfg["implementation_sha256"], history)
        runs.append(dict(kind="native comparator", seconds=float(data["seconds"]), res=cfg["res"], policy=policy,
                         how=how, peak=data["peak_memory_bytes"], job=path.parts[len(NATIVE_COMPARATOR.parts)][3:],
                         process=str(path.parents[2]), mtime=path.stat().st_mtime))
    first = {}
    for x in runs:  # one worker process per calibration task and per comparator chunk
        if x["process"] not in first or x["mtime"] < first[x["process"]]["mtime"]:
            first[x["process"]] = x
    for x in runs:
        x["first"] = first[x["process"]] is x
    return runs


def calibration_is_krea():
    """The Krea calibration task is the one that exports CROPE_TRANSFORMER; return its fleet task name."""
    for path in sorted(TASKQ_DONE.glob("10_calibration_native_*.sh")):
        if "CROPE_TRANSFORMER=" in path.read_text() and "Krea" in path.read_text():
            return path.stem
    return None


def official_dype(intervals):
    out = {}
    for tag, path in DIAG_LOGS.items():
        text = open(path, errors="replace").read().replace("\r", "\n")
        m = re.search(r"DIAG_DONE (\S+) (\d+) ([\d.]+)", text)
        task = path.stem[len("task_"):]
        ok = [iv for iv in intervals.get(task, []) if iv[4] == "0"]
        out[tag] = dict(seconds=float(m.group(3)), res=int(m.group(2)), host=ok[-1][0] if ok else None,
                        start=ok[-1][1] if ok else None)
    return out


# ---------------------------------------------------------------- analysis
def analyse():
    history = code_history()
    telemetry = load_telemetry()
    gpu = gpu_by_host(telemetry)
    intervals = parse_fleets(telemetry)
    saved = parse_saved_lines()
    scripts = parse_task_scripts()
    rows = load_sidecars()
    for r in rows:
        r["policy"], r["policy_how"] = resolve_policy(r["revision"], r["impl"], history)
        if r["method"] not in FAMILY:
            sys.exit(f"method {r['method']} has no family; add it to FAMILY")
        r["family"] = FAMILY[r["method"]]
    attribute(rows, intervals, saved, scripts, telemetry, gpu)
    valid = [r for r in rows if r["complete"] and r["steps"] == 28 and r["res"] == "4096x4096" and r["batch"] == 1]
    return dict(history=history, telemetry=telemetry, gpu=gpu, intervals=intervals, rows=rows, valid=valid,
                campaign=[r for r in valid if not r["stale"]])


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def slow_nodes(campaign, threshold=1.08):
    """GH200 hosts whose median under some SDPA policy exceeds that policy's all-host median by over 8%."""
    by_host, pooled = defaultdict(list), defaultdict(list)
    for r in campaign:
        if r["gpu"] == "GH200":
            by_host[(r["host"], r["policy"])].append(r["seconds"])
            pooled[r["policy"]].append(r["seconds"])
    return sorted({h for (h, pol), v in by_host.items() if median(v) > threshold * median(pooled[pol])})


def generation_samples(images, telemetry):
    """Busy telemetry samples taken while one of the images was being generated on that host."""
    import bisect
    windows = defaultdict(list)
    for r in images:
        for key in r["pairs"]:
            windows[key].append((r["mtime"] - r["seconds"] + 5, r["mtime"] - 5))
    out = []
    for key, spans in windows.items():
        union = []
        for a, b in sorted(spans):
            if union and a <= union[-1][1]:
                union[-1][1] = max(union[-1][1], b)
            else:
                union.append([a, b])
        starts = [a for a, _ in union]
        for s in telemetry.get(key, []):
            i = bisect.bisect_right(starts, s[0]) - 1
            if i >= 0 and s[0] <= union[i][1] and s[1] > BUSY_MIB:
                out.append(s)
    return out


def compute(A):
    camp, telemetry, history, intervals = A["campaign"], A["telemetry"], A["history"], A["intervals"]
    slow = slow_nodes(camp)
    for r in camp:
        r["block"] = r["gpu"] if r["host"] not in slow else "GH200 slow"
    R = dict(slow=slow, nodes=defaultdict(set))
    for r in camp:
        R["nodes"][r["gpu"]] |= set(r["host"].split("|"))
    typical = [r for r in camp if r["block"] == "GH200"]
    gh_c = [r["seconds"] for r in typical if r["policy"] == "cudnn_first"]
    gh_f = [r["seconds"] for r in typical if r["policy"] == "flash_only"]
    gh_c_all = [r["seconds"] for r in camp if r["gpu"] == "GH200" and r["policy"] == "cudnn_first"]
    gh_f_all = [r["seconds"] for r in camp if r["gpu"] == "GH200" and r["policy"] == "flash_only"]
    bw = [r["seconds"] for r in camp if r["gpu"] == "GB200"]
    R.update(gh_c=quartiles(gh_c), gh_f=quartiles(gh_f), n_gh_c=len(gh_c), n_gh_f=len(gh_f),
             gh_c_all=median(gh_c_all), gh_f_all=median(gh_f_all), bw=quartiles(bw), n_bw=len(bw))
    # flash versus cuDNN-first on the same node and method
    cell = defaultdict(lambda: defaultdict(list))
    for r in camp:
        if r["gpu"] == "GH200":
            cell[(r["host"], r["method"])][r["policy"]].append(r["seconds"])
    pairs = {k: median(v["flash_only"]) / median(v["cudnn_first"]) for k, v in cell.items()
             if v.get("flash_only") and v.get("cudnn_first")}
    R.update(flash_pairs=pairs, flash_ratio=median(list(pairs.values())))
    # stricter check: cuDNN side limited to the first cuDNN-first commit, whose only change is the backend line
    first_cudnn = min((h["time"], sha) for sha, h in history.items() if h["policy"] == "cudnn_first")[1]
    strict = defaultdict(lambda: defaultdict(list))
    for r in camp:
        if r["gpu"] == "GH200" and (r["policy"] == "flash_only" or r["revision"] == first_cudnn):
            strict[r["host"]][r["policy"]].append(r["seconds"])
    R["flash_strict"] = {h: (median(v["flash_only"]) / median(v["cudnn_first"]), len(v["flash_only"]),
                             len(v["cudnn_first"])) for h, v in strict.items()
                         if v.get("flash_only") and v.get("cudnn_first")}
    R["first_cudnn"] = first_cudnn[:7]
    # DyPE versus NTK under cuDNN-first on the same node
    dy = {h: median(cell[(h, "dype")]["cudnn_first"]) / median(cell[(h, "ntk2")]["cudnn_first"])
          for h, m in cell if m == "dype" and len(cell[(h, "dype")]["cudnn_first"]) >= 5
          and len(cell[(h, "ntk2")]["cudnn_first"]) >= 5}
    R.update(dype_ratio=median(list(dy.values())), dype_nodes=dy)
    # spread between method families on one node (cuDNN-first, families with >= 5 images on that node)
    fam = defaultdict(lambda: defaultdict(list))
    for r in camp:
        if r["policy"] == "cudnn_first" and "|" not in r["host"]:
            fam[r["host"]][r["family"]].append(r["seconds"])
    spread = {h: max(median(v) for v in f.values() if len(v) >= 5) / min(median(v) for v in f.values() if len(v) >= 5)
              for h, f in fam.items() if sum(len(v) >= 5 for v in f.values()) >= 2}
    R.update(family_spread=max(spread.values()), family_spread_nodes=spread)
    # the slow node
    sl = [r for r in camp if r["block"] == "GH200 slow"]
    R.update(slow_c=median([r["seconds"] for r in sl if r["policy"] == "cudnn_first"]),
             slow_f=median([r["seconds"] for r in sl if r["policy"] == "flash_only"]))
    R["slow_pct_c"] = 100 * (R["slow_c"] / R["gh_c"][1] - 1)
    R["slow_pct_f"] = 100 * (R["slow_f"] / R["gh_f"][1] - 1)
    # telemetry while generating
    samples = {"GH200": generation_samples(typical, telemetry), "GH200 slow": generation_samples(sl, telemetry),
               "GB200": generation_samples([r for r in camp if r["gpu"] == "GB200"], telemetry)}
    R["telemetry"] = {k: dict(n=len(v), util_med=median([s[3] for s in v]),
                              util_mean=statistics.fmean([s[3] for s in v]) if v else float("nan"),
                              power_med=median([s[4] for s in v])) for k, v in samples.items()}
    # totals and memory
    R.update(n_imgs=len(camp), gpu_hours=sum(r["seconds"] for r in camp) / 3600,
             gpu_hours_by={g: sum(r["seconds"] for r in camp if r["gpu"] == g) / 3600 for g in ("GH200", "GB200")},
             peak=median([r["peak"] for r in camp]), peak_by={g: median([r["peak"] for r in camp if r["gpu"] == g])
                                                               for g in ("GH200", "GB200")},
             n_stale=len(A["valid"]) - len(camp), per_root=Counter(r["root"] for r in camp))
    # native 1024^2 Krea
    krea_task = calibration_is_krea()
    calib_hosts = sorted({iv[0] for iv in intervals.get(krea_task, []) if iv[4] == "0"})
    native = native_runs(history)
    nat_f = [x for x in native if x["kind"] == "calibration"]
    nat_c = [x for x in native if x["kind"] == "native comparator"]
    comp_hosts = sorted({h for (h, j) in telemetry if any(j == x["job"] for x in nat_c)})
    R["native"] = {
        "flash_only": dict(runs=nat_f, hosts=calib_hosts, policies={x["policy"] for x in nat_f},
                           gpu={A["gpu"].get(h) for h in calib_hosts}, q=quartiles([x["seconds"] for x in nat_f])),
        "cudnn_first": dict(runs=nat_c, hosts=comp_hosts, policies={x["policy"] for x in nat_c},
                            gpu={A["gpu"].get(h) for h in comp_hosts}, q=quartiles([x["seconds"] for x in nat_c]),
                            peak=median([x["peak"] for x in nat_c]))}
    for d in R["native"].values():
        d["warmup"] = median([x["seconds"] for x in d["runs"] if x["first"]]) - \
            median([x["seconds"] for x in d["runs"] if not x["first"]])
    R["official"] = official_dype(intervals)
    for v in R["official"].values():
        v["gpu"] = A["gpu"].get(v["host"])
    report = json.loads(NATIVE_REPORT.read_text()) if NATIVE_REPORT.exists() else {}
    loads = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "model_loading_seconds" and isinstance(v, (int, float)):
                    loads.append(float(v))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(report)
    R["model_load"] = loads
    return R


# ---------------------------------------------------------------- output
def f1(x):
    return f"{x:.1f}"


def table_rows(A, R):
    """(block header, [(policy, label, rows)]) in display order."""
    camp = A["campaign"]
    blocks = []
    for block, header in (("GH200", f"$4096^2$, GH200 ({len(R['nodes']['GH200']) - len(R['slow'])} nodes)"),
                          ("GH200 slow", f"$4096^2$, GH200 node {', '.join(R['slow'])} (listed separately)"),
                          ("GB200", f"$4096^2$, GB200 ({len(R['nodes']['GB200'])} node, "
                                    f"{GPUS_PER_GB200_NODE} GPUs)")):
        members = [r for r in camp if r["block"] == block]
        groups = defaultdict(list)
        for r in members:
            key = (r["policy"], (0, "all")) if block == "GH200 slow" else (r["policy"], r["family"])
            groups[key].append(r)
        lines = []
        for (policy, family), rows in sorted(groups.items(), key=lambda kv: (BACKEND_ORDER[kv[0][0]], kv[0][1][0])):
            if block == "GH200 slow":
                labels = sorted({r["family"] for r in rows})
                label = ", ".join(l for _, l in labels) if len(labels) <= 2 else "all methods"
            else:
                label = family[1]
            lines.append((policy, label, rows))
        if lines:
            blocks.append((header, lines))
    return blocks


GPUS_PER_GB200_NODE = None  # set from the fleet log header in main()


def render_table(A, R):
    out = [r"% Generated by paper/tables/compute_table.py. Do not edit by hand.",
           r"\begin{table}[t]", r"\centering", r"\small"]
    gh_c, gh_f, bw = R["gh_c"][1], R["gh_f"][1], R["bw"][1]
    t = R["telemetry"]
    caption = (
        r"\textbf{Sampling cost is set by the attention kernel and the GPU, not by the method.} "
        rf"On the same GH200 node and method, forced FlashAttention-2 (FA2) takes {R['flash_ratio']:.2f} times as long "
        rf"as cuDNN-first SDPA ({gh_f:.0f}\,s versus {gh_c:.0f}\,s per $4096^2$ image), and one GB200 GPU is "
        rf"{gh_c / bw:.2f} times faster than a GH200 under cuDNN-first SDPA ({bw:.0f}\,s). "
        rf"Method families on one node differ by at most {100 * (R['family_spread'] - 1):.1f}\%. "
        rf"Node {', '.join(R['slow'])} ran {R['slow_pct_c']:.0f}\% slower than the other GH200 nodes under both kernels "
        rf"(median power {t['GH200 slow']['power_med']:.0f}\,W versus {t['GH200']['power_med']:.0f}\,W while generating), "
        r"so it is listed separately; the forced-FA2 \yarn{} and \dype{} pilots ran only there. "
        r"$n$ counts images. Seconds cover prompt encoding, 28 denoising steps and VAE decoding, without model "
        r"loading or PNG writing; IQR is the interquartile range; peak is \texttt{torch.cuda.max\_memory\_allocated}; "
        r"GPU-h sums the per-image seconds. The SDPA column gives the requested backend policy (cuDNN first: cuDNN, "
        r"then FlashAttention, then memory-efficient); the dispatched kernel was not logged. "
        r"\sega{} family: released and paper-equation \sega{} with their static ablations. "
        r"\segactm{} family: \segactm{} and its dev-only allocation controls. "
        r"\constancy{} family: \constancy{} and its dev sweeps, \constancytm{}, the temperature-only ablations, "
        r"\aliastemp{} and \aliastemps{}. The native rows run \krea{} without RoPE changes; the first image of each "
        rf"worker process carries a one-off warm-up ({R['native']['cudnn_first']['warmup']:.1f}\,s with cuDNN first, "
        rf"{R['native']['flash_only']['warmup']:.1f}\,s with FA2), which widens the cuDNN-first IQR at $1024^2$. "
        r"The official \dype{} script uses the default SDPA dispatch of diffusers, and its single timing also "
        r"includes writing the PNG.")
    out += [rf"\caption{{{caption}}}", r"\label{tab:compute}",
            r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
            r"SDPA & Methods & $n$ & Nodes & Median (s) & IQR (s) & Peak (GB) & GPU-h \\", r"\midrule"]
    first = True
    for header, lines in table_rows(A, R):
        if not first:
            out.append(r"\addlinespace[3pt]")
        first = False
        out.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{header}}}}} \\")
        for policy, label, rows in lines:
            q1, q2, q3 = quartiles([r["seconds"] for r in rows])
            nodes = len({h for r in rows for h in r["host"].split("|")})
            out.append(rf"{BACKEND_LABEL[policy]} & {label} & {len(rows)} & {nodes} & {f1(q2)} & {f1(q1)}--{f1(q3)} "
                       rf"& {median([r['peak'] for r in rows]) / 1e9:.1f} & {sum(r['seconds'] for r in rows) / 3600:.1f} \\")
    nat = R["native"]
    out += [r"\addlinespace[3pt]", rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{native $1024^2$, GH200, \krea{{}}}}}} \\"]
    for policy in ("flash_only", "cudnn_first"):
        d = nat[policy]
        q1, q2, q3 = d["q"]
        peak = f"{d['peak'] / 1e9:.1f}" if d.get("peak") else ""
        out.append(rf"{BACKEND_LABEL[policy]} & no RoPE change & {len(d['runs'])} & {len(d['hosts'])} & {q2:.2f} & "
                   rf"{q1:.2f}--{q3:.2f} & {peak} & \\")
    off = R["official"]["krea"]
    out += [r"\addlinespace[3pt]", r"\multicolumn{8}{@{}l}{\emph{$4096^2$, GH200, official \dype{} script}} \\",
            rf"default & \dype{{}} (official code) & 1 & 1 & {f1(off['seconds'])} & & & \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def macros(A, R):
    t, nat = R["telemetry"], R["native"]
    gh_c, gh_f, bw = R["gh_c"][1], R["gh_f"][1], R["bw"][1]
    m = [
        ("NcompGHcudnnSec", f"{gh_c:.0f}", f"median s/img, 4096^2 Krea, GH200 cuDNN-first, n={R['n_gh_c']}, "
                                           f"excluding node {','.join(R['slow'])}"),
        ("NcompGHcudnnQlo", f"{R['gh_c'][0]:.0f}", "first quartile of the row above"),
        ("NcompGHcudnnQhi", f"{R['gh_c'][2]:.0f}", "third quartile of the row above"),
        ("NcompGHcudnnSecAllNodes", f"{R['gh_c_all']:.0f}", "same, all GH200 nodes pooled"),
        ("NcompGHflashSec", f"{gh_f:.0f}", f"median s/img, GH200 forced FA2, n={R['n_gh_f']}, excluding the slow node"),
        ("NcompGHflashSecAllNodes", f"{R['gh_f_all']:.0f}", "same, all GH200 nodes pooled"),
        ("NcompFlashRatio", f"{R['flash_ratio']:.2f}", f"forced FA2 / cuDNN-first time, median over "
                                                      f"{len(R['flash_pairs'])} same-node same-method pairs"),
        ("NcompFlashSlowdownPct", f"{100 * (R['flash_ratio'] - 1):.0f}", "forced FA2 takes this % longer"),
        ("NcompCudnnSavingPct", f"{100 * (1 - 1 / R['flash_ratio']):.0f}", "cuDNN-first saves this % of the FA2 time"),
        ("NcompBWSec", f"{bw:.0f}", f"median s/img per GPU, GB200, cuDNN-first, n={R['n_bw']}"),
        ("NcompBWSpeedup", f"{gh_c / bw:.2f}", "GH200 cuDNN-first median / GB200 median, per GPU"),
        ("NcompBWGPUs", f"{GPUS_PER_GB200_NODE}", "GPUs per GB200 node"),
        ("NcompBWNode", ",".join(sorted(R["nodes"]["GB200"])), "the GB200 node"),
        ("NcompPeakGB", f"{R['peak'] / 1e9:.1f}", "median peak allocated memory, 4096^2, GB = 1e9 bytes"),
        ("NcompPeakGiB", f"{R['peak'] / 2**30:.1f}", "same in GiB"),
        ("NcompNativeSec", f"{nat['cudnn_first']['q'][1]:.1f}",
         f"median s/img, native 1024^2 Krea, GH200 cuDNN-first, n={len(nat['cudnn_first']['runs'])}"),
        ("NcompNativeFlashSec", f"{nat['flash_only']['q'][1]:.1f}",
         f"median s/img, native 1024^2 Krea, GH200 forced FA2 (calibration), n={len(nat['flash_only']['runs'])}"),
        ("NcompNativePeakGB", f"{nat['cudnn_first']['peak'] / 1e9:.1f}", "peak allocated memory at 1024^2"),
        ("NcompHiResCostRatio", f"{gh_c / nat['cudnn_first']['q'][1]:.0f}",
         "4096^2 / 1024^2 time, both GH200 cuDNN-first"),
        ("NcompOfficialDypeSec", f"{R['official']['krea']['seconds']:.0f}",
         f"official DyPE script, Krea, 4096^2, n=1, node {R['official']['krea']['host']}, includes PNG write"),
        ("NcompOfficialDypeDevSec", f"{R['official']['dev']['seconds']:.0f}",
         f"official DyPE script, FLUX.1-dev weights, 4096^2, n=1, node {R['official']['dev']['host']}, "
         f"includes PNG write"),
        ("NcompDypeOverheadPct", f"{100 * (R['dype_ratio'] - 1):.1f}",
         f"DyPE vs NTK, cuDNN-first, same node, median over {len(R['dype_nodes'])} nodes (one decimal: below 1%)"),
        ("NcompFamilySpreadPct", f"{100 * (R['family_spread'] - 1):.1f}",
         "largest max/min spread of family medians on one node, cuDNN-first (one decimal: below 2%)"),
        ("NcompSlowNode", ",".join(R["slow"]), "GH200 node slower under both kernels"),
        ("NcompSlowNodePct", f"{R['slow_pct_c']:.0f}", "its cuDNN-first slowdown vs the other GH200 nodes"),
        ("NcompSlowNodeCudnnSec", f"{R['slow_c']:.0f}", "its cuDNN-first median s/img"),
        ("NcompSlowNodeFlashSec", f"{R['slow_f']:.0f}", "its forced-FA2 median s/img (YaRN and DyPE pilots)"),
        ("NcompSlowNodePowerW", f"{t['GH200 slow']['power_med']:.0f}", "its median power while generating"),
        ("NcompGHPowerW", f"{t['GH200']['power_med']:.0f}", "median power while generating, other GH200 nodes"),
        ("NcompUtilPct", f"{t['GH200']['util_med']:.0f}", "median nvidia-smi utilization while generating, GH200"),
        ("NcompUtilMeanPct", f"{t['GH200']['util_mean']:.0f}", "mean of the same samples"),
        ("NcompNimgs", f"{R['n_imgs']}", "retained 4096^2 Krea images with sidecars, all campaigns, deduplicated"),
        ("NcompGPUhours", f"{R['gpu_hours']:.0f}", "sum of their per-image seconds / 3600 (excludes loading)"),
        ("NcompGPUhoursGH", f"{R['gpu_hours_by']['GH200']:.0f}", "the GH200 share"),
        ("NcompGPUhoursBW", f"{R['gpu_hours_by']['GB200']:.0f}", "the GB200 share"),
        ("NcompNnodesGH", f"{len(R['nodes']['GH200'])}", "distinct GH200 nodes that generated retained images"),
    ]
    if R["model_load"]:
        m.append(("NcompModelLoadSec", f"{median(R['model_load']):.0f}",
                  f"model load per worker process, native comparator, n={len(R['model_load'])}"))
    for name, _, _ in m:
        assert re.fullmatch(r"Ncomp[A-Za-z]+", name), name
    lines = [r"% Generated by paper/tables/compute_table.py. Do not edit by hand.",
             r"% Re-derived from sidecars, task and fleet logs, telemetry and git objects; see the script header."]
    lines += [rf"\newcommand{{\{name}}}{{{value}}}  % {note}" for name, value, note in m]
    return "\n".join(lines) + "\n", m


def audit(A, R):
    rows, camp = A["rows"], A["campaign"]
    print("\n== audit")
    print("attribution:", dict(Counter(r["via"] for r in rows)), "| telemetry-consistent:",
          sum(r["telemetry_ok"] for r in rows), "of", len(rows))
    print("policy provenance:", dict(Counter(f"{r['policy']}: {r['policy_how']}" for r in rows)))
    hm = defaultdict(list)
    for r in camp:
        hm[(r["host"], r["gpu"], r["policy"])].append(r["seconds"])
    for k, v in sorted(hm.items()):
        q1, q2, q3 = quartiles(v)
        print(f"  host {k[0]:>18} {k[1]} {k[2]:>11} n={len(v):4d} median={q2:6.1f} IQR={q1:6.1f}-{q3:6.1f}")
    off = [r for r in camp if abs(r["seconds"] / median(hm[(r["host"], r["gpu"], r["policy"])]) - 1) > 0.05]
    print("images more than 5% off their node median:", len(off))
    print("flash/cuDNN pairs:", {f"{h}/{m}": round(v, 3) for (h, m), v in sorted(R["flash_pairs"].items())})
    print(f"flash/cuDNN per node, cuDNN side at {R['first_cudnn']} only (ratio, n_fa2, n_cudnn):",
          {h: (round(a, 3), b, c) for h, (a, b, c) in sorted(R["flash_strict"].items())})
    print("DyPE/NTK per node:", {h: round(v, 4) for h, v in sorted(R["dype_nodes"].items())})
    print("family spread per node:", {h: round(v, 4) for h, v in sorted(R["family_spread_nodes"].items())})
    print("telemetry while generating:", {k: {a: round(b, 1) for a, b in v.items()} for k, v in R["telemetry"].items()})
    for policy, d in R["native"].items():
        print(f"native {policy}: hosts={d['hosts']} gpu={d['gpu']} policies={d['policies']} "
              f"seconds={[round(x['seconds'], 2) for x in d['runs']]}")
    print("official DyPE:", R["official"])
    print("per root:", dict(R["per_root"]), "| stale excluded:", R["n_stale"])
    by_root = defaultdict(list)
    for r in A["rows"]:
        by_root[r["root"]].append(r)
    for root, v in sorted(by_root.items()):
        print(f"  {root:>26}: gpu={sorted({r['gpu'] for r in v})} policy={dict(Counter(r['policy'] for r in v))} "
              f"jobs={sorted({j for r in v for _, j in r['pairs']})} hosts={sorted({h for r in v for h, _ in r['pairs']})}")
    odd = [r for r in A["rows"] if r["policy_how"] != "clean tree at recorded revision"]
    print("non-clean trees:", dict(Counter((r["revision"][:7], r["impl"][:8], r["root"], r["method"], r["host"])
                                          for r in odd)))


def main():
    global GPUS_PER_GB200_NODE
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verbose", action="store_true", help="print the per-node and per-root audit")
    args = parser.parse_args()
    A = analyse()
    scopes = {r["scope"] for r in A["rows"]}
    if len(scopes) != 1:
        sys.exit(f"sidecars disagree on the timing scope: {scopes}")
    unknown = [r for r in A["rows"] if r["gpu"] == "unknown" or r["policy"] == "other"]
    if unknown:
        sys.exit(f"{len(unknown)} images without a GPU type or SDPA policy; run with --verbose")
    headers = Counter()
    for path in LOGS.glob("fleet_*.out"):
        for line in open(path, errors="replace"):
            m = re.match(r"^(c\d{3}-\d{3}) gpus=(\d+)", line)
            if m and A["gpu"].get(m.group(1)) == "GB200":
                headers[int(m.group(2))] += 1
    GPUS_PER_GB200_NODE = headers.most_common(1)[0][0]
    R = compute(A)
    (HERE / "tab_compute.tex").write_text(render_table(A, R))
    text, m = macros(A, R)
    (HERE / "num_compute.tex").write_text(text)
    v = {name: value for name, value, _ in m}
    print(f"{R['n_imgs']} retained 4096^2 Krea images ({R['n_stale']} superseded excluded), "
          f"{v['NcompGPUhours']} GPU-h of sampling; every image has a node and an SDPA policy")
    print(f"GH200 cuDNN-first {v['NcompGHcudnnSec']} s/img (n={R['n_gh_c']}), forced FA2 {v['NcompGHflashSec']} s/img "
          f"(n={R['n_gh_f']}); same-node FA2/cuDNN ratio {v['NcompFlashRatio']} over {len(R['flash_pairs'])} pairs")
    print(f"GB200 {v['NcompBWSec']} s/img per GPU (n={R['n_bw']}), {v['NcompBWSpeedup']}x GH200; "
          f"peak {v['NcompPeakGB']} GB")
    print(f"slow node {v['NcompSlowNode']}: +{v['NcompSlowNodePct']}% "
          f"({v['NcompSlowNodeCudnnSec']} s cuDNN-first, {v['NcompSlowNodeFlashSec']} s FA2)")
    print(f"native 1024^2: {v['NcompNativeSec']} s cuDNN-first, {v['NcompNativeFlashSec']} s FA2; "
          f"official DyPE script {v['NcompOfficialDypeSec']} s (Krea), {v['NcompOfficialDypeDevSec']} s (FLUX.1-dev)")
    print(f"wrote {HERE / 'tab_compute.tex'} and {HERE / 'num_compute.tex'}")
    if args.verbose:
        audit(A, R)


if __name__ == "__main__":
    main()
