"""Resumable evaluation with explicit scorer status and prompt bootstrap."""

from __future__ import annotations

from collections import defaultdict
import csv
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from .image import linear_luminance, octave_profile, profile_error, tiling_metrics
from .io import digest, discover_samples, file_signature, load_references, read_jsonl, write_json
from .statistics import distribution_bootstrap, fid, kid, paired_scalar_summaries, scalar_summaries

LEARNED = ("clip", "imagereward", "pickscore", "hpsv3", "musiq", "clipiqa")
DISTRIBUTIONS = ("fid", "kid", "patch_fid", "patch_kid")
ALL_METRICS = (*LEARNED, *DISTRIBUTIONS, "spectrum", "tiling")
METRIC_SCHEMA = 1


def release_model():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize(device):
    torch = sys.modules.get("torch")
    if torch is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def package_versions():
    versions = {}
    for package in ("torch", "transformers", "pyiqa", "image-reward", "hpsv3", "torch-fidelity", "pytorch-fid"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def cached_model_revisions():
    hub = Path(os.environ.get("HF_HOME", "/scratch/09032/saini_2/tmp/hf")) / "hub"
    result = {}
    for repo in ("openai/clip-vit-large-patch14", "yuvalkirstain/PickScore_v1",
                 "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "THUDM/ImageReward", "MizzenAI/HPSv3"):
        ref = hub / ("models--" + repo.replace("/", "--")) / "refs" / "main"
        result[repo] = ref.read_text().strip() if ref.is_file() else None
    return result


def csv_rows(path: Path, rows: list[dict]):
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def cluster_features(records: list[dict], bank: dict, *, method: str, res: str,
                     align_with: str | None = None) -> dict[str, np.ndarray]:
    """Align seeds before assembling prompt clusters for paired comparisons."""
    available = {(r["prompt_id"], str(r["seed"])) for r in records
                 if r["method"] == align_with and r["res"] == res and r["path"] in bank}
    grouped = defaultdict(list)
    for record in records:
        if record["method"] != method or record["res"] != res or record["path"] not in bank:
            continue
        if align_with is not None and (record["prompt_id"], str(record["seed"])) not in available:
            continue
        grouped[record["prompt_id"]].append(bank[record["path"]])
    return {key: np.concatenate(values, axis=0) for key, values in grouped.items()}


class Evaluation:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.metrics = list(ALL_METRICS) if args.metrics == "all" else list(dict.fromkeys(args.metrics.split(",")))
        unknown = set(self.metrics) - set(ALL_METRICS)
        if unknown:
            raise ValueError(f"Unknown metrics: {sorted(unknown)}")
        self.records = discover_samples(args.root, read_jsonl(args.prompts), args.res)
        self.references = load_references(args.references) if args.references else []
        self.versions = package_versions()
        self.settings = {"schema": METRIC_SCHEMA, "crop_seed": args.crop_seed, "crops": args.crops,
                         "versions": self.versions, "model_revisions": cached_model_revisions()}
        self.cache_path = self.out / "evaluation_cache.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.rows = [{**record, "metrics": {"seconds": record["seconds"]}, "eval_seconds": {}}
                     for record in self.records]
        self.status, self.summaries, self.paired = {}, [], []
        self.errors = []
        self.optional_errors = []

    def key(self, record, metric, extra=None):
        return digest({"file": file_signature(record["path"]), "prompt": record.get("prompt"),
                       "prompt_id": record["prompt_id"], "metric": metric, "settings": self.settings,
                       "source_config_hash": record.get("config_hash"), "extra": extra})

    def checkpoint(self):
        write_json(self.cache_path, self.cache)

    def failed(self, metric, exc, record=None):
        message = f"{type(exc).__name__}: {exc}"
        entry = {"metric": metric, "error": message}
        if record:
            entry["path"] = record["path"]
        optional = metric == "hpsv3" and self.args.metrics == "all"
        (self.optional_errors if optional else self.errors).append(entry)
        self.status.setdefault(metric, {})["status"] = "optional_unavailable" if optional else "unavailable_or_incomplete"
        self.status[metric].setdefault("errors", []).append(entry)
        print(f"[{metric}] {message}", file=sys.stderr, flush=True)

    def learned(self, metric):
        from .models import load_scorer

        pending = [(row, self.key(row, metric)) for row in self.rows if self.key(row, metric) not in self.cache]
        scorer = None
        self.status[metric] = {"status": "complete"}
        if pending:
            try:
                scorer = load_scorer(metric, self.args.device)
                self.status[metric]["provenance"] = scorer.provenance
                for row, key in pending:
                    try:
                        synchronize(self.args.device)
                        start = time.perf_counter()
                        scores = scorer.score(row, crop_seed=self.args.crop_seed, crops=self.args.crops)
                        synchronize(self.args.device)
                        if not all(np.isfinite(value) for value in scores.values()):
                            raise ValueError("Scorer returned non-finite values")
                        self.cache[key] = {"scores": scores, "seconds": time.perf_counter() - start,
                                           "provenance": scorer.provenance}
                        self.checkpoint()
                    except Exception as exc:
                        self.failed(metric, exc, row)
            except Exception as exc:
                self.failed(metric, exc)
            finally:
                del scorer
                release_model()
        for row in self.rows:
            saved = self.cache.get(self.key(row, metric))
            if saved:
                row["metrics"].update(saved["scores"])
                row["eval_seconds"][metric] = saved["seconds"]
                self.status[metric].setdefault("provenance", saved["provenance"])

    def profile(self, record):
        key = self.key(record, "luminance_octaves")
        if key not in self.cache:
            with Image.open(record["path"]) as image:
                value = octave_profile(linear_luminance(image))
            self.cache[key] = {k: v.tolist() for k, v in value.items()}
            self.checkpoint()
        return {key: np.asarray(value, dtype=np.float64) for key, value in self.cache[key].items()}

    def diagnostics(self):
        reference_profile = None
        if "spectrum" in self.metrics:
            self.status["spectrum"] = {"status": "complete"}
            try:
                if not self.references:
                    raise ValueError("--references is required for spectrum errors")
                profiles = np.stack([self.profile(record)["log_power"] for record in self.references])
                valid = np.isfinite(profiles)
                reference_profile = np.divide(np.nansum(profiles, axis=0), valid.sum(0),
                                              out=np.full(profiles.shape[1], np.nan), where=valid.sum(0) > 0)
                self.status["spectrum"].update(reference_log_power=reference_profile.tolist(),
                    reference_count=len(self.references), frequency=(2.0 ** (np.arange(-12, -1) + 0.5)).tolist(),
                    definition="linear-sRGB luminance, unwindowed radial mean PSD, natural log; fixed reference mean log profile")
            except Exception as exc:
                self.failed("spectrum", exc)
        if "tiling" in self.metrics:
            self.status["tiling"] = {"status": "complete", "definition":
                "overlap-normalized autocorrelation at lag minus 1.5*lag, averaged over valid axes; null if no valid axis"}
        for row in self.rows:
            if reference_profile is not None:
                try:
                    start = time.perf_counter()
                    row["metrics"].update(profile_error(self.profile(row), reference_profile))
                    row["eval_seconds"]["spectrum"] = time.perf_counter() - start
                except Exception as exc:
                    self.failed("spectrum", exc, row)
            if "tiling" in self.metrics:
                key = self.key(row, "tiling")
                try:
                    if key not in self.cache:
                        start = time.perf_counter()
                        with Image.open(row["path"]) as image:
                            scores = tiling_metrics(linear_luminance(image))
                        self.cache[key] = {"scores": scores, "seconds": time.perf_counter() - start}
                        self.checkpoint()
                    row["metrics"].update(self.cache[key]["scores"])
                    row["eval_seconds"]["tiling"] = self.cache[key]["seconds"]
                except Exception as exc:
                    self.failed("tiling", exc, row)

    def distributions(self):
        selected = [metric for metric in DISTRIBUTIONS if metric in self.metrics]
        if not selected:
            return
        if not self.references:
            for metric in selected:
                self.failed(metric, ValueError("--references is required for FID/KID"))
            return
        from .models import InceptionFeatures

        extractor = None
        banks = {}
        provenance = None
        try:
            # One extractor instance serves both whole images and patches.
            for patch in sorted({metric.startswith("patch_") for metric in selected}):
                bank = {}
                for record in [*self.records, *self.references]:
                    key = self.key(record, "inception_features", {"patch": patch})
                    path = self.out / "features" / f"{key}.npz"
                    if path.is_file():
                        with np.load(path, allow_pickle=False) as saved:
                            values = saved["features"]
                            current_provenance = json.loads(str(saved["provenance"]))
                            elapsed = float(saved["seconds"])
                    else:
                        if extractor is None:
                            extractor = InceptionFeatures(self.args.device)
                        current_provenance = extractor.provenance
                        synchronize(self.args.device)
                        start = time.perf_counter()
                        values = extractor.extract(record, patch=patch, crop_seed=self.args.crop_seed, crops=self.args.crops)
                        synchronize(self.args.device)
                        elapsed = time.perf_counter() - start
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temp = path.with_name(path.name + ".tmp")
                        with temp.open("wb") as handle:
                            np.savez_compressed(handle, features=values,
                                                provenance=json.dumps(current_provenance), seconds=elapsed)
                        temp.replace(path)
                    if provenance is not None and provenance != current_provenance:
                        raise ValueError("Inception cache mixes backends; use a new --out directory")
                    provenance = current_provenance
                    bank[record["path"]] = values
                    for row in self.rows:
                        if row["path"] == record["path"]:
                            row["eval_seconds"]["inception_patch1024" if patch else "inception299"] = elapsed
                banks[patch] = bank
        except Exception as exc:
            for metric in selected:
                self.failed(metric, exc)
            return
        finally:
            del extractor
            release_model()
        groups = sorted({(r["method"], r["res"]) for r in self.records})
        for metric in selected:
            self.status[metric] = {"status": "complete", "provenance": provenance,
                "estimator": "unbiased all-pairs polynomial degree 3" if metric.endswith("kid") else "empirical Gaussian FID",
                "bootstrap": "generated and real prompt clusters independently; shared real draw for paired deltas"}
            bank = banks[metric.startswith("patch_")]
            real = {r["prompt_id"]: bank[r["path"]] for r in self.references}
            statistic = kid if metric.endswith("kid") else fid
            for method, res in groups:
                try:
                    generated = cluster_features(self.records, bank, method=method, res=res)
                    summary = distribution_bootstrap(generated, real, statistic,
                        draws=self.args.bootstrap, seed=self.args.bootstrap_seed)
                    self.summaries.append({"method": method, "res": res, "metric": metric, **summary})
                    reference = self.args.reference_method
                    if method != reference and (reference, res) in groups:
                        aligned = cluster_features(self.records, bank, method=method, res=res, align_with=reference)
                        baseline = cluster_features(self.records, bank, method=reference, res=res, align_with=method)
                        delta = distribution_bootstrap(aligned, real, statistic, baseline=baseline,
                            draws=self.args.bootstrap, seed=self.args.bootstrap_seed)
                        self.paired.append({"method": method, "res": res, "metric": metric,
                                            "reference_method": reference, "delta_definition": "method_minus_reference",
                                            **delta})
                except Exception as exc:
                    self.failed(metric, ValueError(f"{method}/{res}: {exc}"))

    def wandb(self, report):
        if self.args.no_wandb:
            report["wandb"] = {"status": "disabled_explicitly"}
            return
        run = None
        try:
            import wandb

            directory = os.environ.get("WANDB_DIR", "/scratch/09032/saini_2/constancy_rope/wandb")
            Path(directory).mkdir(parents=True, exist_ok=True)
            mode = os.environ.get("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "constancy-rope"), dir=directory,
                             mode=mode, job_type="evaluation", config=report["config"])
            for row in self.rows:
                values = {f"metrics/{k}": v for k, v in row["metrics"].items() if v is not None}
                values.update({f"eval_seconds/{k}": v for k, v in row["eval_seconds"].items()})
                values.update(method=row["method"], resolution=row["res"], prompt_id=row["prompt_id"], seed=row["seed"])
                run.log(values)
            tables = {}
            for name, rows in (("summary", self.summaries), ("paired_deltas", self.paired)):
                columns = sorted({key for row in rows for key in row}) or ["method", "res", "metric", "value"]
                tables[name] = wandb.Table(columns=columns, data=[[row.get(key) for key in columns] for row in rows])
            run.log(tables)
            report["wandb"] = {"status": "logged", "mode": mode, "run_id": run.id}
        except Exception as exc:
            self.failed("wandb", exc)
            report["wandb"] = {"status": "failed", "error": str(exc)}
        finally:
            if run is not None:
                run.finish()

    def run(self):
        started = time.perf_counter()
        self.diagnostics()
        for metric in LEARNED:
            if metric in self.metrics:
                print(f"Evaluating {metric} on {len(self.rows)} images", flush=True)
                self.learned(metric)
        self.distributions()
        self.summaries.extend(scalar_summaries(self.rows, draws=self.args.bootstrap, seed=self.args.bootstrap_seed))
        self.paired.extend(paired_scalar_summaries(self.rows, self.args.reference_method,
                           draws=self.args.bootstrap, seed=self.args.bootstrap_seed))
        report = {"config": vars(self.args), "metric_schema": METRIC_SCHEMA, "versions": self.versions,
                  "status": self.status, "errors": self.errors, "summary": self.summaries,
                  "optional_unavailable": self.optional_errors, "model_revisions": self.settings["model_revisions"],
                  "paired_deltas": self.paired, "per_image": self.rows,
                  "uncertainty": "95% percentile bootstrap over prompt clusters; seeds and crops stay inside prompts",
                  "paired_note": "Only matching resolution/prompt/seed pairs enter deltas; absent reference yields no delta",
                  "seconds": time.perf_counter() - started}
        flat_rows = [{**{k: row[k] for k in ("method", "res", "prompt_id", "seed", "path")},
                      **row["metrics"], **{f"eval_seconds_{k}": v for k, v in row["eval_seconds"].items()}}
                     for row in self.rows]
        csv_rows(self.out / "per_image.csv", flat_rows)
        csv_rows(self.out / "summary.csv", self.summaries)
        csv_rows(self.out / "paired_deltas.csv", self.paired)
        write_json(self.out / "metrics.json", report)
        self.wandb(report)
        report["complete"] = not self.errors
        report["all_metrics_available"] = not self.errors and not self.optional_errors
        write_json(self.out / "metrics.json", report)
        print(f"Evaluation: {len(self.rows)} images, {len(self.summaries)} summaries, "
              f"{len(self.paired)} paired deltas, {len(self.errors)} errors, "
              f"{len(self.optional_errors)} optional unavailable; reports in {self.out}", flush=True)
        return 2 if self.errors and not self.args.allow_missing else 0
