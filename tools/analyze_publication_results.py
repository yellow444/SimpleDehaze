#!/usr/bin/env python3
"""Cluster-aware publication statistics for SimpleDeHaze experiments.

The controlled DIODE grid contains 45 correlated recipes per source frame. This script
aggregates within frame first, then performs an indoor/outdoor-stratified frame bootstrap.
It also reports leave-one-scene-group-out sensitivity because DIODE validation contains
only six physical scene groups. Real paired datasets are bootstrapped by image id.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DIODE_METRICS = {
    "psnr": "higher",
    "ssim": "higher",
    "ciede2000": "lower",
    "hue_error_deg": "lower",
    "chroma_error": "lower",
    "invalid_channel_after": "lower",
    "flat_noise_output_sigma": "lower",
    "required_clip_mean": "lower",
    "ms": "lower",
}
REAL_METRICS = {
    "psnr": "higher",
    "ssim": "higher",
    "ciede2000": "lower",
    "lpips": "lower",
    "flat_noise_x": "lower",
    "clip_pct": "lower",
    "ms_median": "lower",
}
BASELINES = ("B0", "B3", "B7", "B8")
TARGET_VARIANT = "B9"
TARGET_METHOD_PREFIX = "A²CR-Dehaze"


def finite(text: str, field: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"invalid {field}={text!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}={text!r}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def bootstrap_ci(values_by_stratum: dict[str, list[float]], iterations: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    samples: list[float] = []
    total_count = sum(len(values) for values in values_by_stratum.values())
    if total_count == 0:
        return math.nan, math.nan
    for _ in range(iterations):
        total = 0.0
        for values in values_by_stratum.values():
            total += sum(values[rng.randrange(len(values))] for _ in range(len(values)))
        samples.append(total / total_count)
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def contrast_record(
    analysis: str,
    dataset: str,
    contrast: str,
    metric: str,
    direction: str,
    differences: list[tuple[str, str, float]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    values_by_stratum: dict[str, list[float]] = defaultdict(list)
    scene_values: dict[str, list[float]] = defaultdict(list)
    for stratum, scene, value in differences:
        values_by_stratum[stratum].append(value)
        scene_values[scene].append(value)
    values = [value for _, _, value in differences]
    point = sum(values) / len(values)
    ci_low, ci_high = bootstrap_ci(values_by_stratum, iterations, seed)
    favorable = sum(value > 0 if direction == "higher" else value < 0 for value in values) / len(values)

    leave_one_scene_out: list[float] = []
    scenes = sorted(scene_values)
    if len(scenes) > 1:
        for omitted in scenes:
            retained = [value for _, scene, value in differences if scene != omitted]
            leave_one_scene_out.append(sum(retained) / len(retained))

    return {
        "analysis": analysis,
        "dataset": dataset,
        "contrast": contrast,
        "metric": metric,
        "direction": direction,
        "difference_definition": f"target_minus_baseline ({'positive' if direction == 'higher' else 'negative'} is favorable)",
        "clusters": len(values),
        "difference": point,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "ci_excludes_zero": ci_low > 0 or ci_high < 0,
        "favorable_cluster_fraction": favorable,
        "scene_groups": {
            scene: {"frames": len(scene_values[scene]), "mean_difference": sum(scene_values[scene]) / len(scene_values[scene])}
            for scene in scenes
        },
        "leave_one_scene_group_out_min": min(leave_one_scene_out) if leave_one_scene_out else None,
        "leave_one_scene_group_out_max": max(leave_one_scene_out) if leave_one_scene_out else None,
    }


def diode_statistics(path: Path, iterations: int, seed: int) -> list[dict[str, Any]]:
    accum: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    frame_domain: dict[str, str] = {}
    frame_scene: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            frame = row["frame_id"]
            variant = row["variant"]
            frame_domain.setdefault(frame, row["domain"])
            frame_scene.setdefault(frame, row["scene_group"])
            if frame_domain[frame] != row["domain"] or frame_scene[frame] != row["scene_group"]:
                raise ValueError(f"frame metadata changed for {frame}")
            for metric in DIODE_METRICS:
                key = (frame, variant, metric)
                accum[key][0] += finite(row[metric], metric)
                accum[key][1] += 1.0

    frames = sorted(frame_domain)
    if len(frames) != 500:
        raise ValueError(f"expected 500 DIODE frames, found {len(frames)}")
    records: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(BASELINES):
        for metric_index, (metric, direction) in enumerate(DIODE_METRICS.items()):
            differences: list[tuple[str, str, float]] = []
            for frame in frames:
                target_sum, target_count = accum[(frame, TARGET_VARIANT, metric)]
                base_sum, base_count = accum[(frame, baseline, metric)]
                if target_count != 45 or base_count != 45:
                    raise ValueError(f"expected 45 recipes for {frame}/{baseline}/{metric}")
                difference = target_sum / target_count - base_sum / base_count
                differences.append((frame_domain[frame], frame_scene[frame], difference))
            records.append(contrast_record(
                "diode_frame_cluster_bootstrap", "DIODE-controlled", f"B9 - {baseline}",
                metric, direction, differences, iterations,
                seed + baseline_index * 1000 + metric_index,
            ))
    return records


def non_comment_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if not line.startswith("#"):
                yield line


def real_statistics(paths: list[Path], iterations: int, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths):
        rows = list(csv.DictReader(non_comment_lines(path), delimiter=";"))
        if not rows:
            raise ValueError(f"empty real-paired CSV: {path}")
        dataset = rows[0]["dataset"]
        by_image: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for row in rows:
            if row["ok"] != "1":
                raise ValueError(f"failed row in {path}: {row['image']} {row['method']}")
            by_image[row["image"]][row["method"]] = {
                metric: finite(row[metric], metric) for metric in REAL_METRICS
            }
        target_names = {method for methods in by_image.values() for method in methods if method.startswith(TARGET_METHOD_PREFIX)}
        if len(target_names) != 1:
            raise ValueError(f"expected one A²CR method in {path}, found {target_names}")
        target_name = next(iter(target_names))
        baseline_names = sorted({method for methods in by_image.values() for method in methods if method != target_name})

        for baseline_index, baseline in enumerate(baseline_names):
            for metric_index, (metric, direction) in enumerate(REAL_METRICS.items()):
                differences: list[tuple[str, str, float]] = []
                for image, methods in sorted(by_image.items()):
                    if set(methods) != set(baseline_names) | {target_name}:
                        raise ValueError(f"unpaired methods for {dataset}/{image}")
                    differences.append(("all", image, methods[target_name][metric] - methods[baseline][metric]))
                record = contrast_record(
                    "real_image_paired_bootstrap", dataset, f"A2CR - {baseline}",
                    metric, direction, differences, iterations,
                    seed + 100_000 + path_index * 10_000 + baseline_index * 1000 + metric_index,
                )
                # Every image is its own independent cluster here; scene-group sensitivity is not applicable.
                record["scene_groups"] = None
                record["leave_one_scene_group_out_min"] = None
                record["leave_one_scene_group_out_max"] = None
                records.append(record)
    return records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "analysis", "dataset", "contrast", "metric", "direction", "clusters",
        "difference", "ci95_low", "ci95_high", "ci_excludes_zero", "favorable_cluster_fraction",
        "leave_one_scene_group_out_min", "leave_one_scene_group_out_max",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diode", type=Path, required=True)
    parser.add_argument("--real", type=Path, nargs="*", default=[])
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.iterations < 1000:
        raise ValueError("at least 1000 bootstrap iterations are required")

    diode = args.diode.resolve()
    real = [path.resolve() for path in args.real]
    records = diode_statistics(diode, args.iterations, args.seed)
    records.extend(real_statistics(real, args.iterations, args.seed))
    payload = {
        "schema_version": 1,
        "bootstrap_iterations": args.iterations,
        "random_seed": args.seed,
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [diode, *real]
        ],
        "methodology": {
            "diode": "Average 45 recipes within each frame, then stratified bootstrap of 250 indoor and 250 outdoor frames.",
            "scene_sensitivity": "Report leave-one-physical-scene-group-out range; only six DIODE validation scene groups exist.",
            "real_paired": "Paired nonparametric bootstrap by image id within each dataset.",
            "interpretation": "Intervals are descriptive because datasets were already used during development and are not blind publication tests.",
        },
        "records": records,
    }
    args.json_out.resolve().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.csv_out.resolve(), records)
    print(json.dumps({
        "records": len(records),
        "diode_records": sum(record["analysis"].startswith("diode") for record in records),
        "real_records": sum(record["analysis"].startswith("real") for record in records),
        "json": str(args.json_out.resolve()),
        "csv": str(args.csv_out.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
