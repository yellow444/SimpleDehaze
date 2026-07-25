#!/usr/bin/env python3
"""Validate and summarize a SimpleDeHaze controlled-DIODE benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_VARIANTS = {"B0", "B3", "B7", "B8", "B9"}
EXPECTED_TARGETS = {"0.8", "0.6", "0.4", "0.2", "0.1"}
EXPECTED_AIRLIGHTS = {"neutral", "warm", "cool"}
EXPECTED_NOISES = {"clean", "gaussian", "poisson_gaussian"}
PROTECTED_VARIANTS = {"B3", "B8", "B9"}
METRICS = (
    "psnr",
    "ssim",
    "ciede2000",
    "hue_error_deg",
    "chroma_error",
    "invalid_channel_before",
    "invalid_channel_after",
    "required_clip_mean",
    "clip_pct",
    "flat_noise_input_sigma",
    "flat_noise_output_sigma",
    "t_hat_rmse",
    "airlight_l2_error",
    "g_parallel_rmse",
    "g_perp_rmse",
    "projected_pixel_fraction",
    "ms",
    "working_set_mb",
)

REQUIRED_TEXT = (
    "dataset",
    "split",
    "frame_id",
    "scene_group",
    "domain",
    "clear_path",
    "depth_path",
    "depth_mask_path",
    "hazy_artifact",
    "transmission_artifact",
    "recipe",
    "random_seed",
    "airlight",
    "noise",
    "variant",
)

REQUIRED_NUMERIC = (
    "target_t_p90",
    "airlight_r_linear",
    "airlight_g_linear",
    "airlight_b_linear",
    "gaussian_sigma",
    "poisson_peak",
    "width",
    "height",
    "valid_depth_fraction",
    "depth_p90",
    "beta",
    "t_true_mean",
    "t_hat_rmse",
    "airlight_l2_error",
    "psnr",
    "ssim",
    "ciede2000",
    "hue_chromatic_fraction",
    "chroma_error",
    "invalid_channel_before",
    "invalid_channel_after",
    "required_clip_mean",
    "clip_pct",
    "flat_noise_input_sigma",
    "flat_noise_output_sigma",
    "g_parallel_rmse",
    "g_perp_rmse",
    "mean_g_parallel",
    "mean_g_perp",
    "mean_alpha",
    "projected_pixel_fraction",
    "ms",
    "working_set_mb",
)


def finite(value: str, field: str, line: int) -> float:
    if value == "":
        raise ValueError(f"line {line}: required numeric field {field!r} is blank")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"line {line}: invalid {field}={value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"line {line}: non-finite {field}={value!r}")
    return result


def aggregate_factory() -> dict[str, float]:
    return {"count": 0.0, **{metric: 0.0 for metric in METRICS}}


def add_aggregate(store: dict[str, dict[str, float]], key: str, values: dict[str, float]) -> None:
    item = store[key]
    item["count"] += 1.0
    for metric in METRICS:
        item[metric] += values[metric]


def finish_aggregates(store: dict[str, dict[str, float]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for key in sorted(store):
        item = store[key]
        count = int(item["count"])
        result[key] = {
            "count": count,
            **{metric: item[metric] / count for metric in METRICS},
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--expected-frames", type=int, default=500)
    args = parser.parse_args()

    csv_path = args.csv_path.resolve()
    frames: set[str] = set()
    recipes: set[str] = set()
    seeds: set[str] = set()
    variants: set[str] = set()
    targets: set[str] = set()
    airlights: set[str] = set()
    noises: set[str] = set()
    domains: set[str] = set()
    duplicate_keys: set[tuple[str, str]] = set()
    seen_keys: set[tuple[str, str]] = set()
    recipe_counts: dict[str, int] = defaultdict(int)
    recipe_seeds: dict[str, set[str]] = defaultdict(set)
    frame_recipes: dict[str, int] = defaultdict(int)
    frame_splits: dict[str, set[str]] = defaultdict(set)
    frame_domains: dict[str, set[str]] = defaultdict(set)
    scene_splits: dict[str, set[str]] = defaultdict(set)
    split_frames: dict[str, set[str]] = defaultdict(set)
    domain_frames: dict[str, set[str]] = defaultdict(set)
    protected_invalid_max: dict[str, float] = defaultdict(float)
    variant_aggregate: dict[str, dict[str, float]] = defaultdict(aggregate_factory)
    split_aggregate: dict[str, dict[str, float]] = defaultdict(aggregate_factory)
    domain_aggregate: dict[str, dict[str, float]] = defaultdict(aggregate_factory)
    target_aggregate: dict[str, dict[str, float]] = defaultdict(aggregate_factory)
    noise_aggregate: dict[str, dict[str, float]] = defaultdict(aggregate_factory)
    paired: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    rows = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        if len(reader.fieldnames) != 53:
            raise ValueError(f"expected 53 columns, found {len(reader.fieldnames)}")
        missing_columns = (set(REQUIRED_TEXT) | set(REQUIRED_NUMERIC) | {
            "lpips", "lpips_status", "flat_noise_amplification", "gpu_used_mb"
        }) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"missing columns: {sorted(missing_columns)}")

        for line, row in enumerate(reader, start=2):
            rows += 1
            if None in row or len(row) != 53:
                raise ValueError(f"line {line}: malformed column count")
            for field in REQUIRED_TEXT:
                if not row[field]:
                    raise ValueError(f"line {line}: required field {field!r} is blank")
            values = {field: finite(row[field], field, line) for field in REQUIRED_NUMERIC}

            chromatic_fraction = values["hue_chromatic_fraction"]
            if row["hue_error_deg"]:
                values["hue_error_deg"] = finite(row["hue_error_deg"], "hue_error_deg", line)
            elif chromatic_fraction == 0.0:
                values["hue_error_deg"] = 0.0
            else:
                raise ValueError(f"line {line}: hue error blank with chromatic pixels")

            noise = row["noise"]
            if noise == "clean":
                if row["flat_noise_amplification"]:
                    raise ValueError(f"line {line}: clean recipe has a noise amplification ratio")
            else:
                finite(row["flat_noise_amplification"], "flat_noise_amplification", line)
            if row["lpips"] or row["lpips_status"] != "not_requested":
                raise ValueError(f"line {line}: full grid must mark LPIPS as not requested")
            if row["gpu_used_mb"]:
                finite(row["gpu_used_mb"], "gpu_used_mb", line)

            frame = row["frame_id"]
            recipe = row["recipe"]
            seed = row["random_seed"]
            variant = row["variant"]
            split = row["split"]
            domain = row["domain"]
            scene = row["scene_group"]
            target = format(values["target_t_p90"], ".1f")
            key = (recipe, variant)
            if key in seen_keys:
                duplicate_keys.add(key)
            seen_keys.add(key)

            frames.add(frame)
            recipes.add(recipe)
            seeds.add(seed)
            variants.add(variant)
            targets.add(target)
            airlights.add(row["airlight"])
            noises.add(noise)
            domains.add(domain)
            recipe_counts[recipe] += 1
            recipe_seeds[recipe].add(seed)
            frame_splits[frame].add(split)
            frame_domains[frame].add(domain)
            scene_splits[scene].add(split)
            split_frames[split].add(frame)
            domain_frames[domain].add(frame)
            if variant == "B0":
                frame_recipes[frame] += 1
            if variant in PROTECTED_VARIANTS:
                protected_invalid_max[variant] = max(
                    protected_invalid_max[variant], values["invalid_channel_after"]
                )

            metric_values = {metric: values[metric] for metric in METRICS}
            add_aggregate(variant_aggregate, variant, metric_values)
            add_aggregate(split_aggregate, f"{variant}|{split}", metric_values)
            add_aggregate(domain_aggregate, f"{variant}|{domain}", metric_values)
            add_aggregate(target_aggregate, f"{variant}|{target}", metric_values)
            add_aggregate(noise_aggregate, f"{variant}|{noise}", metric_values)
            if variant in {"B0", "B9"}:
                paired[recipe][variant] = metric_values

    expected_recipes = args.expected_frames * 45
    expected_rows = expected_recipes * len(EXPECTED_VARIANTS)
    assertions: dict[str, bool] = {
        "row_count": rows == expected_rows,
        "frame_count": len(frames) == args.expected_frames,
        "recipe_count": len(recipes) == expected_recipes,
        "unique_seed_count": len(seeds) == expected_recipes,
        "variants": variants == EXPECTED_VARIANTS,
        "targets": targets == EXPECTED_TARGETS,
        "airlights": airlights == EXPECTED_AIRLIGHTS,
        "noises": noises == EXPECTED_NOISES,
        "domains": domains == {"indoor", "outdoor"},
        "no_duplicate_recipe_variant": not duplicate_keys,
        "five_variants_per_recipe": all(count == 5 for count in recipe_counts.values()),
        "one_seed_per_recipe": all(len(values) == 1 for values in recipe_seeds.values()),
        "forty_five_recipes_per_frame": all(count == 45 for count in frame_recipes.values()),
        "one_split_per_frame": all(len(values) == 1 for values in frame_splits.values()),
        "one_domain_per_frame": all(len(values) == 1 for values in frame_domains.values()),
        "no_scene_leakage": all(len(values) == 1 for values in scene_splits.values()),
        "protected_variants_finite": set(protected_invalid_max) == PROTECTED_VARIANTS,
        "protected_variants_zero_invalid_after": all(
            value <= 1e-12 for value in protected_invalid_max.values()
        ),
        "all_b0_b9_pairs": len(paired) == expected_recipes
        and all(set(values) == {"B0", "B9"} for values in paired.values()),
    }
    failures = [name for name, passed in assertions.items() if not passed]
    if failures:
        raise ValueError(f"validation failed: {failures}")

    wins = {
        "psnr": sum(pair["B9"]["psnr"] > pair["B0"]["psnr"] for pair in paired.values()),
        "ssim": sum(pair["B9"]["ssim"] > pair["B0"]["ssim"] for pair in paired.values()),
        "ciede2000": sum(
            pair["B9"]["ciede2000"] < pair["B0"]["ciede2000"] for pair in paired.values()
        ),
        "hue_error_deg": sum(
            pair["B9"]["hue_error_deg"] < pair["B0"]["hue_error_deg"] for pair in paired.values()
        ),
        "chroma_error": sum(
            pair["B9"]["chroma_error"] < pair["B0"]["chroma_error"] for pair in paired.values()
        ),
    }
    summary: dict[str, Any] = {
        "source": str(csv_path),
        "bytes": csv_path.stat().st_size,
        "columns": 53,
        "rows": rows,
        "frames": len(frames),
        "recipes": len(recipes),
        "unique_seeds": len(seeds),
        "split_frames": {key: len(value) for key, value in sorted(split_frames.items())},
        "domain_frames": {key: len(value) for key, value in sorted(domain_frames.items())},
        "protected_invalid_after_max": dict(sorted(protected_invalid_max.items())),
        "assertions": assertions,
        "variant": finish_aggregates(variant_aggregate),
        "by_split": finish_aggregates(split_aggregate),
        "by_domain": finish_aggregates(domain_aggregate),
        "by_target_t_p90": finish_aggregates(target_aggregate),
        "by_noise": finish_aggregates(noise_aggregate),
        "b9_wins_over_b0": {**wins, "paired_recipes": len(paired)},
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        output = args.json_out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
