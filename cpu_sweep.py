#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import wandb
import yaml


REPO_ROOT = Path(__file__).resolve().parent
CPU_LAUNCHER = REPO_ROOT / "cpu_from_json.sh"
SWEEPS_ROOT = REPO_ROOT / "sweeps"
DEFAULT_SWEEP_CONFIG = REPO_ROOT / "run_configs" / "sweep_default.yaml"

for candidate in (
    REPO_ROOT / "vsl-rm",
    Path.cwd() / "vsl-rm",
    Path(os.getenv("HOME", "")) / "ValueLearningInGenAI" / "vsl-rm",
):
    if (candidate / "vsllib").exists():
        sys.path.insert(0, str(candidate))
        break

from vsllib.utils import ScriptArguments  # type: ignore[import-not-found]


SCRIPT_ARGUMENT_FIELDS = set(ScriptArguments.__dataclass_fields__.keys())


def load_json_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    if not isinstance(config_data, dict):
        raise ValueError(
            f"Expected JSON object in config file, got {type(config_data).__name__}"
        )

    return config_data


def load_sweep_configuration(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        sweep_config = yaml.safe_load(handle)

    if not isinstance(sweep_config, dict):
        raise ValueError(
            f"Expected mapping in sweep config, got {type(sweep_config).__name__}"
        )

    return sweep_config


def normalize_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in overrides.items():
        normalized_key = key
        if normalized_key in SCRIPT_ARGUMENT_FIELDS:
            normalized[normalized_key] = value

    if "per_device_train_batch_size" in normalized:
        normalized["per_device_eval_batch_size"] = normalized["per_device_train_batch_size"]
    if "learning_rate" in normalized:
        normalized["grounding_learning_rate"] = normalized["learning_rate"]
    
    return normalized


def write_merged_config(default_config_file: Path, run: wandb.sdk.wandb_run.Run) -> Path:
    default_config = load_json_config(default_config_file)
    sweep_overrides = normalize_overrides(dict(run.config.as_dict()))
    merged_config = default_config.copy()
    merged_config.update(sweep_overrides)

    run_dir = SWEEPS_ROOT / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    merged_config_path = run_dir / "config.json"
    with merged_config_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_config, handle, indent=2)
        handle.write("\n")

    return merged_config_path


def main(default_config_file: str, dataset: str, rs: np.random.RandomState, num_train_epochs: int) -> int:
    run = wandb.init()
    if run is None:
        raise RuntimeError("wandb.init() did not return a run")

    run_project = getattr(run, "project", None)
    run_id = run.id

    try:
        merged_config_path = write_merged_config(
            Path(default_config_file).expanduser().resolve(), run
        )
    finally:
        wandb.finish()

    subprocess_env = os.environ.copy()
    if run_project:
        subprocess_env["WANDB_PROJECT"] = run_project
    #subprocess_env["PYTHONUNBUFFERED"] = "1"
    subprocess_env["WANDB_RUN_ID"] = run_id
    subprocess_env["WANDB_RESUME"] = "allow"

    
    completed = subprocess.run(
        ["bash", str(CPU_LAUNCHER), str(merged_config_path), f"--dataset={dataset}", f"--do_save={False}", f"--do_checkpointing={False}", f"--num_train_epochs={num_train_epochs}", f"--seed={rs.randint(0, 1000000)}"],
        env=subprocess_env,
        check=False,
    )
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a W&B sweep on CPU.")
    parser.add_argument(
        "--config_file",
        help="Path to the JSON config file whose values will be overridden by the sweep.",
    )
    parser.add_argument(
        "--project",
        default="ValueLearningInGenAISweeps",
        help="W&B project name for the sweep.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of sweep runs to execute.",
    )
    parser.add_argument(
        "--sweep-config",
        default=str(DEFAULT_SWEEP_CONFIG),
        help="Path to the W&B sweep configuration YAML file.",
    )
    parser.add_argument(
        "--dataset",
        default="ultra",
        choices=["ultra", "pku"],
        help="Dataset to use for the sweep.",
    )
    parser.add_argument(
        "--nepochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    return parser.parse_args()


def run_sweep() -> None:
    rs = np.random.RandomState(seed=42)
    args = parse_args()
    sweep_configuration = load_sweep_configuration(Path(args.sweep_config).expanduser().resolve())
    print(sweep_configuration)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)
    wandb.agent(sweep_id, function=partial(main, args.config_file, args.dataset, rs, args.nepochs), count=args.count)


if __name__ == "__main__":
    run_sweep()
