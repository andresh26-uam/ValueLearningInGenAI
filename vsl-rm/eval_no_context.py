#!/usr/bin/env python3

from dataclasses import dataclass, field
from functools import partial
import csv
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from enum import Enum

import numpy as np
import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

# Make local package imports robust when launched from different working directories.
for candidate in (
    Path(__file__).resolve().parent,
    Path.cwd() / "vsl-rm",
    Path(os.getenv("HOME", "")) / "ValueLearningInGenAI" / "vsl-rm",
):
    if (candidate / "vsllib").exists():
        sys.path.insert(0, str(candidate))
        break

from vsllib.training_utils import (
    MORewardDataCollatorWithPadding,
)
from vsllib.training import MORewardTrainer

from vsllib.dataset_processing import PairwisePreferenceDataset
from vsllib.defines import (
    EXTRA_KEYS,
    MODEL_DIR,
    RESULTS_DIR,
    REWARD_HEADS_INDICES,
    REWARD_HEADS_OUTPUT,
    PROCESSED_DATASET_PATHS,
    VALUE_SYSTEM_OUTPUT,
    get_test_indices,
    get_validation_indices,
)
from vsllib.reward_models import (
    MORMForSequenceClassification,
    MORMForSequenceClassificationConfig,
    mo_compute_loss_func,
)
from vsllib.utils import  ScriptArguments, argument_parser, obtain_tokenizer, seed_everything


@dataclass
class EvalArguments(ScriptArguments):

    checkpoint_paths: List[str] = field(
        default_factory=list,
        metadata={"help": "run name in the directory saved by no_context_vsl.py. If not supplied, will ask the user to select from the available runs in the output directory."},
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "run name in the directory saved by no_context_vsl.py. If not supplied, will ask the user to select from the available runs in the output directory."},
    )
    results_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to output CSV file. (Normally saved under .results/script_args.output_dir/script_args.run_name/metrics.csv)"},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the results to Hugging Face Hub. Requires HF_CLI_TOKEN env variable to be set."},
    )


def _is_valid_checkpoint_dir(path: Path) -> bool:
    return (
        path.exists()
        and path.is_dir()
        and (path / "config.json").exists()
    )


def _normalize_candidate_checkpoint_path(path: str, output_path: str) -> Path:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path(output_path) / path_obj
    return path_obj.resolve()


def _resolve_nested_checkpoint_path(start_path: Path, *, allow_finish: bool) -> Optional[Path]:
    current_path = start_path
    while True:
        if _is_valid_checkpoint_dir(current_path):
            return current_path

        print(
            f"Warning: The selected checkpoint path does not contain expected files "
            f"(config.json): {current_path}"
        )

        subdirs = [item for item in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, item))]
        if not subdirs:
            print(f"No subfolders found in {current_path}.")
            return None

        for i, item in enumerate(subdirs):
            print(f"  - ({i}) {item}")

        prompt = "Please enter the index of the valid checkpoint path from the above list"
        if allow_finish:
            prompt += " or OK to finish"
        prompt += ": "
        index_ = input(prompt).strip()
        if allow_finish and index_.upper() == "OK":
            return None

        try:
            current_path = current_path / subdirs[int(index_)]
        except (ValueError, IndexError):
            print(f"Invalid index entered: {index_}")


def _prompt_for_checkpoint_path(output_path: Path, *, allow_finish: bool) -> Optional[Path]:
    available_runs = [item for item in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, item)) and len(os.listdir(os.path.join(output_path, item))) > 0]
    if not available_runs:
        print(f"No available runs found in {output_path}.")
        return None

    while True:
        print(f"Available runs in {output_path}:")
        for i, item in enumerate(available_runs):
            print(f"  - ({i}) {item}")

        prompt = "Please enter the index of the valid checkpoint path from the above list"
        if allow_finish:
            prompt += " or OK to finish"
        prompt += ": "
        index_ = input(prompt).strip()
        if allow_finish and index_.upper() == "OK":
            return None

        try:
            checkpoint_path = output_path / available_runs[int(index_)]
        except (ValueError, IndexError):
            print(f"Invalid index entered: {index_}")
            continue

        resolved_checkpoint_path = _resolve_nested_checkpoint_path(
            checkpoint_path,
            allow_finish=allow_finish,
        )
        if resolved_checkpoint_path is not None:
            return resolved_checkpoint_path


def _build_eval_arguments(script_args: EvalArguments, checkpoint_path: Path, results_dir: Path) -> EvalArguments:
    arg_values = vars(script_args).copy()
    arg_values["checkpoint_paths"] = [str(checkpoint_path)]
    arg_values["checkpoint_path"] = str(checkpoint_path)
    arg_values["results_dir"] = str(results_dir)
    return EvalArguments(**arg_values)

def parse_eval_args() -> tuple[List[EvalArguments], Dict[str, Any]]:
    parser1 = HfArgumentParser(EvalArguments)
    args = parser1.parse_args_into_dataclasses()[0]

    script_args, preset = argument_parser(args)
    script_args: EvalArguments

    def removed_model_path_segment(path: str) -> str:
        path_obj = Path(path).resolve()
        models_root = Path(MODEL_DIR).resolve()
        try:
            return path_obj.relative_to(models_root).as_posix()
        except ValueError:
            return path_obj.name

    raw_checkpoint_paths: List[str] = []
    checkpoint_paths_value = getattr(script_args, "checkpoint_paths", None)
    if checkpoint_paths_value:
        if isinstance(checkpoint_paths_value, str):
            raw_checkpoint_paths.append(checkpoint_paths_value)
        else:
            raw_checkpoint_paths.extend(str(path) for path in checkpoint_paths_value)
    elif getattr(script_args, "checkpoint_path", None):
        raw_checkpoint_paths.append(str(script_args.checkpoint_path))

    output_path = Path(script_args.output_path)
    resolved_checkpoint_paths: List[Path] = []

    for raw_checkpoint_path in raw_checkpoint_paths:
        candidate_path = _normalize_candidate_checkpoint_path(raw_checkpoint_path, str(output_path))
        if _is_valid_checkpoint_dir(candidate_path):
            resolved_checkpoint_paths.append(candidate_path)
            continue

        print(f"Checkpoint path does not exist: {candidate_path}")
        prompted_path = _prompt_for_checkpoint_path(
            output_path,
            allow_finish=False, 
        )
        if prompted_path is not None:
            resolved_checkpoint_paths.append(prompted_path)

    while True:
        prompted_path = _prompt_for_checkpoint_path(
            output_path,
            allow_finish=len(resolved_checkpoint_paths) > 0,
        )
        if prompted_path is None:
            assert len(resolved_checkpoint_paths) > 0, "At least one valid checkpoint path must be provided."
            break
        else:
            resolved_checkpoint_paths.append(prompted_path)

    results_root = Path(script_args.results_dir) if script_args.results_dir is not None and os.path.exists(script_args.results_dir) else Path(RESULTS_DIR)

    eval_script_args_all: List[EvalArguments] = []
    for checkpoint_path in resolved_checkpoint_paths:
        results_dir = results_root / removed_model_path_segment(str(checkpoint_path))
        eval_script_args_all.append(_build_eval_arguments(script_args, checkpoint_path, results_dir))

    return eval_script_args_all, preset



def flatten_metrics_for_csv(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            for i, entry in enumerate(value):
                flat[f"{key}_{i}"] = float(entry)
        elif isinstance(value, Enum):
            flat[key] = value.value
        elif isinstance(value, (np.floating, np.integer)):
            flat[key] = value.item()
        elif isinstance(value, torch.Tensor):
            flat[key] = float(value.detach().cpu().item()) if value.numel() == 1 else float(value.detach().cpu().mean().item())
        elif isinstance(value, (float, int, str, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat



def write_metrics_csv(metrics: Dict[str, Any], output_path: str, name: str = "test_metrics.csv") -> None:
    path = Path(output_path).joinpath(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted(metrics.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)



def main() -> None:
    script_args_all, preset = parse_eval_args()
    script_args_all: List[EvalArguments]
    

    for script_args in script_args_all:
        seed_everything(int(script_args.seed))
        #torch_dtype = torch.bfloat16 if script_args.bf16 else torch.float32
        #print("TORCH DTYPE:", torch_dtype )
        
        tokenizer = obtain_tokenizer(script_args, preset=preset, checkpoint_path=script_args.checkpoint_path)
        
        model = MORMForSequenceClassification.from_pretrained(
            str(script_args.checkpoint_path),
        )
        torch_dtype = model.config.dtype
        print("VS", model.value_system_layer.get_weights())
        print("MODEL DETAILS:", model, "MODEL DTYPE:", model.dtype, "MODEL CONFIG DTYPE:", torch_dtype)
        print("MODEL DTYPE", model.value_system_layer.weight.dtype)
        print("FIRST PARAMETER:", next(model.parameters()), next(model.parameters()).dtype)
        print("TRAINING VARIABLES:", model.training_variables.state_dict(), model.training_variables.lagrange_multipliers.dtype)
        
        if script_args.push_to_hub:
            from transformers import AutoConfig, AutoModelForSequenceClassification
            print("Pushing model to Hugging Face Hub...?")
            input("")
            AutoConfig.register("morm_for_sequence_classification", MORMForSequenceClassificationConfig)
            AutoModelForSequenceClassification.register(MORMForSequenceClassificationConfig, MORMForSequenceClassification)
            
            mo_1 = AutoModelForSequenceClassification.from_pretrained(
                    str(script_args.checkpoint_path),
                )
            print("MODEL DETAILS:", model)
            print("MODEL DTYPE", model.value_system_layer.weight.dtype)
            print("FIRST PARAMETER:", next(model.parameters()))
            print("TRAINING VARIABLES:", model.training_variables.state_dict())
            

        model.eval()

        dc = MORewardDataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=int(script_args.max_length),
            dtype=torch_dtype,
            use_embeddings=bool(script_args.use_embeddings),
        )

        train_path = PROCESSED_DATASET_PATHS[script_args.dataset]
        extra_keep_keys = EXTRA_KEYS[script_args.dataset]
        test_proportion_or_indices = get_test_indices(script_args.dataset)
        eval_proportion_or_indices = get_validation_indices(script_args.dataset)

        embed_model = model.full_model if script_args.use_embeddings else None

        dataset = PairwisePreferenceDataset(
            train_path,
            tokenizer,
            from_disk=True,
            extra_keep_keys=extra_keep_keys,
            retokenize=False,
            recalculate_embeddings=False,
            use_embeddings=bool(script_args.use_embeddings),
            model_reference=embed_model,
            collator=dc,
            split_seed=int(script_args.seed),
            eval_proportion_or_indices=eval_proportion_or_indices,
            test_proportion_or_indices=test_proportion_or_indices,
            cleanup_cache_files=False,
        )

        model_name_for_maps = getattr(model.config, "base_model_name_or_path", script_args.model_name)
        reward_heads_module_name = REWARD_HEADS_OUTPUT.get(model_name_for_maps, None)
        value_system_module_name = VALUE_SYSTEM_OUTPUT.get(model_name_for_maps, None)
        reward_head_indices = REWARD_HEADS_INDICES.get(model_name_for_maps, {}).get(script_args.dataset, None)

        if bool(script_args.use_frozen_base_model):
            if reward_head_indices is not None:
                if len(dataset.value_keys) != len(reward_head_indices):
                    raise ValueError(
                        "Mismatch between dataset value key count and reward head indices: "
                        f"{len(dataset.value_keys)} vs {len(reward_head_indices)}"
                    )

        # Read-only eval args; MORewardTrainer still needs TrainingArguments.
        eval_args = TrainingArguments(
            output_dir=str(script_args.results_dir),
            seed=int(script_args.seed),
            data_seed=int(script_args.seed),
            per_device_eval_batch_size=len(dataset.test_dataset),
            remove_unused_columns=False,
            bf16=bool(script_args.bf16),
            report_to="none",
            label_names=["labels"],
            use_cpu=bool(script_args.use_cpu),
            do_train=False,
            do_eval=True,
            save_strategy="no",
            logging_strategy="no",
        )

        # Keep these in sync with model config if base-model reward heads are active.
        model.config.base_model_reward_heads_module_name = reward_heads_module_name if bool(script_args.use_frozen_base_model) else model.config.base_model_reward_heads_module_name
        model.config.base_model_value_system_module_name = value_system_module_name if bool(script_args.use_frozen_base_model) else model.config.base_model_value_system_module_name

        # Use the exact same metric and loss functions as training.
        trainer = MORewardTrainer(
            model=model,
            args=eval_args,
            eval_dataset=dataset.test_dataset,
            compute_metrics=partial(
                MORewardTrainer.compute_metrics,
                config=model.config,
                training_variables=model.training_variables,
            ),
            compute_loss_func=partial(
                mo_compute_loss_func,
                config=model.config,
                training_variables=model.training_variables,
            ),
            data_collator=dc,
        )

        metrics_test = trainer.evaluate(eval_dataset=dataset.test_dataset, metric_key_prefix="test")
        flat_metrics_test = flatten_metrics_for_csv(metrics_test)
        write_metrics_csv(flat_metrics_test, script_args.results_dir, name="test_metrics.csv")

        print("Test evaluation complete.")
        print(f"Checkpoint: {script_args.checkpoint_path}")
        print(f"CSV saved to: {Path(script_args.results_dir).resolve()}")

        metrics_eval = trainer.evaluate(eval_dataset=dataset.eval_dataset, metric_key_prefix="eval")
        flat_metrics_eval = flatten_metrics_for_csv(metrics_eval)
        write_metrics_csv(flat_metrics_eval, script_args.results_dir, name="eval_metrics.csv")

        print("Eval evaluation complete.")
        print(f"Checkpoint: {script_args.checkpoint_path}")
        print(f"CSV saved to: {Path(script_args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
