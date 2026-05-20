#!/usr/bin/env python3

from dataclasses import dataclass, field
from functools import partial
import csv
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from enum import Enum
import json
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
from vsllib.utils import  ScriptArguments, argument_parser, flatten_metrics_for_csv, obtain_tokenizer, seed_everything, write_metrics_csv


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
    weights_only: bool = field(
        default=False,
        metadata={"help": "If True, only extract and save value system weights without running full evaluation."},
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

        # Auto-select if only one checkpoint available
        if len(subdirs) == 1:
            print(f"Only one checkpoint available: {subdirs[0]}. Auto-selecting.")
            current_path = current_path / subdirs[0]
            continue

        for i, item in enumerate(subdirs):
            print(f"  - ({i}) {item}")

        prompt = "Please enter the index of the valid checkpoint path from the above list"
        if allow_finish:
            prompt += " or OK to finish"
        prompt += ": "
        print(prompt)
        index_ = input().strip()
        if allow_finish and index_.upper() == "OK":
            return None

        try:
            current_path = current_path / subdirs[int(index_)]
        except (ValueError, IndexError):
            print(f"Invalid index entered: {index_}")
        print("trying...")


def _prompt_for_checkpoint_path(output_path: Path, *, allow_finish: bool, dataset: Optional[str] = None, already_selected: Optional[List[Path]] = None) -> Optional[Path]:
    already_selected = already_selected or []
    
    # Extract top-level checkpoint names from already selected paths
    already_selected_names = set()
    for p in already_selected:
        try:
            # Get the relative path from output_path and extract the top-level name
            rel_path = p.relative_to(output_path)
            top_level_name = rel_path.parts[0]
            already_selected_names.add(top_level_name)
        except ValueError:
            # If p is not relative to output_path, use the name
            already_selected_names.add(p.name)
    
    all_runs = [item for item in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, item)) and len(os.listdir(os.path.join(output_path, item))) > 0]
    
    # Filter: exclude already selected and those not matching dataset
    available_runs = []
    if isinstance(dataset, Enum):
        dataset_str = dataset.value.lower()
    else:   
        dataset_str = dataset.lower() if dataset else None
    for item in all_runs:
        if item in already_selected_names:
            continue
        if dataset_str and dataset_str not in item.lower():
            continue
        available_runs.append(item)
    
    if not available_runs:
        print(f"No available runs found in {output_path}" + (f" matching dataset '{dataset}'" if dataset else "") + ".")
        return None

    while True:
        print(f"Available runs in {output_path}" + (f" (matching dataset '{dataset}')" if dataset else "") + ":")
        for i, item in enumerate(available_runs):
            print(f"  - ({i}) {item}")

        prompt = "Please enter the index of the valid checkpoint path from the above list"
        if allow_finish:
            prompt += " or OK to finish"
        prompt += ": "
        print(prompt)
        index_ = input().strip()
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
        print("trying... 2")

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
            allow_finish=len(resolved_checkpoint_paths) > 0,
            dataset=script_args.dataset,
            already_selected=resolved_checkpoint_paths,
        )
        if prompted_path is not None:
            resolved_checkpoint_paths.append(prompted_path)

    while True:
        prompted_path = _prompt_for_checkpoint_path(
            output_path,
            allow_finish=len(resolved_checkpoint_paths) > 0,
            dataset=script_args.dataset,
            already_selected=resolved_checkpoint_paths,
        )
        if prompted_path is None:
            if len(resolved_checkpoint_paths) == 0:
                print(f"There are no valid checkpoint paths available for model type: {script_args.model_name}. Revise the folders in {output_path}")
                exit(0)
            assert len(resolved_checkpoint_paths) > 0, "At least one valid checkpoint path must be provided."
            break
        else:
            resolved_checkpoint_paths.append(prompted_path)
        print("trying... 3")
    print("RESOLVED.")
    results_root = Path(script_args.results_dir) if script_args.results_dir is not None and os.path.exists(script_args.results_dir) else Path(RESULTS_DIR)

    eval_script_args_all: List[EvalArguments] = []
    for checkpoint_path in resolved_checkpoint_paths:
        results_dir = results_root / removed_model_path_segment(str(checkpoint_path))
        eval_script_args_all.append(_build_eval_arguments(script_args, checkpoint_path, results_dir))

    return eval_script_args_all, preset




def load_training_args_from_checkpoint(checkpoint_path: str, default_batch_size: int = 8) -> tuple[int, int]:
    """
    Try to load per_device_train_batch_size from checkpoint's training_args.
    Supports both training_args.json and training_args.bin formats.
    Falls back to default if file doesn't exist or is missing the key.
    Returns (per_device_train_batch_size, per_device_eval_batch_size)
    """
    checkpoint_dir = Path(checkpoint_path)
    

    
    # Try BIN format with torch.load (transformers uses torch.save for .bin files)
    training_args_bin = checkpoint_dir / "training_args.bin"
    if training_args_bin.exists():
        try:
            training_args_obj = torch.load(training_args_bin, weights_only=False)
            batch_size = getattr(training_args_obj, "per_device_train_batch_size", default_batch_size)
            eval_batch_size = getattr(training_args_obj, "per_device_eval_batch_size", batch_size)
            print(f"Loaded training batch size from checkpoint (BIN): {batch_size}")
            return batch_size, eval_batch_size
        except Exception as e:
            print(f"Warning: Failed to load training_args.bin: {e}")
    
    print(f"Using default batch size: {default_batch_size}")
    return default_batch_size, default_batch_size


def extract_and_save_value_system_weights(model, results_dir: Path) -> None:
    """
    Extract value system weights from model and save to CSV.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    weights = model.value_system_layer.get_weights()
    
    # Convert weights to numpy if needed
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().detach().numpy()
    
    weights = np.asarray(weights).flatten()
    
    # Create CSV with weights for each value dimension
    csv_path = results_dir / "value_system_weights.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([f"value_{i}" for i in range(len(weights))])
        # Write weights row
        writer.writerow(weights.tolist())
    
    print(f"Value system weights saved to: {csv_path.resolve()}")





def main() -> None:
    script_args_all, preset = parse_eval_args()
    script_args_all: List[EvalArguments]
    
    print("EVAL ARGUMENTS PARSED")
    for script_args in script_args_all:
        with open(os.path.join(str(script_args.checkpoint_path), "seed_info.json"), "r", encoding="utf-8") as fp:
            seed_info = json.load(fp)
        script_args.seed = seed_info.get("seed", script_args.seed)
        script_args.data_seed = seed_info.get("dataseed", script_args.data_seed)
        seed_everything(int(script_args.seed))
        
        #torch_dtype = torch.bfloat16 if script_args.bf16 else torch.float32
        #print("TORCH DTYPE:", torch_dtype )
        print("LOADING TOKENIZER...")
        tokenizer = obtain_tokenizer(script_args, preset=preset, checkpoint_path=script_args.checkpoint_path)
        print("LOADED TOKENIZER")
        print(f"LOADING MODEL... ({script_args.checkpoint_path})")
        import subprocess

        def du(path):
            """disk usage in human readable format (e.g. '2,1GB')"""
            return subprocess.check_output(['du','-sh', path]).split()[0].decode('utf-8')
        checkpoint_size = du(script_args.checkpoint_path)
        print(f"Checkpoint size: {checkpoint_size}")

        model = MORMForSequenceClassification.from_pretrained(
            str(script_args.checkpoint_path),
        ).to(device="cuda:0")
        # Read seed info:
        """seed_info = {
            "seed": self.args.seed,
            "dataseed": self.args.data_seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "torch_initial_seed": int(th.initial_seed()),
        }
        with open(os.path.join(checkpoint_dir, "seed_info.json"), "w", encoding="utf-8") as fp:
            json.dump(seed_info, fp, indent=2, sort_keys=True)"""
        


        
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

         # Extract and save value system weights if requested
        if bool(script_args.weights_only):
            results_path = Path(script_args.results_dir)
            extract_and_save_value_system_weights(
                model,
                results_path
            )
            print(f"Weight extraction complete for checkpoint: {script_args.checkpoint_path}")
            model = model.to(device="cpu")
            embed_model = embed_model.to(device="cpu") if embed_model is not None else None
            del dataset, model, tokenizer, dc, embed_model
            torch.cuda.empty_cache()
            continue

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
            split_seed=int(script_args.data_seed),
            eval_proportion_or_indices=eval_proportion_or_indices,
            test_proportion_or_indices=test_proportion_or_indices,
            cleanup_cache_files=False,
        )
        if model.num_values != len(dataset.value_keys):
            raise ValueError(
                f"Model num_values ({model.num_values}) does not match dataset value key count ({len(dataset.value_keys)}). Perhaps you have loaded a model that is not compatible with the dataset? Check your checkpoint path and dataset choice."
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
        per_device_train_batch_size, per_device_eval_batch_size = load_training_args_from_checkpoint(
            str(script_args.checkpoint_path)
        )
        
        eval_args = TrainingArguments(
            output_dir=str(script_args.results_dir),
            seed=int(script_args.seed),
            data_seed=int(script_args.data_seed),
            per_device_eval_batch_size=per_device_eval_batch_size,
            per_device_train_batch_size=per_device_train_batch_size,
            remove_unused_columns=False,
            bf16=bool(script_args.bf16),
            logging_strategy="steps",
            logging_steps=1,
            report_to="none",
            label_names=["labels"],
            use_cpu=bool(script_args.use_cpu),
            do_train=False,
            do_eval=True,
            save_strategy="no",
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
        print(f"Starting test evaluation... {len(dataset.test_dataset)} examples")
        
        metrics_test = trainer.evaluate(eval_dataset=dataset.test_dataset, metric_key_prefix="test")
        flat_metrics_test = flatten_metrics_for_csv(metrics_test)
        write_metrics_csv(flat_metrics_test, script_args.results_dir, name="test_metrics.csv")

        print("Test evaluation complete.")
        print(f"Checkpoint: {script_args.checkpoint_path}")
        print(f"CSV saved to: {Path(script_args.results_dir).resolve()}")
        print("Starting eval evaluation...")
        metrics_eval = trainer.evaluate(eval_dataset=dataset.eval_dataset, metric_key_prefix="eval")
        flat_metrics_eval = flatten_metrics_for_csv(metrics_eval)
        write_metrics_csv(flat_metrics_eval, script_args.results_dir, name="eval_metrics.csv")

        print("Eval evaluation complete.")
        print(f"Checkpoint: {script_args.checkpoint_path}")
        print(f"CSV saved to: {Path(script_args.results_dir).resolve()}")

        model = model.to(device="cpu")
        embed_model = embed_model.to(device="cpu") if embed_model is not None else None
        del trainer, dataset, model, tokenizer, dc, embed_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
