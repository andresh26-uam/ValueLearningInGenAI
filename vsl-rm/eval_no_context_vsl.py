#!/usr/bin/env python3

from dataclasses import dataclass, field
from functools import partial
import csv
import os
from pathlib import Path
import sys
from typing import Any, Dict
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
    mo_compute_loss_func,
)
from vsllib.utils import  ScriptArguments, argument_parser, obtain_tokenizer, seed_everything


@dataclass
class EvalArguments(ScriptArguments):
    checkpoint_path: str = field(
        default=None,
        metadata={"help": "run name in the directory saved by no_context_vsl.py. If not supplied, will ask the user to select from the available runs in the output directory."},
    )
    results_dir: str = field(
        default=None,
        metadata={"help": "Path to output CSV file. (Normally saved under .results/script_args.output_dir/script_args.run_name/metrics.csv)"},
    )

def parse_eval_args() -> EvalArguments:
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

    


    if script_args.checkpoint_path is not None:
        checkpoint_path = os.path.join(script_args.output_path, script_args.checkpoint_path)
    else:
        checkpoint_path = None
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"Checkpoint path does not exist: {checkpoint_path}")
        print(f"Available runs in {script_args.output_path}:")
        for i, item in enumerate(os.listdir(script_args.output_path)):
            if os.path.isdir(os.path.join(script_args.output_path, item)):
                print(f"  - ({i}) {item}")
        index_ = input("Please enter the index of the valid checkpoint path from the above list: ").strip()
        try:
            script_args.checkpoint_path = os.path.join(script_args.output_path, os.listdir(script_args.output_path)[int(index_)])
    
        except (ValueError, IndexError):
            print(f"Invalid index entered: {index_}")
            sys.exit(1)
        checkpoint_path = script_args.checkpoint_path 
        if not os.path.exists(checkpoint_path):
            print(f"File does not exist: {checkpoint_path}")
            sys.exit(1)
        # if there is no config.json and seed_info.json in the checkpoint path, it is likely not a valid checkpoint directory
        if not (Path(checkpoint_path) / "config.json").exists() or not (Path(checkpoint_path) / "training_variables.pt").exists():
            print(f"Warning: The selected checkpoint path does not contain expected files (config.json and training_variables.pt).")
            print("Select subfolder with valid checkpoint.")
        for i, item in enumerate(os.listdir(checkpoint_path)):
            if os.path.isdir(os.path.join(checkpoint_path, item)):
                print(f"  - ({i}) {item}")
        index_ = input("Please enter the index of the valid checkpoint path from the above list: ").strip()
        try:
            script_args.checkpoint_path = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[int(index_)])
        except (ValueError, IndexError):
            print(f"Invalid index entered: {index_}")
            sys.exit(1)
    
    if script_args.results_dir is None or not os.path.exists(script_args.results_dir):
        script_args.results_dir = Path(RESULTS_DIR) / removed_model_path_segment(script_args.checkpoint_path)
        
    return args, preset



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
    script_args, preset = parse_eval_args()
    script_args: EvalArguments
    seed_everything(int(script_args.seed))

    
    torch_dtype = torch.bfloat16 if script_args.bf16 else torch.float32
    tokenizer = obtain_tokenizer(script_args, preset=preset, checkpoint_path=script_args.checkpoint_path)
    
    model = MORMForSequenceClassification.from_pretrained(
        str(script_args.checkpoint_path),
    )
    print("MODEL DETAILS:", model)
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

    metrics = trainer.evaluate(eval_dataset=dataset.test_dataset, metric_key_prefix="test")
    flat_metrics = flatten_metrics_for_csv(metrics)
    write_metrics_csv(flat_metrics, script_args.results_dir)

    print("Test evaluation complete.")
    print(f"Checkpoint: {script_args.checkpoint_path}")
    print(f"CSV saved to: {Path(script_args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
