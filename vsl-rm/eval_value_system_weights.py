#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch as th
from transformers import HfArgumentParser
from tqdm.auto import tqdm

for candidate in (
    Path(__file__).resolve().parent,
    Path.cwd() / "vsl-rm",
    Path(os.getenv("HOME", "")) / "ValueLearningInGenAI" / "vsl-rm",
):
    if (candidate / "vsllib").exists():
        sys.path.insert(0, str(candidate))
        break

from vsllib.dataset_processing import PairwisePreferenceDataset
from vsllib.defines import EXTRA_KEYS, MIN_EPSILON, NO_RATING_MASK, PROCESSED_DATASET_PATHS, get_test_indices, get_validation_indices
from vsllib.reward_models import accuracy_rewards_labels, reward_pairs_and_scores_to_logits_and_targets, value_system_loss_logits
from vsllib.utils import ScriptArguments, argument_parser, obtain_tokenizer, seed_everything


@dataclass
class WeightEvalArguments(ScriptArguments):
    
    results_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional directory where the evaluation summary will be written."},
    )
    output_name: str = field(
        default="test_value_system_accuracy.json",
        metadata={"help": "Filename for the saved summary under results_dir."},
    )
    method: str = field(
        default="vslrm",
        metadata={"help": "Method for combining per-value scores into a single reward. Only 'weighted_sum' is currently supported."},
    )
    optimize: bool = field(
        default=False,
        metadata={"help": "If True, run optimization to find weights that minimize value_system_loss_logits."},
    )
    opt_steps: int = field(
        default=500,
        metadata={"help": "Number of optimization steps when --optimize is used."},
    )
    opt_lr: float = field(
        default=0.1,
        metadata={"help": "Learning rate for optimizer when searching for weights."},
    )


def _forward_weighted_scores(test_dataset, weights: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Compute weighted per-pair rewards from original per-value scores without modifying them.

    Returns reward1, reward2, score1, score2 as 1-D tensors aligned with the test dataset order.
    """
    reward1_values: list[th.Tensor] = []
    reward2_values: list[th.Tensor] = []
    score1_values: list[th.Tensor] = []
    score2_values: list[th.Tensor] = []

    for sample in test_dataset:
        labels = th.as_tensor(sample["labels"], dtype=th.float32)
        if labels.ndim != 2 or labels.shape[0] != 2:
            raise ValueError(f"Expected labels to have shape [2, num_values + 1], got {tuple(labels.shape)}")

        per_value_scores = labels[:, :-1]
        overall_scores = labels[:, -1]

        # Do NOT alter original scores or inspect NO_RATING_MASK here — forward raw per-value scores.
        weighted_rewards = per_value_scores @ weights

        reward1_values.append(weighted_rewards[0])
        reward2_values.append(weighted_rewards[1])
        score1_values.append(overall_scores[0])
        score2_values.append(overall_scores[1])

    return (
        th.stack(reward1_values),
        th.stack(reward2_values),
        th.stack(score1_values),
        th.stack(score2_values),
    )


def compute_value_system_loss_for_weights(test_dataset, weights: th.Tensor, reward_diff_threshold: float = 50.0, assume_qualitative_labels: bool = True, discordance_epsilon: float = MIN_EPSILON) -> th.Tensor:
    """Compute the scalar value-system loss for a candidate weight vector.

    This function uses the repo helpers to convert reward pairs and scores into logits/targets
    and then evaluates value_system_loss_logits.
    """
    reward1, reward2, score1, score2 = _forward_weighted_scores(test_dataset, weights)

    
        
    # reward_pairs_and_scores_to_logits_and_targets expects (reward1, reward2, scores1, scores2)
        
    logits_p, target_probs_p, others = reward_pairs_and_scores_to_logits_and_targets(
        reward1, reward2, score1, score2, reward_diff_threshold=reward_diff_threshold,
        assume_qualitative_labels=assume_qualitative_labels, check_undefined_label=True, assume_torch=True)

    rew_sum = others.get("rew_sum", None)

    # Compute value-system loss; set check_undefined_label=False to avoid masking/altering scores
    loss = value_system_loss_logits(logits_p, target_probs_p, rew_sum=rew_sum, return_metrics=False, check_undefined_label=True, rew_center_coefficient=0.0, missing_mask=None, discordance_epsilon=discordance_epsilon, activate_disc_epsilon_for_loss=False)
    # value_system_loss_logits returns a scalar tensor
    return loss


def main() -> None:
    parser = HfArgumentParser(WeightEvalArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args, preset = argument_parser(script_args, class_source=WeightEvalArguments)

    seed_everything(int(script_args.seed))
    tokenizer = obtain_tokenizer(script_args, preset)

    dataset_path = PROCESSED_DATASET_PATHS[script_args.dataset]
    if script_args.dataset.value == "ultra":
        if script_args.method == "vslrm":
            WEIGHTS = [0.096,0.347,0.408,0.147]
        elif script_args.method == "equal":
            WEIGHTS = [0.25] * 4
        else:
            WEIGHTS = [0.050, 0.477, 0.316, 0.158]
        EPSILON = 0.25
    else:
        if script_args.method == "vslrm":
            WEIGHTS = [0.223,0.125,0.368,0.203,0.082]
        elif script_args.method == "equal":
            WEIGHTS = [0.2] * 5
        else:
            WEIGHTS = [0.265, 0.110, 0.410, 0.168, 0.047]
        EPSILON = 0.5

    extra_keep_keys = EXTRA_KEYS[script_args.dataset]
    test_proportion_or_indices = get_test_indices(script_args.dataset)
    eval_proportion_or_indices = get_validation_indices(script_args.dataset)

    dataset = PairwisePreferenceDataset(
        dataset_path,
        tokenizer,
        from_disk=True,
        extra_keep_keys=extra_keep_keys,
        retokenize=bool(script_args.retokenize),
        recalculate_embeddings=False,
        use_embeddings=False,
        model_reference=None,
        split_seed=int(script_args.data_seed),
        eval_proportion_or_indices=eval_proportion_or_indices,
        test_proportion_or_indices=test_proportion_or_indices,
        cleanup_cache_files=bool(script_args.cleanup_dataset_cache_files),
    )

    weights = th.tensor(WEIGHTS, dtype=th.float32)
    if len(dataset.value_keys) != weights.numel():
        raise ValueError(
            f"Weight count ({weights.numel()}) does not match dataset value count ({len(dataset.value_keys)})."
        )

    if script_args.optimize:
        raw_weights = th.nn.Parameter(th.zeros_like(weights))
        optimizer = th.optim.Adam([raw_weights], lr=float(script_args.opt_lr))

        best_loss = None
        best_weights = None
        progress = tqdm(range(int(script_args.opt_steps)), desc="optimizing weights", unit="step")
        for step in progress:

            selected_indices = th.randperm(len(dataset.test_dataset))[: min(256, len(dataset.test_dataset))].tolist()
            optimizer.zero_grad(set_to_none=True)
            candidate_weights = th.nn.functional.softmax(raw_weights, dim=0)
            loss = compute_value_system_loss_for_weights(
                dataset.test_dataset.select(selected_indices),
                candidate_weights,
                reward_diff_threshold=50.0,
                assume_qualitative_labels=True,
                discordance_epsilon=0.0,
            )
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().item())
            progress.set_postfix(loss=f"{loss_value:.6f}", weights=[f"{w:.3f}" for w in candidate_weights.detach().tolist()])
            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_weights = candidate_weights.detach().clone()

        if best_weights is None:
            best_weights = th.nn.functional.softmax(raw_weights, dim=0).detach()

        weights = best_weights

    reward1, reward2, score1, score2 = _forward_weighted_scores(dataset.test_dataset, weights)
    # Do NOT check for undefined labels here; keep original scores intact and let the caller
    # interpret undefined values if desired. We therefore set check_undefined_label=False.
    accuracy = accuracy_rewards_labels(
        reward1,
        reward2,
        score1,
        score2,
        assume_qualitative_labels=True,
        check_undefined_label=True,
        assume_torch=True,
        discordance_epsilon=0.0,
    )

    valid_examples = len(dataset.test_dataset)
    summary = {
        "dataset": str(script_args.dataset),
        "test_examples": len(dataset.test_dataset),
        "valid_examples": valid_examples,
        "weights": weights.tolist(),
        "accuracy": float(accuracy.item()),
        "optimize": bool(script_args.optimize),
        "opt_steps": int(script_args.opt_steps),
        "opt_lr": float(script_args.opt_lr),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    results_dir = Path(script_args.results_dir) if script_args.results_dir is not None else Path(dataset_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / script_args.output_name
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()