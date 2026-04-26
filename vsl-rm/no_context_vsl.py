#!/usr/bin/env python3
# SBATCH --job-name=ValueLearningInGenAI
# SBATCH --chdir=/home/aholg/ValueLearningInGenAI
from functools import partial
import os
from pathlib import Path
import sys
from pprint import pprint
from dotenv import load_dotenv
from transformers import Trainer
# import evaluate
import torch
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

USE_CPU = False
# Make local package imports robust when sbatch executes from a temporary path.
for candidate in (
    Path(__file__).resolve().parent,
    Path.cwd() / "vsl-rm",
    Path(os.getenv("HOME", "")) / "ValueLearningInGenAI" / "vsl-rm",
):
    if (candidate / "vsllib").exists():
        sys.path.insert(0, str(candidate))
        break

from vsllib.utils import ScriptArguments, argument_parser, maybe_assign_pad_token, obtain_tokenizer, seed_everything
from vsllib.dataset_processing import PairwisePreferenceDataset
from vsllib.training_utils import MORewardDataCollatorWithPadding
from vsllib.training import ConstrainedOptimizer, MORewardTrainer
from vsllib.reward_models import MORMForSequenceClassification, MORMForSequenceClassificationConfig, mo_compute_loss_func
from vsllib.defines import HAS_UNDEFINED_LABELS, REWARD_HEADS_INDICES, REWARD_HEADS_OUTPUT, VALUE_SYSTEM_OUTPUT, EXTRA_KEYS, PROCESSED_DATASET_PATHS, get_test_indices, get_validation_indices


load_dotenv()

parser = HfArgumentParser(ScriptArguments)  # type: ignore
script_args = parser.parse_args_into_dataclasses()[0]
script_args, preset = argument_parser(script_args)
seed_everything(int(script_args.seed))

tokenizer = obtain_tokenizer(script_args, preset)

dataset_path = PROCESSED_DATASET_PATHS[script_args.dataset]
extra_keep_keys = EXTRA_KEYS[script_args.dataset]
test_proportion_or_indices = get_test_indices(script_args.dataset)
eval_proportion_or_indices = get_validation_indices(script_args.dataset)

output_dir = os.path.join(script_args.output_path, script_args.run_name)

training_args = TrainingArguments(
    output_dir=output_dir,
    seed=int(script_args.seed),
    data_seed=int(script_args.seed),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    eval_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim_args={},
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=0,  # TODO. 50?
    label_names=["labels"],
    report_to="wandb",  # 'wandb'
    max_grad_norm=script_args.max_grad_norm,  # TODO 0.01?
    run_name=script_args.run_name,
    # report_to=None, # 'wandb'
    use_cpu=script_args.use_cpu,
)


def main_fun() -> None:
    torch_dtype = torch.bfloat16 if script_args.bf16 else torch.float32

    model_kwargs = dict(dtype=torch_dtype, trust_remote_code=True)
    if preset.get("use_flash_attention_2", False):
        model_kwargs["use_flash_attention_2"] = True

    model_full = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, **model_kwargs)
    base_model = model_full.base_model if hasattr(
        model_full, 'base_model') else model_full

    if script_args.use_frozen_base_model:
        # For models like ArmoRM that have built-in multiple reward structure
        model = model_full
    else:
        model = base_model

    for mod in [model, base_model]:
        maybe_assign_pad_token(mod, script_args, tokenizer, preset)

    pad_token_id = model.config.pad_token_id

    dc = MORewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length, dtype=torch_dtype, use_embeddings=script_args.use_embeddings)  # type: ignore

    dataset = PairwisePreferenceDataset(dataset_path, tokenizer,
                                        from_disk=True,
                                        extra_keep_keys=extra_keep_keys,
                                        retokenize=script_args.retokenize,
                                        recalculate_embeddings=script_args.recalculate_embeddings,
                                        use_embeddings=script_args.use_embeddings,
                                        model_reference=base_model,
                                        collator=dc,
                                        split_seed=int(42),
                                        eval_proportion_or_indices=eval_proportion_or_indices,
                                        test_proportion_or_indices=test_proportion_or_indices,
                                        cleanup_cache_files=bool(
                                            script_args.cleanup_dataset_cache_files),
                                        )
    print("Training set: ", len(dataset.train_dataset), " Eval set: ", len(
        dataset.eval_dataset), " Test set: ", len(dataset.test_dataset))
    num_values_to_use = len(dataset.value_keys)
    print("Dataset value keys: ", dataset.value_keys,
          "\n Total number of values: ", num_values_to_use)

    if script_args.use_frozen_base_model:
        reward_heads_module_name = REWARD_HEADS_OUTPUT.get(
            script_args.model_name, None)
        value_system_module_name = VALUE_SYSTEM_OUTPUT.get(
            script_args.model_name, None)
        reward_head_indices = REWARD_HEADS_INDICES.get(
            script_args.model_name, {}).get(script_args.dataset, None)

        if reward_head_indices is not None:
            print(f"Using reward head indices: {reward_head_indices}")
            assert num_values_to_use == len(
                reward_head_indices), f"Number of values to use ({num_values_to_use}) does not match the length of reward head indices ({len(reward_head_indices)})"

    mo_config = MORMForSequenceClassificationConfig(
        check_undefined_label=HAS_UNDEFINED_LABELS[script_args.dataset],
        pad_token_id=pad_token_id,
        num_values=len(dataset.value_keys),
        dtype=str(torch_dtype).replace("torch.", ""),
        base_model_name_or_path=script_args.model_name,
        base_model_trust_remote_code=True,
        base_model_num_labels=1,
        loss_func_type=script_args.loss_func_type,
        loss_func_kwargs=script_args.loss_func_type_kwargs,
        lambda_decay=script_args.lambda_decay,
        hidden_sizes=[script_args.hidden_size]*script_args.num_hidden_layers, value_layer_dropout=script_args.value_layer_dropout,
        value_layer_intermediate_activation=script_args.layer_activation,
        value_layer_final_activation=script_args.final_layer_activation,
        layer_normalization=script_args.layer_normalization,
        grounding_loss_tendency_update_ratio=script_args.grounding_loss_tendency_update_ratio,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        use_metrics_or_losses_for_lagrange_updates=script_args.use_metrics_or_losses_for_lagrange_updates,
        use_exponential_moving_average_or_optimum_targets=script_args.use_exponential_moving_average_or_optimum_targets,
        grad_on_only_worst_value=script_args.grad_on_only_worst_value,
        zero_constraint=script_args.zero_constraint,
        use_ideal_grounding_model=script_args.use_ideal_grounding_model,
        rew_center_coefficient=script_args.rew_center_coefficient,
        base_model_reward_heads_module_name=reward_heads_module_name if script_args.use_frozen_base_model else None,
        base_model_value_system_module_name=value_system_module_name if script_args.use_frozen_base_model else None,
        base_model_reward_head_indices=reward_head_indices if script_args.use_frozen_base_model else None,
        use_base_model_heads=script_args.use_frozen_base_model,
        gather_train_metrics=script_args.gather_train_metrics,
        lr_value_system=script_args.learning_rate,
        lr_grounding=script_args.grounding_learning_rate,
        lr_lambda=script_args.lagrange_learning_rate,
    )

    mo_model = MORMForSequenceClassification(
        config=mo_config, base_model=model)

    sub_optimizer_cls, sub_optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
        training_args, mo_model)

    print("Sub optimizer class: ", sub_optimizer_cls,
          " Sub optimizer kwargs: ", sub_optimizer_kwargs)

    print("Script arguments: ")

    pprint(vars(script_args))
    # exit(0)
    trainer: Trainer = MORewardTrainer(
        model=mo_model,
        args=training_args,
        train_dataset=dataset.train_dataset,  # TODO: RESET THIS!!
        eval_dataset=dataset.eval_dataset,
        compute_metrics=partial(MORewardTrainer.compute_metrics,
                                config=mo_config, training_variables=mo_model.training_variables),
        compute_loss_func=partial(
            mo_compute_loss_func, config=mo_config, training_variables=mo_model.training_variables),
        optimizer_cls_and_kwargs=(ConstrainedOptimizer, {
            'params_gr': list(mo_model.grounding_parameters()),
            'params_gr_ideal': list(mo_model.reward_heads_ideal.parameters()) if script_args.use_ideal_grounding_model else None,
            'params_vs': list(mo_model.value_system_parameters()),
            'n_values': mo_config.num_values,
            'lr_value_system': mo_config.lr_value_system,
            'lr_grounding': mo_config.lr_grounding,
            'lr_lambda': mo_config.lr_lambda,
            'loss_func_type': mo_config.loss_func_type,
            'loss_func_type_kwargs': mo_config.loss_func_type_kwargs,
            'sub_optimizer_class': sub_optimizer_cls,
            'training_variables': mo_model.training_variables,
            ** sub_optimizer_kwargs
        }),
        data_collator=dc,
    )
    # trainer.train()
    print("Saving last checkpoint of the model")
    print(mo_model.training_variables.lagrange_multipliers)

    trainer.train()
    save_location = trainer.save_with_seed(checkpoint_name="last_checkpoint")

    mo_model = MORMForSequenceClassification.from_pretrained(save_location)
    print("TRAINED MODEL", mo_model)
    print(mo_model.training_variables.lagrange_multipliers)


if __name__ == "__main__":

    main_fun()
