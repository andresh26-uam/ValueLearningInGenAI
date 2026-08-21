#!/usr/bin/env python3
# SBATCH --job-name=ValueLearningInGenAI
# SBATCH --chdir=/home/aholg/ValueLearningInGenAI
from functools import partial
import os
from pathlib import Path
import sys
from pprint import pprint
from accelerate import PartialState
import accelerate
from dotenv import load_dotenv
import numpy as np
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

from vsllib.utils import ScriptArguments, argument_parser, maybe_assign_pad_token, obtain_tokenizer, sample_example_profiles_exact, sample_example_profiles_scipy, seed_everything
from vsllib.dataset_processing import FeatureBasedPreferenceDataset, PairwisePreferenceDataset
from vsllib.training_utils import MORewardDataCollator, MORewardDataCollatorWithPadding
from vsllib.training import ConstrainedOptimizer, CtxMORewardTrainer, MORewardTrainer
from vsllib.reward_models import MORMForClassification, MORMForSequenceClassification, MORMForClassificationConfig, mo_compute_loss_func
from vsllib.defines import MIN_EPSILON, HAS_UNDEFINED_LABELS, RESULTS_DIR, REWARD_HEADS_INDICES, REWARD_HEADS_OUTPUT, VALUE_SYSTEM_OUTPUT, EXTRA_KEYS, PROCESSED_DATASET_PATHS, ContextImplementations, get_test_indices, get_validation_indices
from vsllib.utils import flatten_metrics_for_csv, write_metrics_csv

load_dotenv()

def main_fun(script_args: ScriptArguments, training_args, tokenizer=None) -> None:
    
    with accelerate_state.main_process_first():
        torch_dtype = torch.bfloat16 if script_args.bf16 else torch.float32
        if script_args.task_type == "nlp_based":
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
                tokenizer=tokenizer, max_length=script_args.max_length, dtype=torch_dtype, use_embeddings=script_args.use_extracted_features) 

            dataset = PairwisePreferenceDataset(dataset_path, tokenizer,
                                                normalize_context=script_args.normalize_context_features,
                                                from_disk=True,
                                                extra_keep_keys=extra_keep_keys,
                                                retokenize=script_args.repostprocess,
                                                recalculate_embeddings=script_args.recalculate_features,
                                                use_embeddings=script_args.use_extracted_features,
                                                model_reference=base_model,
                                                collator=dc,
                                                split_seed=int(training_args.data_seed),
                                                eval_proportion_or_indices=eval_proportion_or_indices,
                                                test_proportion_or_indices=test_proportion_or_indices,
                                                cleanup_cache_files=bool(
                                                    script_args.cleanup_dataset_cache_files),
                                                )
        else:
            pad_token_id = None
            dc = MORewardDataCollator(dtype=torch_dtype)
            dataset = FeatureBasedPreferenceDataset(dataset_path, 
                                                    from_disk=True,
                                                normalize_context=script_args.normalize_context_features,
                                                extra_keep_keys=extra_keep_keys,
                                                repostprocess=script_args.repostprocess,
                                                recalculate_features=script_args.recalculate_features,
                                                use_extracted_features=script_args.use_extracted_features,
                                                
                                                collator=dc,
                                                split_seed=int(training_args.data_seed),
                                                eval_proportion_or_indices=eval_proportion_or_indices,
                                                test_proportion_or_indices=test_proportion_or_indices,
                                                cleanup_cache_files=bool(
                                                    script_args.cleanup_dataset_cache_files),
                                                )
        if script_args.discordance_epsilon is None:
            suggested_epsilon =dataset.calculate_suggested_epsilon()
        else:
            suggested_epsilon = script_args.discordance_epsilon
        suggested_epsilon = max(suggested_epsilon, MIN_EPSILON)  # Avoid too small epsilon
        print("Suggested discordance_epsilon based on eval dataset: ", suggested_epsilon)
        
        print("Training set: ", len(dataset.train_dataset), " Eval set: ", len(
            dataset.eval_dataset), " Test set: ", len(dataset.test_dataset))
        num_values_to_use = len(dataset.value_keys)
        print("Dataset value keys: ", dataset.value_keys,
            "\n Total number of values: ", num_values_to_use)
    
    if script_args.do_train:
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

        if script_args.task_type == "feature_based":
            input_size = np.array(dataset.train_dataset["grounding_features_1"][0]).shape[0]
            input_size_vs = np.array(dataset.train_dataset["context_features"][0]).shape[0]
            
        elif script_args.task_type == "nlp_based":
            input_size, input_size_vs = MORMForSequenceClassification.infer_model_inputs_sizes(base_model)
            
        else:

            raise ValueError(f"Unrecognized task type {script_args.task_type}")
        
        mo_config = MORMForClassificationConfig(
            do_initialization=script_args.do_initialization,

            direct_gmm=script_args.direct_gmm,
            sharp_context_classification=script_args.sharp_context_classification,
            detach_vs_selection_for_value_system_weight_training=script_args.detach_vs_selection_for_value_system_weight_training,
            detach_context_selection_for_value_system_selection=script_args.detach_context_selection_for_value_system_selection,
            ctx_coefficient=script_args.ctx_coefficient,
            entropy_coefficient=script_args.entropy_coefficient,
            vs_weight_initialization=script_args.vs_weight_initialization,
            vs_selection_coefficient=script_args.vs_selection_coefficient,
            training_initialization_data_size=script_args.training_initialization_data_size,
            max_contexts=script_args.max_contexts,
            max_value_systems=script_args.max_value_systems,
            vs_layer_hidden_sizes=[script_args.vs_hidden_size]*script_args.vs_num_hidden_layers,
            vs_layer_dropout=script_args.vs_layer_dropout,
            vs_layer_intermediate_activation=script_args.vs_layer_activation,
            context_implementation=script_args.context_implementation,
            
            input_size=input_size,
            input_size_vs=input_size_vs,
            activate_discordance_epsilon_for_loss=script_args.activate_discordance_epsilon_for_loss,
            check_undefined_label=HAS_UNDEFINED_LABELS[script_args.dataset],
            pad_token_id=pad_token_id,
            num_values=len(dataset.value_keys),
            dtype=str(torch_dtype).replace("torch.", ""),
            assume_qualitative_labels=script_args.assume_qualitative_labels,
            use_validation_for_tendencies=script_args.use_validation_for_tendencies,
            update_tendencies_every_n_steps=script_args.update_tendencies_every_n_steps,
            discordance_epsilon=suggested_epsilon,
            base_model_name_or_path=script_args.model_name,
            base_model_trust_remote_code=True,
            base_model_num_labels=1,
            loss_func_type=script_args.loss_func_type,
            loss_func_kwargs=script_args.loss_func_type_kwargs,
            lambda_decay=script_args.lambda_decay,
            hidden_sizes=[script_args.hidden_size]*script_args.num_hidden_layers, 
            value_layer_dropout=script_args.value_layer_dropout,
            value_layer_intermediate_activation=script_args.vs_layer_activation,
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
            lr_context=script_args.context_learning_rate,
            lr_lambda=script_args.lagrange_learning_rate,
        )

        if script_args.task_type == "feature_based":
            mo_model = MORMForClassification(
                config=mo_config, )
        elif script_args.task_type == "nlp_based":
            mo_model = MORMForSequenceClassification(
                config=mo_config, base_model=model)
        else:
            raise ValueError(f"Unrecognized task type {script_args.task_type}")
        
        

        sub_optimizer_cls, sub_optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            training_args, mo_model)

        print("Sub optimizer class: ", sub_optimizer_cls,
            " Sub optimizer kwargs: ", sub_optimizer_kwargs)
        sub_optimizer_kwargs = sub_optimizer_kwargs or {}
        sub_optimizer_kwargs["weight_decay"] = training_args.weight_decay

        print("Script arguments: ")

        pprint(vars(script_args))
        
        # exit(0)
        if ContextImplementations(mo_config.context_implementation) == ContextImplementations.NO_CONTEXT:
            trainer_class = MORewardTrainer 
            trainer_extra_kwargs = dict(
                compute_loss_func=partial(
                    mo_compute_loss_func, config=mo_config, training_variables=mo_model.training_variables),
            )
        elif ContextImplementations(mo_config.context_implementation) in [ContextImplementations.BASIC,ContextImplementations.GMM,ContextImplementations.BASIC_SMOOTH, ContextImplementations.BASIC_HARSH]:
            trainer_class = CtxMORewardTrainer
            trainer_extra_kwargs = dict(
                compute_loss_func=partial(
                    mo_compute_loss_func, config=mo_config, training_variables=mo_model.training_variables),
            )
        elif ContextImplementations(mo_config.context_implementation) in [ContextImplementations.DIRECT_VS,] :
            trainer_class = CtxMORewardTrainer
            trainer_extra_kwargs = dict(
                compute_loss_func=partial(
                    mo_compute_loss_func, config=mo_config, training_variables=mo_model.training_variables),
            )
        else:
            raise NotImplementedError(f"This type of context implementation is not implemented yet. {mo_config.context_implementation}")
        
        trainer: Trainer = trainer_class(
            model=mo_model,
            args=training_args,
            train_dataset=dataset.train_dataset if not script_args.use_frozen_base_model else dataset.train_dataset.select(list(range(min(len(dataset.train_dataset), training_args.per_device_train_batch_size * mo_config.gradient_accumulation_steps*2)))),  # TODO: RESET THIS!!
            eval_dataset=dataset.eval_dataset,
            compute_metrics=partial(trainer_class.compute_metrics_custom,
                                    config=mo_config, training_variables=mo_model.training_variables),
            optimizer_cls_and_kwargs=(ConstrainedOptimizer, {
                'params_gr': list(mo_model.grounding_parameters()),
                'params_gr_ideal': list(mo_model.reward_heads_ideal.parameters()) if script_args.use_ideal_grounding_model else None,
                'params_vs': list(mo_model.value_system_parameters()),
                'params_ctx': list(mo_model.context_parameters()),
                'n_values': mo_config.num_values,
                'lr_value_system': mo_config.lr_value_system,
                'lr_grounding': mo_config.lr_grounding,
                'lr_context': mo_config.lr_context,
                'lr_lambda': mo_config.lr_lambda,
                'loss_func_type': mo_config.loss_func_type,
                'loss_func_type_kwargs': mo_config.loss_func_type_kwargs,
                'sub_optimizer_class': sub_optimizer_cls,
                'training_variables': mo_model.training_variables,
                ** sub_optimizer_kwargs
            }),
            data_collator=dc,
            **trainer_extra_kwargs
        )
        # trainer.train()
        print("Saving last checkpoint of the model")
        print(mo_model.training_variables.lagrange_multipliers)
        print("Starting trainer.train()", flush=True)
        
        print("EVALUATING")
        if not script_args.use_frozen_base_model:
            ret = trainer.evaluate()
        print("EVALUATED: ")
        pprint(ret)
        trainer.train()
        print("TRAINING FINISHED")
        trainer.evaluate()

        if script_args.use_frozen_base_model:
            print("Starting test evaluation...")
            metrics_eval = trainer.evaluate(eval_dataset=dataset.test_dataset, metric_key_prefix="test")
            flat_metrics_eval = flatten_metrics_for_csv(metrics_eval)
            write_metrics_csv(flat_metrics_eval, os.path.join(RESULTS_DIR, script_args.model_name, script_args.run_name), name="test_metrics.csv")

            print("Starting eval evaluation...")
            metrics_eval = trainer.evaluate(eval_dataset=dataset.eval_dataset, metric_key_prefix="eval")
            flat_metrics_eval = flatten_metrics_for_csv(metrics_eval)
            write_metrics_csv(flat_metrics_eval, os.path.join(RESULTS_DIR, script_args.model_name, script_args.run_name), name="eval_metrics.csv")

        return trainer, dataset


if __name__ == "__main__":
    
    
    accelerate_state = PartialState()
    using_accelerate = accelerate_state.num_processes > 1
    is_main_accelerate_process = accelerate_state.is_main_process if using_accelerate else True

    # Configure W&B only on the main process under Accelerate.
    if is_main_accelerate_process:
        os.environ.setdefault("WANDB_INIT_TIMEOUT", "600")
        os.environ.setdefault("WANDB__SERVICE_WAIT", "600")

    # Parse/configure once on the main process, then broadcast to workers.
    if using_accelerate and torch.distributed.is_available() and torch.distributed.is_initialized():
        shared_config = [None]
        if is_main_accelerate_process:
            parser = HfArgumentParser(ScriptArguments) 
            main_script_args = parser.parse_args_into_dataclasses()[0]
            main_script_args, main_preset = argument_parser(main_script_args)
            shared_config[0] = (vars(main_script_args), main_preset)
        torch.distributed.broadcast_object_list(shared_config, src=0)
        script_args_dict, preset = shared_config[0]
        script_args = ScriptArguments(**script_args_dict)
    else:
        parser = HfArgumentParser(ScriptArguments) 
        script_args = parser.parse_args_into_dataclasses()[0]
        script_args, preset = argument_parser(script_args)

    accelerate_state.on_main_process()
    seed_everything(int(script_args.seed))

    dataset_path = PROCESSED_DATASET_PATHS[script_args.dataset]
    extra_keep_keys = EXTRA_KEYS[script_args.dataset]
    test_proportion_or_indices = get_test_indices(script_args.dataset)
    eval_proportion_or_indices = get_validation_indices(script_args.dataset)

    output_dir = os.path.join(script_args.output_path, script_args.run_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=int(script_args.seed),
        data_seed=int(script_args.data_seed),
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        eval_strategy="steps",
        eval_steps=script_args.eval_every_steps,
        save_strategy="steps" if script_args.do_checkpointing else "no",
        save_steps=script_args.save_every_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing if script_args.task_type == "nlp_based" else False,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=1,
        optim_args={},
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=0,
        label_names=["labels"],
        report_to="wandb" if is_main_accelerate_process else "none",
        #report_to="none",
        max_grad_norm=script_args.max_grad_norm,
        run_name=script_args.run_name,
        use_cpu=script_args.use_cpu,
    )

    if script_args.task_type == "nlp_based":
        tokenizer = obtain_tokenizer(script_args, preset)
    else: 
        tokenizer= None

    trainer, dataset = main_fun(script_args, training_args, tokenizer)
    trainer : CtxMORewardTrainer
    dataset : PairwisePreferenceDataset
    @accelerate_state.on_main_process
    def saving():
        
        if script_args.do_save:

                save_location = trainer.save_with_seed(checkpoint_name="last_checkpoint")
                if script_args.task_type == "nlp_based":
                    mo_model = MORMForSequenceClassification.from_pretrained(save_location)
                else:
                    mo_model = MORMForClassification.from_pretrained(save_location)
                print("TRAINED MODEL", mo_model)
                print(mo_model.training_variables.lagrange_multipliers)
                trainer.model = mo_model
                
                print(trainer.evaluate(eval_dataset=dataset.test_dataset, metric_key_prefix="test"))

    saving()
        
