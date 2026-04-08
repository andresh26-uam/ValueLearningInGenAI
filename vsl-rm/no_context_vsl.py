#!/usr/bin/env python3
#SBATCH --job-name=ValueLearningInGenAI
#SBATCH --chdir=/home/aholg/ValueLearningInGenAI

from dataclasses import dataclass, field
from functools import partial
import json
import os
from pathlib import Path
import random
import sys

# Make local package imports robust when sbatch executes from a temporary path.
for candidate in (
    Path(__file__).resolve().parent,
    Path.cwd() / "vsl-rm",
    Path(os.getenv("HOME", "")) / "ValueLearningInGenAI" / "vsl-rm",
):
    if (candidate / "vsllib").exists():
        sys.path.insert(0, str(candidate))
        break

from transformers.utils import PaddingStrategy

from typing import Any, Dict, List, Optional, Union

from transformers import Trainer
# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


from vsllib.defines import ULTRAFEEDBACK_EXTRA_KEYS, ULTRAFEEDBACK_PROCESSED_PATH
from vsllib.reward_models import MORMForSequenceClassification, MORMForSequenceClassificationConfig, mo_compute_loss_func
from vsllib.training import ConstrainedOptimizer, MORewardTrainer, PairwisePreferenceDataset
from vsllib.utils import MORewardDataCollatorWithPadding

# Define and parse arguments.

# IMport HF_TOKEN from .env
from dotenv import load_dotenv
load_dotenv()

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    retokenize: Optional[bool] = field(
		default=False, metadata={"help": "Whether to retokenize the dataset. Set this to False if you have already tokenized and saved the dataset to disk, and just want to load it."})
    recalculate_embeddings: Optional[bool] = field(
        default=False, metadata={"help": "Whether to recalculate embeddings for the dataset."})
    save_embedded_dataset: Optional[bool] = field(
        default=True, metadata={"help": "Whether to save the tokenized+embedded dataset to disk."})
    cleanup_dataset_cache_files: Optional[bool] = field(
        default=True, metadata={"help": "Whether to remove temporary Hugging Face dataset cache files after preprocessing."})
    deepspeed: Optional[str] = field(
        # default="dp3.json",
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=32)
    per_device_eval_batch_size: Optional[int] = field(default=32)
    gradient_accumulation_steps: Optional[int] = field(default=5) # TODO 32?
    metrics_accumulation_steps: Optional[int] = field(default=5) # TODO 32?
    learning_rate: Optional[float] = field(default=1e-4)
    lambda_decay: Optional[float] = field(default=1e-5)
    rew_center_coefficient: Optional[float] = field(default=0.01) # TODO Recommended by TRL library (RewardTrainer)
    grounding_learning_rate: Optional[float] = field(default=1e-5) # TODO must be > 1e-4 to make any effect??
    lagrange_learning_rate: Optional[float] = field(default=0.0) # TODO 0.01
    grounding_loss_tendency_update_ratio: Optional[float] = field(default=0.02)
    use_metrics_or_losses_for_lagrange_updates: Optional[str] = field(default="metrics")
    grad_on_only_worst_value: Optional[bool] = field(default=True)
    zero_constraint: Optional[bool] = field(default=True)
    use_ideal_grounding_model : Optional[bool] = field(default=False)


    weight_decay: Optional[float] = field(default=0.0001)
    model_name: Optional[str] = field(
        #default="mistralai/Mistral-7B-Instruct-v0.2",
        #default="meta-llama/Llama-3.2-1B",
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default=ULTRAFEEDBACK_PROCESSED_PATH,
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    
    output_path: Optional[str] = field(
        default=f"./models/baselines/no_context_vsl-",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=200,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        #default=999999,
        default=50,
        metadata={"help": "Eval the model every x steps"},
    )
    inner_optimization_iterations: Optional[int] = field(
        default=5,
        metadata={"help": "Inner optimization iterations for the ideal model."},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Global seed for Python, NumPy, PyTorch, and Transformers."},
    )

parser = HfArgumentParser(ScriptArguments) # type: ignore
script_args = parser.parse_args_into_dataclasses()[0]


def seed_everything(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def save_checkpoint_with_seed(trainer: Trainer, tokenizer: AutoTokenizer, checkpoint_dir: str, seed: int):
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    seed_info = {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_initial_seed": int(torch.initial_seed()),
    }
    with open(os.path.join(checkpoint_dir, "seed_info.json"), "w", encoding="utf-8") as fp:
        json.dump(seed_info, fp, indent=2, sort_keys=True)


seed_everything(int(script_args.seed))

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length

# Get the dataset
train_path = script_args.train_set_path
output_name = script_args.output_path + script_args.model_name.split("/")[-1]

training_args = TrainingArguments(
    output_dir=output_name,
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
    #label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim_args={ },
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    #warmup_ratio=0.03,
    warmup_steps=0, # TODO. 50?
    label_names=["labels"],
    report_to="wandb", # 'wandb'
    #report_to=None, # 'wandb'
    #use_cpu=False,
)

#with tempfile.TemporaryDirectory() as tmp:
# Do not force FP16 weights here: AMP/Accelerate expects master grads handling.


extra_keep_keys = ULTRAFEEDBACK_EXTRA_KEYS if 'ltrafeedback' in script_args.train_set_path else []


def main_fun():
    torch_dtype = torch.bfloat16 if script_args.bf16 else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=1, dtype=torch_dtype).base_model
    #
    # send model to a gpu if available
    

    model.config.use_cache = not script_args.gradient_checkpointing
    if getattr(tokenizer, 'pad_token_id', None) is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    pad_token_id = model.config.pad_token_id

    dc = MORewardDataCollatorWithPadding(
                tokenizer=tokenizer, max_length=script_args.max_length, dtype=torch_dtype) # type: ignore

    
    
    dataset= PairwisePreferenceDataset(train_path, tokenizer, 
                                       from_disk=True, 
                                       extra_keep_keys=extra_keep_keys, 
                                       retokenize=script_args.retokenize,
                                       recalculate_embeddings=script_args.recalculate_embeddings,
                                       model_for_embeddings=model,
                                       collator=dc,
                                       cleanup_cache_files=bool(script_args.cleanup_dataset_cache_files),
                                       )
    print("Training set: ", len(dataset.train_dataset), " Eval set: ", len(dataset.eval_dataset), " Test set: ", len(dataset.test_dataset))
    #exit(0)
    original_columns = dataset.data.column_names
    

    mo_config = MORMForSequenceClassificationConfig(pad_token_id=pad_token_id, num_values=len(dataset.value_keys),
                                                    dtype=torch_dtype, 
                                                    lambda_decay=script_args.lambda_decay,
                                            hidden_sizes=[4096], value_layer_dropout=0.1,
                                            value_layer_intermediate_activation="SiLU", 
                                            value_layer_final_activation="none",
                                            grounding_loss_tendency_update_ratio=script_args.grounding_loss_tendency_update_ratio, 
                                            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
                                            metrics_accumulation_steps=script_args.metrics_accumulation_steps,
                            use_metrics_or_losses_for_lagrange_updates=script_args.use_metrics_or_losses_for_lagrange_updates,
                            grad_on_only_worst_value=script_args.grad_on_only_worst_value,
                            zero_constraint=script_args.zero_constraint,
                            use_ideal_grounding_model=script_args.use_ideal_grounding_model,
                            rew_center_coefficient=script_args.rew_center_coefficient,
                                            )

    mo_model = MORMForSequenceClassification(config=mo_config, base_model=model)
    sub_optimizer_cls, sub_optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args, mo_model)

    print("Sub optimizer class: ", sub_optimizer_cls
            , " Sub optimizer kwargs: ", sub_optimizer_kwargs)

    
    trainer : Trainer = MORewardTrainer(
            model=mo_model,
            args=training_args,
            train_dataset=dataset.train_dataset,
            
            eval_dataset=dataset.eval_dataset,
            compute_metrics=partial(MORewardTrainer.compute_metrics, training_variables=mo_model.training_variables),
            compute_loss_func = partial(mo_compute_loss_func, config=mo_config, training_variables=mo_model.training_variables, accelerator=trainer.accelerator if hasattr(trainer, 'accelerator') else None),
            optimizer_cls_and_kwargs =  (ConstrainedOptimizer, {
                'params_gr': list(mo_model.reward_heads.parameters()),
                'params_gr_ideal': list(mo_model.reward_heads_ideal.parameters()) if script_args.use_ideal_grounding_model else None,
                'params_vs': list(mo_model.value_system_layer.parameters()),
                'n_values': len(dataset.value_keys),
                'lr_value_system': script_args.learning_rate,
                'lr_grounding': script_args.grounding_learning_rate,
                'max_grad_norm': training_args.max_grad_norm,
                'inner_optimization_iterations': script_args.inner_optimization_iterations,
                'lr_lambda': script_args.lagrange_learning_rate,
                'initial_lambda': 1.0,
                'sub_optimizer_class': sub_optimizer_cls,
                'training_variables': mo_model.training_variables,
                ** sub_optimizer_kwargs
            }),
            data_collator=dc,
    )

    save_checkpoint_with_seed(
        trainer=trainer,
        tokenizer=tokenizer,
        checkpoint_dir=output_name + "/last_checkpoint",
        seed=int(script_args.seed),
    )

    trainer.train()
    

    print("Saving last checkpoint of the model")
    save_checkpoint_with_seed(
        trainer=trainer,
        tokenizer=tokenizer,
        checkpoint_dir=output_name + "/last_checkpoint",
        seed=int(script_args.seed),
    )

if __name__ == "__main__":
        
    main_fun()
