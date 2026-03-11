from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers.utils import PaddingStrategy


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
)

import pdb

from vsllib.defines import ULTRAFEEDBACK_EXTRA_KEYS, ULTRAFEEDBACK_PROCESSED_PATH
from vsllib.reward_models import MultiObjectiveRewardModel
from vsllib.training import MORewardTrainer, PairwisePreferenceDataset
from vsllib.utils import MORewardDataCollatorWithPadding

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    retokenize: Optional[bool] = field(
		default=False, metadata={"help": "Whether to retokenize the dataset. Set this to False if you have already tokenized and saved the dataset to disk, and just want to load it."})
    deepspeed: Optional[str] = field(
        # default="dp3.json",
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
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
        default=f"./models/baselines/no_context_vsl_from-{model_name}",
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
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )

parser = HfArgumentParser(ScriptArguments) # type: ignore
script_args = parser.parse_args_into_dataclasses()[0]

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length

# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.train_set_path # splits are done later.
output_name = script_args.output_path

training_args = TrainingArguments(
    output_dir=output_name,
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
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to='wandb',
    use_cpu=True,
)


model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16,
)

model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

num_proc = 24  # Can adjust to be higher if you have more processors.


extra_keep_keys = ULTRAFEEDBACK_EXTRA_KEYS if 'ltrafeedback' in script_args.train_set_path else []
dataset= PairwisePreferenceDataset(train_path, tokenizer, from_disk=True, extra_keep_keys=extra_keep_keys, retokenize=script_args.retokenize)
print("Training set: ", len(dataset.train_dataset), " Eval set: ", len(dataset.eval_dataset))
original_columns = dataset.data.column_names

print(original_columns)

print(model, vars(model))
input("??")
mo_model = MultiObjectiveRewardModel(model, num_values=len(dataset.value_keys))
trainer = MORewardTrainer(
    model=mo_model,
    args=training_args,
    train_dataset=dataset.train_dataset,
    eval_dataset=dataset.eval_dataset,
    compute_metrics=MORewardTrainer.compute_metrics,
    data_collator=MORewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length),
)


trainer.train()


print("Saving last checkpoint of the model")
#model.save_pretrained(output_name + "/last_checkpoint")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")

