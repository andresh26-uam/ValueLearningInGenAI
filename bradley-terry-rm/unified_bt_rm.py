########################
# Unified Bradley-Terry reward model training script.
# This combines Gemma, Llama3, and Mistral variants into a single entrypoint.
########################
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy


@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If unset, model-specific default is used."},
    )
    learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "If unset, model-specific default is used."},
    )
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={"help": "Model name from Hugging Face hub."},
    )
    model_variant: Optional[str] = field(
        default="auto",
        metadata={
            "help": "One of: auto, gemma, llama3, mistral. auto infers from model_name."
        },
    )
    bf16: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=1)
    train_set_path: Optional[str] = field(default="hendrydong/preference_700K")
    eval_set_path: Optional[str] = field(default="hendrydong/preference_700K")
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "If unset, model-specific default is used."},
    )
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    max_length: Optional[int] = field(default=4096)
    save_every_steps: Optional[int] = field(default=999999)
    eval_every_steps: Optional[int] = field(default=999999)


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "gemma": {
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 32,
        "output_path": "./bt_models/gemma2b_rm",
        "tokenizer_use_fast": None,
        "tokenizer_add_pad_token": False,
        "use_flash_attention_2": False,
    },
    "llama3": {
        "learning_rate": 2e-6,
        "gradient_accumulation_steps": 64,
        "output_path": "./models/llama3_rm",
        "tokenizer_use_fast": False,
        "tokenizer_add_pad_token": True,
        "use_flash_attention_2": True,
    },
    "mistral": {
        "learning_rate": 5e-6,
        "gradient_accumulation_steps": 64,
        "output_path": "./bt_models/mistral_rm",
        "tokenizer_use_fast": False,
        "tokenizer_add_pad_token": True,
        "use_flash_attention_2": False,
    },
}


def infer_variant(model_name: str, requested_variant: str) -> str:
    if requested_variant and requested_variant != "auto":
        variant = requested_variant.lower()
        if variant not in MODEL_PRESETS:
            raise ValueError(
                f"Unsupported model_variant={requested_variant}. "
                f"Expected one of: auto, gemma, llama3, mistral"
            )
        return variant

    lower_name = model_name.lower()
    if "gemma" in lower_name:
        return "gemma"
    if "llama" in lower_name:
        return "llama3"
    if "mistral" in lower_name:
        return "mistral"

    raise ValueError(
        "Could not infer model variant from model_name. "
        "Please set --model_variant to one of: gemma, llama3, mistral"
    )


def resolve_script_args(script_args: ScriptArguments, preset: Dict[str, Any]) -> None:
    if script_args.learning_rate is None:
        script_args.learning_rate = preset["learning_rate"]
    if script_args.gradient_accumulation_steps is None:
        script_args.gradient_accumulation_steps = preset["gradient_accumulation_steps"]
    if script_args.output_path is None:
        script_args.output_path = preset["output_path"]


def maybe_strip_bos(text: str, bos_token: Optional[str]) -> str:
    if bos_token:
        return text.replace(bos_token, "")
    return text


def build_dataset(tokenizer, train_path: str, eval_path: str):
    def tokenize(sample):
        sample["positive"] = maybe_strip_bos(
            tokenizer.apply_chat_template(
                sample["chosen"], tokenize=False, add_generation_prompt=False
            ),
            tokenizer.bos_token,
        )
        sample["negative"] = maybe_strip_bos(
            tokenizer.apply_chat_template(
                sample["rejected"], tokenize=False, add_generation_prompt=False
            ),
            tokenizer.bos_token,
        )

        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample

    train_dataset = load_dataset(train_path, split="train").shuffle(seed=42)
    train_dataset = train_dataset.map(tokenize, num_proc=8)

    eval_dataset = load_dataset(eval_path, split="train").shuffle(seed=42).select(range(500))
    return train_dataset, eval_dataset


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )

        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }


def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    result["accuracy"] = np.sum(pos_predictions_scores > neg_predictions_scores) / len(
        pos_predictions_scores
    )
    return result


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = None):
        del num_items_in_batch
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    variant = infer_variant(script_args.model_name, script_args.model_variant or "auto")
    preset = MODEL_PRESETS[variant]
    resolve_script_args(script_args, preset)

    tokenizer_kwargs: Dict[str, Any] = {}
    if preset["tokenizer_use_fast"] is not None:
        tokenizer_kwargs["use_fast"] = preset["tokenizer_use_fast"]
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, **tokenizer_kwargs)

    if preset["tokenizer_add_pad_token"] and tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = script_args.max_length

    train_dataset, eval_dataset = build_dataset(
        tokenizer,
        script_args.train_set_path,
        script_args.eval_set_path,
    )
    print("Model variant:", variant)
    print("Training set:", len(train_dataset), "Eval set:", len(eval_dataset))

    training_args = TrainingArguments(
        output_dir=script_args.output_path,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
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
        report_to="wandb",
    )

    model_kwargs: Dict[str, Any] = {
        "num_labels": 1,
        "torch_dtype": torch.bfloat16,
    }
    if preset["use_flash_attention_2"]:
        model_kwargs["use_flash_attention_2"] = True

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        **model_kwargs,
    )

    model.config.use_cache = not script_args.gradient_checkpointing
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if preset["tokenizer_add_pad_token"]:
        model.resize_token_embeddings(len(tokenizer))

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=script_args.max_length
        ),
    )

    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.save_model(script_args.output_path + "/last_checkpoint")
    tokenizer.save_pretrained(script_args.output_path + "/last_checkpoint")


if __name__ == "__main__":
    main()
