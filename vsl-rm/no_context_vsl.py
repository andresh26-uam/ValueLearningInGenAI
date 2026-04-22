#!/usr/bin/env python3
#SBATCH --job-name=ValueLearningInGenAI
#SBATCH --chdir=/home/aholg/ValueLearningInGenAI

from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
import random
import sys
from pprint import pprint


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


from typing import Any, Optional

from transformers import Trainer
# import evaluate
import numpy as np
import torch
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


from vsllib.defines import REWARD_HEADS_INDICES, REWARD_HEADS_OUTPUT, VALUE_SYSTEM_OUTPUT, SupportedDatasets, EXTRA_KEYS, TRAIN_PATHS, get_test_indices, get_validation_indices
from vsllib.reward_models import VALUE_LAYER_ACTIVATIONS, MOLossFunctions, MORMForSequenceClassification, MORMForSequenceClassificationConfig, mo_compute_loss_func
from vsllib.training import ConstrainedOptimizer, MORewardTrainer
from vsllib.utils import MORewardDataCollatorWithPadding, save_checkpoint_with_seed
from vsllib.dataset_processing import PairwisePreferenceDataset

# Define and parse arguments.

# IMport HF_TOKEN from .env
from dotenv import load_dotenv
load_dotenv()


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

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    use_frozen_base_model: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a frozen base model with built-in reward structure (e.g. ArmoRM). If False, we will use the base model as a starting point and train the reward heads from scratch."})
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    retokenize: Optional[bool] = field(
		default=False, metadata={"help": "Whether to retokenize the dataset. Set this to False if you have already tokenized and saved the dataset to disk, and just want to load it."})
    recalculate_embeddings: Optional[bool] = field(
        default=False, metadata={"help": "Whether to recalculate embeddings for the dataset."})
    use_embeddings: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use embeddings for the dataset."})
    use_cpu: Optional[bool] = field(
        default=USE_CPU, metadata={"help": "Whether to use CPU for training. If False, will use GPU if available."})
    
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

    hidden_size: Optional[int] = field(default=1024, metadata={"help": "The hidden size of the grounding MLP."})
    num_hidden_layers: Optional[int] = field(default=0, metadata={"help": "The number of hidden layers in the grounding MLP. If 0, there will be no hidden layers and the value head will be a simple linear layer from the prompt-respose final embedding into the number of values."})
    value_layer_dropout: Optional[float] = field(default=0.0)
    layer_activation: Optional[str] = field(default="ReLU", metadata={"help": f"The activation function to use for the hidden layers. Use one of {list(VALUE_LAYER_ACTIVATIONS.keys())}"}) # TODO "ReLU" see 
    final_layer_activation: Optional[str] = field(default="none", metadata={"help": f"The activation function to use for the final layer. Use one of {list(VALUE_LAYER_ACTIVATIONS.keys())}"}) # TODO "ReLU" or "GELU" or "none"


    per_device_train_batch_size: Optional[int] = field(default=128)
    per_device_eval_batch_size: Optional[int] = field(default=128)
    gradient_accumulation_steps: Optional[int] = field(default=5) # TODO 32?
    metrics_accumulation_steps: Optional[int] = field(default=5) # TODO 32?
    lambda_decay: Optional[float] = field(default=0.0005)
    rew_center_coefficient: Optional[float] = field(default=0.01) # TODO Recommended by TRL library (RewardTrainer): 0.01
    layer_normalization: Optional[str] = field(default="none") # TODO "BatchNorm" or "LayerNorm" or "none". 
    max_grad_norm: Optional[float] = field(default=0.01) # TODO 0.01?

    learning_rate: Optional[float] = field(default=0.001)
    grounding_learning_rate: Optional[float] = field(default=0.001) # TODO must be > 1e-4 to make any effect??
    lagrange_learning_rate: Optional[float] = field(default=0.1) # TODO 0.01

    grounding_loss_tendency_update_ratio: Optional[float] = field(default=0.01)
    use_exponential_moving_average_or_optimum_targets: Optional[str] = field(default="optimum")
    use_metrics_or_losses_for_lagrange_updates: Optional[str] = field(default="metrics")
    grad_on_only_worst_value: Optional[bool] = field(default=False)
    zero_constraint: Optional[bool] = field(default=True)
    use_ideal_grounding_model : Optional[bool] = field(default=False)

    loss_func_type: Optional[str] = field(
        default=MOLossFunctions.DEFAULT.value,
        metadata={"help": "The name of the run for logging purposes."},
    )
    loss_func_type_kwargs: Optional[str] = field(
        default=None,#json.dumps({'value_indices': [2]}),
        metadata={"help": "A json string of the kwargs to use for the loss function. E.g. for ONLY_VALUES_IN_KWARGS, you can specify which value indexes to use for the grounding loss."},
    )

    
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        #default="mistralai/Mistral-7B-Instruct-v0.2",
        #default="meta-llama/Llama-3.2-1B",
        default="HuggingFaceTB/SmolLM-135M-Instruct",
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
        default=100,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    dataset: Optional[SupportedDatasets] = field(
        default=SupportedDatasets.PKUALIGNMENT.value, 
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    
    output_path: Optional[str] = field(
        default=f"./models/no_context_vsl-",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit", # TODO. adamw_torch_fused is much faster on CPU. PagedAdamW_32bit is slower on CPU but works on GPU.
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="constant", # TODO "cosine" or "linear" or "constant"
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    run_name: Optional[str] = field(
        default_factory=lambda: f"run_",
        metadata={"help": "The name of the run for logging purposes."},
    )

    save_every_steps: Optional[int] = field(
        default=50000,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        #default=999999,
        default=100,
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
    config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a JSON file containing ScriptArguments values."},
    )
        
def argument_parser(parser):
    script_args = parser.parse_args_into_dataclasses()[0]
    if script_args.config_file:
        config_path = os.path.abspath(script_args.config_file)
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if not isinstance(config_data, dict):
            raise ValueError(f"Expected JSON object in config file, got {type(config_data).__name__}")

        valid_fields = set(ScriptArguments.__dataclass_fields__.keys())
        unknown_keys = set(config_data.keys()) - valid_fields
        if unknown_keys:
            raise ValueError(f"Unknown keys in config file: {sorted(unknown_keys)}")

    # Start from parsed CLI/default values.
        merged_args = vars(script_args).copy()
    # Apply JSON values.
        merged_args.update(config_data)

    # Re-apply explicit CLI overrides so precedence is: CLI > JSON > defaults.
        cli_override_keys = set()
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                continue
            key = arg[2:].split("=", 1)[0]
            if key != "config_file" and key in valid_fields:
                cli_override_keys.add(key)
        for key in cli_override_keys:
            merged_args[key] = getattr(script_args, key)

        script_args = ScriptArguments(**merged_args)
    
    # Parsing enum values
    script_args.dataset = SupportedDatasets(script_args.dataset)
    script_args.loss_func_type = MOLossFunctions(script_args.loss_func_type)
    if script_args.use_cpu:
            script_args.bf16 = False  # bf16 is not supported on CPU, so we disable it if use_cpu is True.
            script_args.deepspeed = None  # Deepspeed is not compatible with CPU training, so we disable it if use_cpu is True.
            script_args.optim = "adamw_torch_fused"  # Use a more CPU-friendly optimizer if use_cpu is True.
            
    return script_args


parser = HfArgumentParser(ScriptArguments) # type: ignore

script_args = argument_parser(parser)

seed_everything(int(script_args.seed))

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length

# Get the dataset

train_path = TRAIN_PATHS[script_args.dataset]
extra_keep_keys = EXTRA_KEYS[script_args.dataset]
test_proportion_or_indices = get_test_indices(script_args.dataset)
eval_proportion_or_indices = get_validation_indices(script_args.dataset)

output_name = script_args.output_path + script_args.model_name.split("/")[-1]

run_name = f"{script_args.dataset.value}_" + script_args.run_name + f"_{datetime.now().strftime('%m%d_%H%M%S')}_epo{script_args.num_train_epochs}_s{script_args.seed}" if script_args.run_name is not None else None

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
    max_grad_norm=script_args.max_length, # TODO 0.01?
    run_name = run_name ,
    #report_to=None, # 'wandb'
    use_cpu=script_args.use_cpu,
)

#with tempfile.TemporaryDirectory() as tmp:
# Do not force FP16 weights here: AMP/Accelerate expects master grads handling.

def main_fun() -> None:
    torch_dtype = torch.bfloat16 if script_args.bf16 else torch.float32

    model_kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True)
    
    if script_args.use_frozen_base_model:
        # For models like ArmoRM that have built-in reward structure
        
        model = AutoModelForSequenceClassification.from_pretrained(
                script_args.model_name, **model_kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name, num_labels=1, **model_kwargs).base_model
    
    # For non-frozen models, get base model; for frozen models, use as-is
    if not script_args.use_frozen_base_model and hasattr(model, 'base_model'):
        model = model.base_model
        
    #
    # send model to a gpu if available
    
    model.config.use_cache = not script_args.gradient_checkpointing
    if getattr(tokenizer, 'pad_token_id', None) is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    pad_token_id = model.config.pad_token_id

    dc = MORewardDataCollatorWithPadding(
                tokenizer=tokenizer, max_length=script_args.max_length, dtype=torch_dtype, use_embeddings=script_args.use_embeddings) # type: ignore

    
    
    dataset= PairwisePreferenceDataset(train_path, tokenizer, 
                                       from_disk=True, 
                                       extra_keep_keys=extra_keep_keys, 
                                       retokenize=script_args.retokenize,
                                       recalculate_embeddings=script_args.recalculate_embeddings,
                                       use_embeddings=script_args.use_embeddings,
                                       model_for_embeddings=model,
                                       collator=dc,
                                       split_seed=int(42),
                                       eval_proportion_or_indices=eval_proportion_or_indices, 
                                       test_proportion_or_indices=test_proportion_or_indices,
                                       cleanup_cache_files=bool(script_args.cleanup_dataset_cache_files),
                                       )
    print("Training set: ", len(dataset.train_dataset), " Eval set: ", len(dataset.eval_dataset), " Test set: ", len(dataset.test_dataset))
    #exit(0)

    reward_heads_module_name = REWARD_HEADS_OUTPUT.get(script_args.model_name, None)
    value_system_module_name = VALUE_SYSTEM_OUTPUT.get(script_args.model_name, None)
    reward_head_indices = REWARD_HEADS_INDICES.get(script_args.model_name, {}).get(script_args.dataset, None)
    num_values_to_use = len(dataset.value_keys)
    
    
    if script_args.use_frozen_base_model:
        # Parse reward head indices if provided
        if reward_head_indices is not None:
            print(f"Using reward head indices: {reward_head_indices}")
            # Update num_values based on selected indices
            assert num_values_to_use == len(reward_head_indices), f"Number of values to use ({num_values_to_use}) does not match the length of reward head indices ({len(reward_head_indices)})"
            #num_values_to_use = len(reward_head_indices)
    


    mo_config = MORMForSequenceClassificationConfig(pad_token_id=pad_token_id, num_values=len(dataset.value_keys),
                                                    dtype=torch_dtype,
                                                    loss_func_type=script_args.loss_func_type,
                                                    loss_func_kwargs=json.loads(script_args.loss_func_type_kwargs) if script_args.loss_func_type_kwargs is not None else {},
                                                    lambda_decay=script_args.lambda_decay,
                                            hidden_sizes=[script_args.hidden_size]*script_args.num_hidden_layers, value_layer_dropout=script_args.value_layer_dropout,
                                            value_layer_intermediate_activation=script_args.layer_activation, 
                                            value_layer_final_activation=script_args.final_layer_activation,
                                            layer_normalization=script_args.layer_normalization,
                                            grounding_loss_tendency_update_ratio=script_args.grounding_loss_tendency_update_ratio, 
                                            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
                                            metrics_accumulation_steps=script_args.metrics_accumulation_steps,
                            use_metrics_or_losses_for_lagrange_updates=script_args.use_metrics_or_losses_for_lagrange_updates,
                            use_exponential_moving_average_or_optimum_targets=script_args.use_exponential_moving_average_or_optimum_targets,
                            grad_on_only_worst_value=script_args.grad_on_only_worst_value,
                            zero_constraint=script_args.zero_constraint,
                            use_ideal_grounding_model=script_args.use_ideal_grounding_model,
                            rew_center_coefficient=script_args.rew_center_coefficient,
                            base_model_reward_heads_module_name=reward_heads_module_name if script_args.use_frozen_base_model else None,
                            base_model_value_system_module_name=value_system_module_name if script_args.use_frozen_base_model else None,
                            base_model_reward_head_indices=reward_head_indices,
                            use_base_model_heads=script_args.use_frozen_base_model
                            )

    
    mo_model = MORMForSequenceClassification(config=mo_config, base_model=model)
    
    sub_optimizer_cls, sub_optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args, mo_model)

    print("Sub optimizer class: ", sub_optimizer_cls
            , " Sub optimizer kwargs: ", sub_optimizer_kwargs)

    print("Script arguments: ")
    
    pprint(vars(script_args))
    #exit(0)
    trainer : Trainer = MORewardTrainer(
            model=mo_model,
            args=training_args,
            train_dataset=dataset.train_dataset,
            eval_dataset=dataset.eval_dataset,
            compute_metrics=partial(MORewardTrainer.compute_metrics, config=mo_config, training_variables=mo_model.training_variables),
            compute_loss_func = partial(mo_compute_loss_func, config=mo_config, training_variables=mo_model.training_variables),
            optimizer_cls_and_kwargs =  (ConstrainedOptimizer, {
                'params_gr': list(mo_model.grounding_parameters()),
                'params_gr_ideal': list(mo_model.reward_heads_ideal.parameters()) if script_args.use_ideal_grounding_model else None,
                'params_vs': list(mo_model.value_system_parameters()),
                'n_values': len(dataset.value_keys),
                'lr_value_system': script_args.learning_rate,
                'lr_grounding': script_args.grounding_learning_rate,
                'max_grad_norm': training_args.max_grad_norm,
                'inner_optimization_iterations': script_args.inner_optimization_iterations,
                'lr_lambda': script_args.lagrange_learning_rate,
                'initial_lambda': 1.0,
                'config': mo_config,
                'sub_optimizer_class': sub_optimizer_cls,
                'training_variables': mo_model.training_variables,
                ** sub_optimizer_kwargs
            }),
            data_collator=dc,
    )

    trainer.train()
    

    print("Saving last checkpoint of the model")
    save_checkpoint_with_seed(
        trainer=trainer,
        tokenizer=tokenizer,
        checkpoint_dir=output_name + "/" + script_args.run_name + "/last_checkpoint",
        seed=int(script_args.seed),
    )

if __name__ == "__main__":
        
    main_fun()
