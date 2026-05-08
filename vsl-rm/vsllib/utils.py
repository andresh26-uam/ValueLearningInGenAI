from datetime import datetime
from dataclasses import dataclass, field
import json
import sys
from typing import Any

import numpy as np
import torch as th

from typing import Any, Optional, Dict, Tuple
import os
import random

from transformers import (
    set_seed, AutoTokenizer
)
from triton.language import assume

from vsllib.defines import MODEL_DIR, MODEL_PRESETS, infer_variant, MOLossFunctions, SupportedDatasets, VALUE_LAYER_ACTIVATIONS


def seed_everything(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    set_seed(seed)

    if deterministic:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        try:
            th.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def to_float(value: Any) -> float:
    if isinstance(value, th.Tensor):
        if value.numel() == 1:
            return float(value.detach().item())
        return float(value.detach().mean().item())
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return float(value.mean())
    return float(value)


def fuse_parameters(model) -> th.Tensor:
    """Move model parameters to a contiguous tensor, and return that tensor."""
    n = sum(p.numel() for p in model.parameters())
    params = th.zeros(n, requires_grad=True, device=next(
        model.parameters()).device, dtype=next(model.parameters()).dtype)
    params.grad = th.zeros(n, device=params.device, dtype=params.dtype)
    i = 0
    for p in model.parameters():
        params_slice = params[i:i + p.numel()]
        with th.no_grad():
            params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        p.grad = params.grad[i:i + p.numel()].view(p.shape)
        i += p.numel()
    return params



def print_tensor_and_grad_fn(grad_fn, level=0):
    indent = "  " * level
    if grad_fn is None:
        print("NO GRAD FN")
        return
    if getattr(grad_fn, 'variable', None) is not None:
        if grad_fn.variable.requires_grad:
            print(f"{indent}AccumulateGrad for tensor: {grad_fn.variable.shape}")
    else:
        print(f"{indent}Grad function: {grad_fn}")
        if hasattr(grad_fn, 'next_functions'):
            for next_fn in grad_fn.next_functions:
                if next_fn[0] is not None:
                    print_tensor_and_grad_fn(next_fn[0], level + 1)

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
        default=False, metadata={"help": "Whether to use CPU for training. If False, will use GPU if available."})

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

    do_train: Optional[bool] = field(default=True)

    hidden_size: Optional[int] = field(
        default=1024, metadata={"help": "The hidden size of the grounding MLP."})
    num_hidden_layers: Optional[int] = field(default=0, metadata={
                                             "help": "The number of hidden layers in the grounding MLP. If 0, there will be no hidden layers and the value head will be a simple linear layer from the prompt-respose final embedding into the number of values."})
    value_layer_dropout: Optional[float] = field(default=0.0)
    layer_activation: Optional[str] = field(default="ReLU", metadata={
                                            "help": f"The activation function to use for the hidden layers. Use one of {list(VALUE_LAYER_ACTIVATIONS.keys())}"})
    final_layer_activation: Optional[str] = field(default="none", metadata={
                                                  "help": f"The activation function to use for the final layer. Use one of {list(VALUE_LAYER_ACTIVATIONS.keys())}"})

    per_device_train_batch_size: Optional[int] = field(default=128)
    per_device_eval_batch_size: Optional[int] = field(default=128)
    gradient_accumulation_steps: Optional[int] = field(default=5)  
    
    lambda_decay: Optional[float] = field(default=0.0005)
    rew_center_coefficient: Optional[float] = field(default=0.01)
    layer_normalization: Optional[str] = field(default="LayerNorm")
    max_grad_norm: Optional[float] = field(default=0.01)  

    learning_rate: Optional[float] = field(default=0.0001)
    grounding_learning_rate: Optional[float] = field(
        default=0.0001) 
    lagrange_learning_rate: Optional[float] = field(default=0.1)  
    grounding_loss_tendency_update_ratio: Optional[float] = field(default=0.01)
    use_exponential_moving_average_or_optimum_targets: Optional[str] = field(
        default="average")
    use_metrics_or_losses_for_lagrange_updates: Optional[str] = field(
        default="losses")
    assume_qualitative_labels: Optional[bool] = field(default=True)

    gather_train_metrics: Optional[bool] = field(default=True)
    grad_on_only_worst_value: Optional[bool] = field(default=False)
    zero_constraint: Optional[bool] = field(default=True)
    use_ideal_grounding_model: Optional[bool] = field(default=False)

    loss_func_type: Optional[str] = field(
        default=MOLossFunctions.DEFAULT.value,
        metadata={"help": "The name of the run for logging purposes."},
    )
    loss_func_type_kwargs: Optional[str] = field(
        default=None,  # json.dumps({'value_indices': [2]}),
        metadata={"help": "A json string of the kwargs to use for the loss function. E.g. for ONLY_VALUES_IN_KWARGS, you can specify which value indexes to use for the grounding loss."},
    )

    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        # default="mistralai/Mistral-7B-Instruct-v0.2",
        # default="meta-llama/Llama-3.2-1B",
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    activate_discordance_epsilon_for_loss : Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to activate the discordance epsilon for the loss function. If True, will use the discordance epsilon to calculate a target probability for each pair, and will ignore pairs that are within the epsilon of each other. This can help with training stability if there is a lot of noise in the labels."
        },
    )
    model_variant: Optional[str] = field(
        default="auto",
        metadata={
            "help": "The model variant, which determines some default settings for the tokenizer and model. Set this to 'auto' to infer from the model name, or set it explicitly to one of: gemma, llama3, mistral (see infer model variant)"
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
    dataset: Optional[SupportedDatasets] = field(
        default=SupportedDatasets.PKUALIGNMENT.value,
        metadata={"help": "The dir of the subset of the training data to use"},
    )

    output_path: Optional[str] = field(
        default=os.path.join(MODEL_DIR, "no_context_vsl-"),
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        # TODO. adamw_torch_fused is much faster on CPU. PagedAdamW_32bit is slower on CPU but works on GPU.
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="constant",  # TODO "cosine" or "linear" or "constant"
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)
    update_tendencies_every_n_steps: Optional[int] = field(
        default=1,
        metadata={"help": "How often to update the loss/metric tendencies for the Lagrange multiplier updates."},
    )
    use_validation_for_tendencies: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the validation set for calculating the loss/metric tendencies for the Lagrange multiplier updates. If False, will use the training set."},
    )
    run_name: Optional[str] = field(
        default_factory=lambda: f"run_",
        metadata={"help": "The name of the run for logging purposes."},
    )
    discordance_epsilon: Optional[float] = field(
        default=None,
        metadata={"help": "The epsilon to use for calculating discordance-aware representativeness. If None, will be set to half of the minimum nonzero difference between any pair of labels in the training dataset."},
    )
    save_every_steps: Optional[int] = field(
        default=10000,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        # default=999999,
        default=100,
        metadata={"help": "Eval the model every x steps"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "Global seed for Python, NumPy, PyTorch, and Transformers."},
    )

    do_save: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to save the model after training."},
    )
    do_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to save checkpoints during training."},
    )
    config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a JSON file containing ScriptArguments values."},
    )


def argument_parser(script_args: ScriptArguments) -> Tuple[ScriptArguments, Dict[str, Any]]:

    if script_args.config_file:
        config_path = os.path.abspath(script_args.config_file)
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if not isinstance(config_data, dict):
            raise ValueError(
                f"Expected JSON object in config file, got {type(config_data).__name__}")

        valid_fields = set(ScriptArguments.__dataclass_fields__.keys())
        unknown_keys = set(config_data.keys()) - valid_fields
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config file: {sorted(unknown_keys)}")

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
        # bf16 is not supported on CPU, so we disable it if use_cpu is True.
        script_args.bf16 = False
        # Deepspeed is not compatible with CPU training, so we disable it if use_cpu is True.
        script_args.deepspeed = None
        # Use a more CPU-friendly optimizer if use_cpu is True.
        script_args.optim = "adamw_torch_fused"
    script_args.output_path = script_args.output_path + \
        script_args.model_name.split("/")[-1]

    script_args.run_name = f"{script_args.dataset.value}_" + script_args.run_name + \
        f"_{datetime.now().strftime('%m%d_%H%M%S')}_epo{script_args.num_train_epochs}_s{script_args.seed}" if script_args.run_name is not None else None
    variant = infer_variant(script_args.model_name,
                            script_args.model_variant or "auto")
    
    if script_args.loss_func_type_kwargs is not None:
        if isinstance(script_args.loss_func_type_kwargs, str):  
            script_args.loss_func_type_kwargs = json.loads(
            script_args.loss_func_type_kwargs)
        else:
            script_args.loss_func_type_kwargs = script_args.loss_func_type_kwargs
    else:
        script_args.loss_func_type_kwargs = {}

    script_args.update_tendencies_every_n_steps = script_args.eval_every_steps if script_args.use_validation_for_tendencies else script_args.update_tendencies_every_n_steps
    preset = MODEL_PRESETS[variant]
    return script_args, preset


def obtain_tokenizer(script_args: ScriptArguments, preset: Dict[str, Any], checkpoint_path=None) -> AutoTokenizer:
    tokenizer_kwargs = {}
    if preset["tokenizer_use_fast"] is not None:
        tokenizer_kwargs["use_fast"] = preset["tokenizer_use_fast"]
        tokenizer_kwargs["use_auth_token"] = preset["tokenizer_use_auth_token"]
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name if checkpoint_path is None else checkpoint_path, **tokenizer_kwargs)

    if preset["tokenizer_add_pad_token"] and tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = script_args.max_length
    return tokenizer


def maybe_assign_pad_token(mod, script_args: ScriptArguments, tokenizer: AutoTokenizer, preset: Dict[str, any]) -> None:
    mod.config.use_cache = not script_args.gradient_checkpointing
    if mod.config.pad_token_id is None or mod.config.pad_token_id != tokenizer.pad_token_id:
        mod.config.pad_token_id = tokenizer.pad_token_id
    assert mod.config.pad_token_id is not None, "Tokenizer does not have a pad token, which is required for this script. To add a padtoken, see defines.py."
    if preset["tokenizer_add_pad_token"]:
        mod.resize_token_embeddings(len(tokenizer))
