import csv
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import sys
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import KMeans
import torch as th

from pathlib import Path
from typing import Any, Optional, Dict, Tuple
import os
import random

import math


import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt



from transformers import (
    set_seed, AutoTokenizer
)
from triton.language import assume

from vsllib.defines import MODEL_DIR, MODEL_PRESETS, NO_RATING_MASK, ContextImplementations, infer_variant, MOLossFunctions, SupportedDatasets, VALUE_LAYER_ACTIVATIONS


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



def to_tensor(value: Any, dtype=None, device=None) -> float:
    if isinstance(value, th.Tensor):
        return value
    return th.tensor(value, dtype=dtype, device=device)

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



def print_tensor_and_grad_fn(grad_fn, level: int =0) -> None:
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

from transformers.utils import TensorType, is_numpy_array
from transformers.utils.import_utils import is_torch_available, is_mlx_available
from transformers.tokenization_utils_base import flatten

def convert_to_tensors(data: Dict, tensor_type: str | TensorType | None = None, prepend_batch_axis: bool = False):
        """
        Convert the inner content to tensors (TAKEN FROM TRANSFORMERS LIBRARY)

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return data

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        if tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            def as_tensor(value, dtype=None) -> th.Tensor:
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    return torch.from_numpy(np.array(value))
                if len(flatten(value)) == 0 and dtype is None:
                    dtype = torch.int64
                return torch.tensor(value, dtype=dtype)

            is_tensor = torch.is_tensor

            """ elif tensor_type == TensorType.MLX:
            if not is_mlx_available():
                raise ImportError("Unable to convert output to MLX tensors format, MLX is not installed.")
            import mlx.core as mx

            def as_tensor(value: Any, dtype=None):
                if len(flatten(value)) == 0 and dtype is None:
                    dtype = mx.int32
                return mx.array(value, dtype=dtype)

            def is_tensor(obj):
                return isinstance(obj, mx.array)"""
        else:

            def as_tensor(value: Any, dtype=None):
                if (
                    isinstance(value, (list, tuple))
                    and len(value) > 0
                    and isinstance(value[0], (list, tuple, np.ndarray))
                ):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # we have a ragged list so handle explicitly
                        value = as_tensor([np.asarray(val) for val in value], dtype=object)
                if len(flatten(value)) == 0 and dtype is None:
                    dtype = np.int64
                return np.asarray(value, dtype=dtype)

            is_tensor = is_numpy_array

        # Do the tensor conversion in batch
        for key, value in data.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
                    # # at-least2d
                    # if tensor.ndim > 2:
                    #     tensor = tensor.squeeze(0)
                    # elif tensor.ndim < 2:
                    #     tensor = tensor[None, :]

                    data[key] = tensor
            except Exception as e:
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    ) from e
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding with"
                    " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                    f" features (`{key}` in this case) have excessive nesting (inputs type `list` where type `int` is"
                    " expected)."
                ) from e

        return data



import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist



def sample_example_profiles_scipy(profile_variety, n_values=3,
                            repulsion_power=2,
                            n_restarts=50,
                            seed=0):

    if n_values < 1:
        raise ValueError("n_values must be >= 1")

    if n_values == 1:
        return [(1.0,)] * profile_variety

    rng = np.random.default_rng(seed)

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z.reshape(profile_variety, n_values)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def objective(z):
        P = softmax(z)

        D = pdist(P)

        # avoid division by zero
        D = np.maximum(D, 1e-12)

        return np.sum(D ** (-repulsion_power))

    best_x = None
    best_energy = np.inf

    for _ in range(n_restarts):

        # Dirichlet initialization
        P0 = rng.dirichlet(np.ones(n_values),
                           size=profile_variety)

        Z0 = np.log(P0)
        Z0 = Z0.ravel()

        result = minimize(
            objective,
            Z0,
            method="L-BFGS-B",
            options={
                "maxiter": 10000,
                "ftol": 1e-12,
            },
        )

        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x

    P = softmax(best_x)

    # deterministic ordering
    P = P[np.lexsort(P.T[::-1])]

    ret = [
        tuple(round(float(v), 4) for v in row)
        for row in P
    ]
    return ret


def sample_example_value_systems_exact(profile_variety: int, n_values: int = 3):
    """
    Return exactly `profile_variety` profiles distributed as evenly
    as possible on the (n_values-1)-simplex.

    Each profile sums to 1.
    """

    if n_values < 1:
        raise ValueError("n_values must be >= 1")

    if n_values == 1:
        return [(1.0,)] * profile_variety

    # ------------------------------------------------------------------
    # Generate simplex lattice (Das-Dennis reference directions)
    # ------------------------------------------------------------------

    def lattice_count(H):
        return math.comb(H + n_values - 1, n_values - 1)

    H = 1
    while lattice_count(H) < profile_variety:
        H += 1

    def compositions(total, parts):
        if parts == 1:
            yield (total,)
            return

        for i in range(total + 1):
            for rest in compositions(total - i, parts - 1):
                yield (i,) + rest

    lattice = np.array(
        [np.array(c, dtype=float) / H
         for c in compositions(H, n_values)],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # If needed, downsample using farthest-point sampling
    # ------------------------------------------------------------------

    if len(lattice) > profile_variety:

        selected = []

        # Start with simplex vertices when possible
        vertices = np.eye(n_values)

        for v in vertices:
            idx = np.argmin(np.linalg.norm(lattice - v, axis=1))
            if idx not in selected:
                selected.append(idx)
            if len(selected) == profile_variety:
                break

        while len(selected) < profile_variety:
            chosen = lattice[selected]

            dists = np.min(
                np.linalg.norm(
                    lattice[:, None, :] - chosen[None, :, :],
                    axis=2,
                ),
                axis=1,
            )

            dists[selected] = -1
            selected.append(np.argmax(dists))

        lattice = lattice[selected]

    profiles = [
        tuple(round(float(x), 3) for x in row)
        for row in lattice
    ]
    profiles.sort()
    
    return profiles
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    task_type: Optional[str] = field(
        default="nlp_based", metadata={"help": "Options: nlp_based, feature_based"}
    )
    report_to: Optional[str] = field(
        default="wandb", metadata={"help": "Options: wandb, none"}
    )
    use_frozen_base_model: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a frozen base model with built-in reward structure (e.g. ArmoRM). If False, we will use the base model as a starting point and train the reward heads from scratch."})
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    repostprocess: Optional[bool] = field(
        default=False, metadata={"help": "Whether to retokenize/postprocess the dataset. Set this to False if you have already tokenized and saved the dataset to disk, and just want to load it."})
    recalculate_features: Optional[bool] = field(
        default=False, metadata={"help": "Whether to recalculate embeddings/features for the dataset."})
    use_extracted_features: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use embeddings/features for the dataset."})
    use_cpu: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use CPU for training. If False, will use GPU if available."})

    use_sentence_transformer: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a sentence transformer for context embedding. If False, will use the base model's embeddings."}
    )
    sentence_transformer_name: Optional[str] = field(
            default="all-MiniLM-L6-v2",
            metadata={"help": "The name of the sentence transformer model to use for context embedding. This is only used if use_sentence_transformer is True."},
        )

    clustering_algorithm: Optional[str] = field(
        default="kmeans",
        metadata={"help": "The clustering algorithm to use for context selection. Options: kmeans, gmm, spectral, agglomerative, kNLPmeans."},
    )
        
    save_postprocessed_and_feature_extracted_dataset: Optional[bool] = field(
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
    ctx_coefficient: Optional[float] = field(default=1.0, metadata={"help": "The coefficient for the context loss term in the total loss function. This is only used if the context implementation is not NO_CONTEXT."})
    entropy_coefficient: Optional[float] = field(default=0.0, metadata={"help": "The coefficient for the entropy/mutual information penalty on context-to-vs selection. This is only used if the context implementation is GMM."})

    vs_selection_coefficient: Optional[float] = field(default=1.0, metadata={"help": "The coefficient for the context loss term in the total loss function. This is only used if the context implementation is not NO_CONTEXT."})
    sharp_context_classification: Optional[bool] = field(default=True, metadata={"help": "Whether to use sharp classification for the context loss. If True, will use a hard classification for the context loss. If False, will use a soft classification for the context loss."})
    hidden_size: Optional[int] = field(
        default=1024, metadata={"help": "The hidden size of the grounding MLP."})
    vs_hidden_size: Optional[int] = field(
        default=24,  metadata={"help": "The hidden size of neuron layers for the MLP models used for predicting value systems. Depends on the context implementation."}
    )
    vs_layer_dropout: float = field(default=0,  metadata={"help": "The dropout probability of the models used for predicting value systems."})
    vs_layer_activation: Optional[str] = field(default="ReLU", metadata={
                                            "help": f"The activation function to use for the hidden layers of the value system prediction models. Use one of {list(VALUE_LAYER_ACTIVATIONS.keys())}"})
    vs_weight_initialization: Optional[str] = field(default="dirichlet", metadata={
                                            "help": f"The activation function to use for the hidden layers of the value system prediction models. Use one of {list(VALUE_LAYER_ACTIVATIONS.keys())}"})
    
    vs_num_hidden_layers: Optional[int] = field(default=0, metadata={
                                             "help": "The number of hidden layers in the value system prediction models. If 0, there will be no hidden layers and the value head will be a simple linear layer from the prompt-respose final embedding into the number of values."})
    
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
    context_learning_rate: Optional[float] = field(
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
    extra_weight_decay1: Optional[float] = field(default=0.001)
    extra_weight_decay2: Optional[float] = field(default=0.001)
    
    model_name: Optional[str] = field(
        # default="mistralai/Mistral-7B-Instruct-v0.2",
        # default="meta-llama/Llama-3.2-1B",
        # default="HuggingFaceTB/SmolLM-135M-Instruct",
        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
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

    smooth_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use smooth evaluation, i.e. select, when in eval mode, value system weights as a linear combination of value system probabilities (True), or rather, with argmax (False)."},
    )
    update_tendencies_every_n_steps: Optional[int] = field(
        default=1,
        metadata={"help": "How often to update the loss/metric tendencies for the Lagrange multiplier updates."},
    )
    normalize_context_features: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to normalize the context embeddings before clustering. When using GMM, this is set to true independetly of the given value."},
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
        default=-1,
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
    data_seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "Data seed for splittig datasets. Do not change"},
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

    # Context-specific arguments:
    max_value_systems: Optional[int] = field(
        default=3,
        metadata={"help": "The maximum number of value systems to use. If the dataset has more value systems than this, we will use the ones with the most examples."},
    )
    max_contexts: Optional[int] = field(
        default=10,
        metadata={"help": "The maximum number of contexts to use. If the dataset has more contexts than this, we will use the ones with the most examples."},
    )
    training_initialization_data_size: Optional[str] = field( 
        default="all",
        metadata={"help": "The maximum number of examples to use for training initialization step when assigning value systems to contexts. If the training dataset is larger than this, we will sample a subset of this size for the k-means clustering."},
    )
    do_initialization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to do the training initialization step when assigning value systems to contexts. If False, we will skip this step and use the initial value system assignments."},
    )
    do_vs_initialization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to do the value system initialization step when assigning value systems to contexts. If False, we will skip this step and use the initial value system assignments."},
    )
    context_implementation: Optional[str] = field(
        default="NO_CONTEXT",
        metadata={"help": "Choose between ContextImplementations in defines.py"}
    )
    direct_context_to_vs_relation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to consider a value system for each gmm-obtained context (no context-to-vs matrix)"},
    )
    detach_context_selection_for_value_system_selection: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to detach the context selection from the value system selection. If True, the context selection will not be used for the value system selection."},
    )
    detach_vs_selection_for_value_system_weight_training: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to detach the value system selection from the context selection. If True, the value system selection will not be used for the context selection."},
    )

    vae_pretrain_epochs: Optional[int] = field(
        default=10,
        metadata={"help": "The number of epochs to pretrain the VAE used for context selection. If 0, no pretraining will be done."},
    )
    vae_latent_dim: Optional[int] = field(
        default=10,
        metadata={"help": "The latent dimension of the VAE used for context selection. If 0, no VAE will be used."},
    )
    vae_type: Optional[str] = field(
        default="VAE",
        metadata={"help": "The type of VAE to use for context selection. Options: VAE, BetaVAE, FactorVAE, etc."},
    )
    vae_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "The dropout probability for the VAE used for context selection."},
    )
    vae_layer_activation: Optional[str] = field(
        default="ReLU",
        metadata={"help": "The activation function to use for the hidden layers of the VAE used for context selection. Use one of {list(VALUE_LAYER_ACTIVATIONS.keys())}"},
    )
    vae_reconstruction_loss: Optional[str] = field(
        default="mse",
        metadata={"help": "The reconstruction loss to use for the VAE used for context selection. Options: mse, bce, etc."},
    )   
    vae_n_hidden_layers: Optional[int] = field( 
        default=2,
        metadata={"help": "The number of hidden layers in the VAE used for context selection."},
    )
    vae_hidden_dim_size: Optional[int] = field(
        default=256,
        metadata={"help": "The hidden dimension size of the VAE used for context selection."},
    )
    vae_beta: Optional[float] = field(
        default=1.0,
        metadata={"help": "The beta parameter for the BetaVAE used for context selection."},
    )
    vae_final_encoder_layer_activation: Optional[str] = field(
        default="none",
        metadata={"help": "The activation function to use for the final layer of the VAE used for context selection. Use one of {list(VALUE_LAYER_ACTIVATIONS.keys())}"},
    )
    vae_resampling_iterations: Optional[float] = field(
        default=0.5,
        metadata={"help": "The logit threshold for the VAE used for context selection. If the logit is below this threshold, the context will be considered as not selected."},
    )
    vae_similarity: Optional[str] = field(
        default="cosine",
        metadata={"help": "The similarity metric to use for the VAE used for context selection. Options: cosine, euclidean, etc."},
    )
    vae_initial_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "The initial temperature for the Gumbel-Softmax used for context selection."},
    )
    vae_lambda_clustering: Optional[float] = field(
        default=1.0,
        metadata={"help": "The lambda parameter for the clustering loss used for context selection in VAE KMEANS: https://arxiv.org/pdf/1806.10069."},
    )


    
def argument_parser(script_args: ScriptArguments, class_source=ScriptArguments) -> Tuple[ScriptArguments, Dict[str, Any]]:

    if script_args.config_file:

        config_path = os.path.abspath(script_args.config_file)
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if not isinstance(config_data, dict):
            raise ValueError(
                f"Expected JSON object in config file, got {type(config_data).__name__}")

        valid_fields = set(class_source.__dataclass_fields__.keys())
        config_data["use_extracted_features"] = config_data.pop("use_embeddings", True)
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

        script_args = class_source(**merged_args)

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
    preset = MODEL_PRESETS[variant]

    if script_args.loss_func_type_kwargs is not None:
        if isinstance(script_args.loss_func_type_kwargs, str):  
            script_args.loss_func_type_kwargs = json.loads(
            script_args.loss_func_type_kwargs)
        else:
            script_args.loss_func_type_kwargs = script_args.loss_func_type_kwargs
    else:
        script_args.loss_func_type_kwargs = {}
    if "n_epochs_for_grounding" in script_args.loss_func_type_kwargs and isinstance(script_args.loss_func_type_kwargs["n_epochs_for_grounding"], float) and script_args.loss_func_type_kwargs["n_epochs_for_grounding"] < 1:
        script_args.loss_func_type_kwargs["n_epochs_for_grounding"] = int(script_args.loss_func_type_kwargs["n_epochs_for_grounding"] * script_args.num_train_epochs)
    print(f"Using loss function {script_args.loss_func_type} with kwargs {script_args.loss_func_type_kwargs}")
    if script_args.discordance_epsilon < 0:
        script_args.discordance_epsilon = None
    script_args.update_tendencies_every_n_steps = script_args.eval_every_steps if script_args.use_validation_for_tendencies else script_args.update_tendencies_every_n_steps

    if ContextImplementations(script_args.context_implementation) in [ContextImplementations.GMM, ContextImplementations.GMM_AND_CLASSIFIER,]:
        if script_args.normalize_context_features is False:
            print(f"Warning: Context implementation {script_args.context_implementation} requires normalized context features, but normalize_context_features is set to False. Make sure the dataset has normalized context features/embeddings!")
    if ContextImplementations(script_args.context_implementation) in [ContextImplementations.KMEANS_THEN_VS,]:
            script_args.do_initialization = True
    try:
        script_args.training_initialization_data_size = int(script_args.training_initialization_data_size)
    
    except ValueError:
        if script_args.training_initialization_data_size != "all":
            raise ValueError(
                f"training_initialization_data_size must be an integer or 'all', got {script_args.training_initialization_data_size}")
        script_args.training_initialization_data_size = "all"
    
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



def transform_weights_to_tuple(weights: Iterable|str, size_should_be: int = None, reduce_precision=False) -> tuple:
        if isinstance(weights, str):
            weights = weights.split(",")
        if isinstance(weights, th.Tensor):
            weights = weights.detach().tolist()
        assert len(weights) > 1, f"Unrecognized weights {weights}"
        if size_should_be is not None:
            assert len(weights) == size_should_be, f"Expected {size_should_be} weights, got {len(weights)}"
        if reduce_precision:
            weights_real = tuple([float(f"{float(a):.3f}") for a in weights])
        else:
            weights_real = tuple([float(a) for a in weights])
        return weights_real


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
        elif isinstance(value, th.Tensor):
            flat[key] = float(value.detach().cpu().item()) if value.numel() == 1 else float(value.detach().cpu().mean().item())
        elif isinstance(value, (float, int, str, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def compute_response_token_lengths(hf_dataset, tokenizer) -> np.ndarray:
    """Per-sample (interleaved option1, option2, option1, option2, ...) response token
    counts, tokenized from the dataset's raw `response1`/`response2` text columns
    (not the chat-templated `option1`/`option2`, so prompt tokens aren't counted).

    Requires a tokenizer (nlp_based task_type) and `response1`/`response2` columns on
    the dataset -- these are only retained when the dataset was (re)processed with them
    in `extra_keep_keys` (see `defines.py`'s `*_EXTRA_KEYS`). Asserts rather than
    degrading silently: re-run preprocessing with `--repostprocess` if this fails on a
    dataset processed before they were added.
    """
    assert tokenizer is not None, (
        "compute_response_token_lengths requires a tokenizer (nlp_based task_type); got tokenizer=None."
    )
    columns = hf_dataset.column_names
    assert "response1" in columns and "response2" in columns, (
        f"compute_response_token_lengths requires response1/response2 columns, got columns={columns!r}. "
        "Re-run preprocessing with --repostprocess to retain them (see defines.py's *_EXTRA_KEYS)."
    )
    response1 = hf_dataset["response1"]
    response2 = hf_dataset["response2"]
    lengths = np.empty(2 * len(response1), dtype=np.int64)
    lengths[0::2] = [len(tokenizer(str(text), add_special_tokens=False)["input_ids"]) for text in response1]
    lengths[1::2] = [len(tokenizer(str(text), add_special_tokens=False)["input_ids"]) for text in response2]
    return lengths

def entropy(p, eps=1e-12):    
        p = p.cpu().detach().numpy()    
        p = np.clip(p, eps, 1.0)  # avoid log(0)    
        return -np.sum(p * np.log(p))

def kmeans_clustering(dataset_ctxs: np.ndarray, K=None, max_iter=10000)-> KMeans:
        
            
        assert K is not None
        if dataset_ctxs.shape[0] == 0:
            raise ValueError("No context embeddings were found in the dataset, so KMeans cannot be fitted.")

        n_clusters = min(int(K), int(dataset_ctxs.shape[0]))
        random_state = 42
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto", random_state=random_state, max_iter=max_iter)
        kmeans.fit(dataset_ctxs)

        
        return kmeans

def auto_tsne(full_dataset: np.array, n_random_seeds=3, example_perps=(5, 30, 50), tsne_perplexity=-1, tsne_seed=-1, **kwargs) -> tuple[np.array, TSNE, float, float]:
        best_metric = None
        best_reducer = None
        best_perp = None

        """if full_dataset.shape[1] > 50:
            tsne_init = PCA(n_components=50).fit_transform(full_dataset)
        else:
            tsne_init = "random"
        """
        subset_for_selection = full_dataset if full_dataset.shape[0] < 1000 else full_dataset[np.random.choice(full_dataset.shape[0], 1000, replace=False)]
                
        perps_to_test = [*example_perps, 0.01*len(subset_for_selection), 0.05*len(subset_for_selection)]
        random_seeds = list(range(0,n_random_seeds))
        if tsne_perplexity > 0:
            perps_to_test = [tsne_perplexity]
        if tsne_seed >= 0:
            random_seeds = [tsne_seed]

        for perp in perps_to_test:
            for rs in random_seeds:
                print(f"TSNE {rs} with perplexity {perp}...")
                reducer_tsne: TSNE = TSNE(n_components=2, perplexity=perp, random_state=rs, init="pca")
                
                reduction_tsne = reducer_tsne.fit_transform(subset_for_selection) #if full_dataset.shape[1] > 50 else reducer_tsne.fit_transform(full_dataset)
                    # FROM: https://arxiv.org/pdf/1708.03229
                metric = 2*reducer_tsne.kl_divergence_ + np.log(len(subset_for_selection))*perp/len(subset_for_selection)
                print("Done")
                
                if  best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_reducer = reducer_tsne
                    best_perp = perp
                print("Metric: ", metric, "Best so far: ", best_metric)
        reduction_tsne = best_reducer.fit_transform(full_dataset)
        
        return reduction_tsne, best_reducer, best_perp, best_metric

from matplotlib.backends.backend_pdf import PdfPages


def plot_alternative_clusterings(
    features,
    label_sets,
    label_set_names=None,
    label_display_sets=None,
    dim_reduction="pca",
    output_path="clusterings.pdf"
):
    X = np.asarray(features)

    # --- Default names ---
    if label_set_names is None:
        label_set_names = [f"Clustering {i+1}" for i in range(len(label_sets))]

    if len(label_set_names) != len(label_sets):
        raise ValueError("label_set_names must match label_sets length")

    if label_display_sets is None:
        label_display_sets = [{} for _ in label_sets]
    if len(label_display_sets) != len(label_sets):
        raise ValueError("label_display_sets must match label_sets length")

    title_suffix = f"({dim_reduction.upper()})" if dim_reduction is not None else ""

    n_plots = len(label_sets)

    # --- Palettes and markers ---
    palettes = ["tab10", "tab20", "Set1", "Set2", "Dark2"]
    markers = ["o", "s", "^", "D", "P", "X"]

    with PdfPages(output_path) as pdf:

        # ==========================================================
        # 1. INDIVIDUAL CLUSTERINGS
        # ==========================================================
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        for i, (labels, name) in enumerate(zip(label_sets, label_set_names)):
            ax = axes[i]
            labels = np.asarray(labels)

            # Sort clusters by size
            unique, counts = np.unique(labels, return_counts=True)
            sorted_clusters = [
                u for u, _ in sorted(zip(unique, counts), key=lambda x: -x[1])
            ]

            cmap = plt.cm.get_cmap(palettes[i % len(palettes)], len(sorted_clusters))

            for j, lab in enumerate(sorted_clusters):
                mask = labels == lab
                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    color=cmap(j),
                    label=f"{label_display_sets[i].get(lab, f'Cluster {lab}')} (n={mask.sum()})",
                    s=30,
                )

            ax.set_title(f"{name} ({title_suffix})")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(title="Clusters", fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # 2. COMBINED PLOT
        # ==========================================================
        fig, ax = plt.subplots(figsize=(8, 7))

        for i, (labels, name) in enumerate(zip(label_sets, label_set_names)):
            labels = np.asarray(labels)

            unique, counts = np.unique(labels, return_counts=True)
            sorted_clusters = [
                u for u, _ in sorted(zip(unique, counts), key=lambda x: -x[1])
            ]

            cmap = plt.cm.get_cmap(palettes[i % len(palettes)], len(sorted_clusters))

def plot_groundings_violin(
    groundings: np.ndarray,
    value_names: Iterable[str],
    output_path: str,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "",
    ylabel: str = "predicted grounding",
    reference_line: Optional[float] = None,
) -> None:
    """
    Violin plots of the reward heads' per-value predictions (`groundings`, shape
    (num_samples, num_values)), one subplot per cluster id (a single "all" subplot if
    `cluster_labels` is None), with one violin per value inside each subplot.
    """
    value_names = list(value_names)
    num_values = groundings.shape[1]
    cluster_ids = np.zeros(len(groundings), dtype=int) if cluster_labels is None else np.asarray(cluster_labels)
    unique_clusters = [c for c in sorted(np.unique(cluster_ids).tolist()) if np.sum(cluster_ids == c) > 0]

    columns = min(4, len(unique_clusters))
    rows = (len(unique_clusters) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4 * rows), squeeze=False)
    for i, c in enumerate(unique_clusters):
        ax = axes.flat[i]
        data = [groundings[cluster_ids == c, v] for v in range(num_values)]
        ax.violinplot(data, showmeans=True, showmedians=True)
        if reference_line is not None:
            ax.axhline(reference_line, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, num_values + 1))
        ax.set_xticklabels(value_names, rotation=45, ha="right")
        ax.set_title(f"cluster {c}" if cluster_labels is not None else "all")
        ax.set_ylabel(ylabel)
    for ax in axes.flat[len(unique_clusters):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grounding_differences_violin(
    groundings: np.ndarray,
    value_names: Iterable[str],
    output_path: str,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    """
    Violin plots of the per-value grounding difference between the chosen and rejected
    response of each preference pair (chosen - rejected), one subplot per cluster id,
    one violin per value inside each subplot.

    `groundings` (shape (num_samples, num_values)) and `cluster_labels` (shape
    (num_samples,)) must be in the original interleaved
    (chosen, rejected, chosen, rejected, ...) order produced by the model -- i.e. the
    same order as `others["groundings"]` in `rewards_and_labels_to_logits_and_targets`
    (`logits[:, :-1]`), where consecutive rows (2*p, 2*p+1) form preference pair `p`.
    """
    n_pairs = len(groundings) // 2
    chosen = groundings[0:2 * n_pairs:2]
    rejected = groundings[1:2 * n_pairs:2]
    diffs = chosen - rejected
    pair_cluster_labels = np.asarray(cluster_labels)[0:2 * n_pairs:2] if cluster_labels is not None else None
    plot_groundings_violin(
        diffs,
        value_names,
        output_path,
        cluster_labels=pair_cluster_labels,
        title=title,
        ylabel="grounding difference (chosen - rejected)",
        reference_line=0.0,
    )


def plot_label_differences_violin(
    chosen_labels: np.ndarray,
    rejected_labels: np.ndarray,
    value_names: Iterable[str],
    output_path: str,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "",
    missing_value: float = NO_RATING_MASK,
) -> None:
    """
    Violin plots of the *ground-truth dataset* per-value label difference between the
    chosen and rejected response of each preference pair (chosen - rejected), one
    subplot per cluster id, one violin per value inside each subplot.

    `chosen_labels`/`rejected_labels` (shape (num_pairs, num_values), e.g.
    `dataset["labels"][:, 0, :-1]` / `[:, 1, :-1]`) and `cluster_labels` (shape
    (num_pairs,)) must be aligned row-for-row (one row per preference pair). A pair
    whose chosen or rejected rating for a given value equals `missing_value` (an
    undefined rating) is dropped from that value's violin rather than plotted as a
    bogus difference.
    """
    value_names = list(value_names)
    num_values = chosen_labels.shape[1]
    diffs = np.asarray(chosen_labels) - np.asarray(rejected_labels)
    invalid = (np.asarray(chosen_labels) == missing_value) | (np.asarray(rejected_labels) == missing_value)
    diffs = np.where(invalid, np.nan, diffs)

    cluster_ids = np.zeros(len(diffs), dtype=int) if cluster_labels is None else np.asarray(cluster_labels)
    unique_clusters = [c for c in sorted(np.unique(cluster_ids).tolist()) if np.sum(cluster_ids == c) > 0]

    columns = min(4, len(unique_clusters))
    rows = (len(unique_clusters) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4 * rows), squeeze=False)
    for i, c in enumerate(unique_clusters):
        ax = axes.flat[i]
        positions, data = [], []
        for v in range(num_values):
            column = diffs[cluster_ids == c, v]
            column = column[~np.isnan(column)]
            if len(column) > 0:
                positions.append(v + 1)
                data.append(column)
        if data:
            ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, num_values + 1))
        ax.set_xticklabels(value_names, rotation=45, ha="right")
        ax.set_title(f"cluster {c}" if cluster_labels is not None else "all")
        ax.set_ylabel("label difference (chosen - rejected)")
    for ax in axes.flat[len(unique_clusters):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(metrics: Dict[str, Any], output_path: str, name: str = "test_metrics.csv") -> None:
    path = Path(output_path).joinpath(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted(metrics.keys())
    #print(f"Writing metrics", metrics)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)