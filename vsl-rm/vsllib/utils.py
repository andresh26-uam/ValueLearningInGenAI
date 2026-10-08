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

from vsllib.defines import MODEL_DIR, MODEL_PRESETS, NO_RATING_MASK, ContextImplementations, infer_variant, LLMProvider, MOLossFunctions, SupportedDatasets, VALUE_LAYER_ACTIVATIONS


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
    ctx_chr_coefficient: Optional[float] = field(default=0.0, metadata={"help": "The coefficient for the context coherence loss. Unique in VAE_KMEANS."})
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
    run_name_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file where the actual run name will be written by the main process. Useful for tracking runs across subprocesses."},
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
    vae_orthogonality_coefficient: Optional[float] = field(
        default=0.0,
        metadata={"help": "The coefficient for the orthogonality loss used inspired from other works in VAE."},
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
    llm_provider: Optional[str] = field(
        default=LLMProvider.GROQ.value,
        metadata={"help": "Choose between LLMProvider in defines.py: GROQ, OPENROUTER. Used for LLM-based context cluster summarization."},
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
    script_args.llm_provider = LLMProvider(script_args.llm_provider).value
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


def compute_response_token_lengths(hf_dataset) -> np.ndarray:
    """Per-sample (interleaved option1, option2, option1, option2, ...) token counts,
    read directly from the dataset's already-tokenized `input_ids_1`/`input_ids_2`
    columns -- the exact tokenization used for the model's chat-templated (prompt +
    response) `option1`/`option2` inputs (not re-tokenized here, and not the response
    alone). These columns are always retained by preprocessing for nlp_based datasets
    (no `--repostprocess` re-run needed); feature_based datasets don't keep them. Since
    prompt length varies across examples, this length proxy can confound a
    length-vs-reward correlation with prompt-length effects.
    """
    columns = hf_dataset.column_names
    assert "input_ids_1" in columns and "input_ids_2" in columns, (
        f"compute_response_token_lengths requires input_ids_1/input_ids_2 columns "
        f"(nlp_based datasets only), got columns={columns!r}."
    )
    input_ids_1 = hf_dataset["input_ids_1"]
    input_ids_2 = hf_dataset["input_ids_2"]
    lengths = np.empty(2 * len(input_ids_1), dtype=np.int64)
    lengths[0::2] = [len(ids) for ids in input_ids_1]
    lengths[1::2] = [len(ids) for ids in input_ids_2]
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

def _to_numpy_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _cross_agreement(value_side, outcome_side, value_threshold: float, outcome_threshold: float,
                      value_missing: Optional[float] = None, outcome_missing: Optional[float] = None) -> np.ndarray:
    """Per-value percentage of pairs where `value_side`'s sign (relative to
    `value_threshold`) agrees with `outcome_side`'s sign (relative to
    `outcome_threshold`): both favor option1, both favor option2, or both are exactly
    tied at their own threshold. `value_side` has shape (n_pairs, num_values);
    `outcome_side` has shape (n_pairs,), broadcast across values. A ground-truth label
    uses threshold 0.5 and `NO_RATING_MASK` as its missing-value sentinel (undefined
    ratings are excluded from that value's percentage); a raw model prediction
    (logit) uses threshold 0.0 and has no missing-value concept (`*_missing=None`).
    """
    value_side = _to_numpy_array(value_side)
    outcome_side = _to_numpy_array(outcome_side)[:, None]

    missing = np.zeros_like(value_side, dtype=bool)
    if value_missing is not None:
        missing = missing | (value_side == value_missing)
    if outcome_missing is not None:
        missing = missing | (outcome_side == outcome_missing)

    agreement = (np.sign(value_side - value_threshold) == np.sign(outcome_side - outcome_threshold)).astype(float)
    agreement[missing] = np.nan
    with np.errstate(invalid="ignore"):
        return np.nanmean(agreement, axis=0) * 100.0


def compute_value_outcome_agreement(pairwise_labels) -> np.ndarray:
    """GTGR-To-GTVS: per-value percentage of pairs, over a whole set of pairs (not
    model-dependent), where the Ground-Truth per-value GRounding label (GTGR) agrees
    with the Ground-Truth overall Value-System/preference label (GTVS): value_i favors
    option1 (>0.5) and the overall label favors option1 (>0.5), value_i favors option2
    (<0.5) and the overall label favors option2 (<0.5), or value_i and the overall
    label are both exactly tied (==0.5).

    `pairwise_labels` is the ground-truth target-probability array (n_pairs, num_values
    + 1) -- i.e. `target_probs_p`/`pairwise_labels` as produced by
    `reward_pairs_and_scores_to_logits_and_targets` -- whose last column is the overall
    preference label and >0.5/<0.5/==0.5 already encode the sign of the underlying
    score difference (see `scores_to_target_probs`). Pairs where the value's own label
    or the overall label is undefined (`NO_RATING_MASK`) are excluded from that value's
    percentage, matching how `accuracy_logits`/`get_missing_rating_mask` treat
    undefined ratings elsewhere.
    """
    pairwise_labels = _to_numpy_array(pairwise_labels)
    return _cross_agreement(
        pairwise_labels[..., :-1], pairwise_labels[..., -1],
        value_threshold=0.5, outcome_threshold=0.5,
        value_missing=NO_RATING_MASK, outcome_missing=NO_RATING_MASK,
    )


def compute_lrgr_to_gtvs_agreement(pairwise_predictions, pairwise_labels) -> np.ndarray:
    """LRGR-To-GTVS: per-value percentage of pairs where the model's *predicted*
    per-value grounding (LRGR: Learned/predicted GRounding -- `pairwise_predictions`,
    raw logits, sign relative to 0) agrees with the *ground-truth* overall preference
    (GTVS: Ground-Truth Value-System/preference -- `pairwise_labels[..., -1]`, a
    probability, sign relative to 0.5). Both are (n_pairs, num_values + 1), aligned
    pair-for-pair (see `compute_value_outcome_agreement` for the ground-truth-only
    equivalent, GTGR-To-GTVS).
    """
    pairwise_predictions = _to_numpy_array(pairwise_predictions)
    pairwise_labels = _to_numpy_array(pairwise_labels)
    return _cross_agreement(
        pairwise_predictions[..., :-1], pairwise_labels[..., -1],
        value_threshold=0.0, outcome_threshold=0.5,
        value_missing=None, outcome_missing=NO_RATING_MASK,
    )


def compute_lrgr_to_lrvs_agreement(pairwise_predictions) -> np.ndarray:
    """LRGR-To-LRVS: per-value percentage of pairs where the model's *predicted*
    per-value grounding (LRGR) agrees with the model's own *predicted* overall
    value-system preference (LRVS: Learned/predicted Value-System --
    `pairwise_predictions[..., -1]`) -- both raw logits, sign relative to 0.
    `pairwise_predictions` is (n_pairs, num_values + 1); entirely model-derived, no
    ground truth involved.
    """
    pairwise_predictions = _to_numpy_array(pairwise_predictions)
    return _cross_agreement(
        pairwise_predictions[..., :-1], pairwise_predictions[..., -1],
        value_threshold=0.0, outcome_threshold=0.0,
        value_missing=None, outcome_missing=None,
    )


def compute_value_outcome_agreement_per_cluster(pairwise_labels, cluster_ids) -> Dict[Any, np.ndarray]:
    """`compute_value_outcome_agreement` (GTGR-To-GTVS), applied separately to each
    cluster's subset of pairs. `cluster_ids` must be a per-pair array, aligned
    one-per-row with `pairwise_labels` (one label per pair, not per sample).
    """
    pairwise_labels = _to_numpy_array(pairwise_labels)
    cluster_ids = _to_numpy_array(cluster_ids)
    return {
        c: compute_value_outcome_agreement(pairwise_labels[cluster_ids == c])
        for c in sorted(np.unique(cluster_ids).tolist())
    }


def compute_lrgr_to_gtvs_agreement_per_cluster(pairwise_predictions, pairwise_labels, cluster_ids) -> Dict[Any, np.ndarray]:
    """`compute_lrgr_to_gtvs_agreement` (LRGR-To-GTVS), applied separately to each
    cluster's subset of pairs.
    """
    pairwise_predictions = _to_numpy_array(pairwise_predictions)
    pairwise_labels = _to_numpy_array(pairwise_labels)
    cluster_ids = _to_numpy_array(cluster_ids)
    return {
        c: compute_lrgr_to_gtvs_agreement(pairwise_predictions[cluster_ids == c], pairwise_labels[cluster_ids == c])
        for c in sorted(np.unique(cluster_ids).tolist())
    }


def compute_lrgr_to_lrvs_agreement_per_cluster(pairwise_predictions, cluster_ids) -> Dict[Any, np.ndarray]:
    """`compute_lrgr_to_lrvs_agreement` (LRGR-To-LRVS), applied separately to each
    cluster's subset of pairs.
    """
    pairwise_predictions = _to_numpy_array(pairwise_predictions)
    cluster_ids = _to_numpy_array(cluster_ids)
    return {
        c: compute_lrgr_to_lrvs_agreement(pairwise_predictions[cluster_ids == c])
        for c in sorted(np.unique(cluster_ids).tolist())
    }


def format_value_outcome_agreements(agreements: Optional[Dict[str, Optional[np.ndarray]]]) -> Optional[str]:
    """Multi-line rendering of one or more named agreement results -- e.g.
    `{"GTGR-To-GTVS": compute_value_outcome_agreement(...), "LRGR-To-GTVS":
    compute_lrgr_to_gtvs_agreement(...), "LRGR-To-LRVS":
    compute_lrgr_to_lrvs_agreement(...)}` -- as one "<LABEL>: [...]" line per entry, in
    `agreements`' insertion order, e.g. for a figure suptitle or a per-cluster subplot
    title. Returns None if `agreements` is empty/None or every value in it is None.
    """
    if not agreements:
        return None
    lines = [
        f"{label}: [" + ", ".join(f"{pct:.1f}%" for pct in values) + "]"
        for label, values in agreements.items() if values is not None
    ]
    return "\n".join(lines) if lines else None


def _agreements_for_cluster(agreements_per_cluster: Optional[Dict[str, Optional[Dict[Any, np.ndarray]]]], cluster_id) -> Dict[str, np.ndarray]:
    """Picks out one cluster's row from each named `compute_*_agreement_per_cluster`
    result in `agreements_per_cluster` (a dict of label -> {cluster_id -> array}),
    for `format_value_outcome_agreements`.
    """
    if not agreements_per_cluster:
        return {}
    return {
        label: per_cluster.get(cluster_id)
        for label, per_cluster in agreements_per_cluster.items()
        if per_cluster is not None
    }


def plot_groundings_violin(
    groundings: np.ndarray,
    value_names: Iterable[str],
    output_path: str,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "",
    ylabel: str = "predicted grounding",
    reference_line: Optional[float] = None,
    value_outcome_agreements: Optional[Dict[str, Optional[np.ndarray]]] = None,
    value_outcome_agreements_per_cluster: Optional[Dict[str, Optional[Dict[Any, np.ndarray]]]] = None,
) -> None:
    """
    Violin plots of the reward heads' per-value predictions (`groundings`, shape
    (num_samples, num_values)), one subplot per cluster id (a single "all" subplot if
    `cluster_labels` is None), with one violin per value inside each subplot.

    `value_outcome_agreements`/`value_outcome_agreements_per_cluster` -- named results
    from `compute_value_outcome_agreement` (GTGR-To-GTVS), `compute_lrgr_to_gtvs_agreement`
    (LRGR-To-GTVS), `compute_lrgr_to_lrvs_agreement` (LRGR-To-LRVS), and their
    `_per_cluster` counterparts, keyed by those same labels -- are diagnostics unrelated
    to the violins' own data, shown once above the whole figure (dataset-wide) and, per
    cluster id, inside that subplot's own title.
    """
    value_names = list(value_names)
    num_values = groundings.shape[1]
    cluster_ids = np.zeros(len(groundings), dtype=int) if cluster_labels is None else np.asarray(cluster_labels)
    unique_clusters = [c for c in sorted(np.unique(cluster_ids).tolist()) if np.sum(cluster_ids == c) > 0]

    columns = min(4, len(unique_clusters))
    rows = (len(unique_clusters) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 5.5 * rows), squeeze=False)
    for i, c in enumerate(unique_clusters):
        ax = axes.flat[i]
        positions, data = [], []
        for v in range(num_values):
            column = groundings[cluster_ids == c, v]
            column = column[~np.isnan(column)]
            if len(column) > 0:
                positions.append(v + 1)
                data.append(column)
        if data:
            ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        if reference_line is not None:
            ax.axhline(reference_line, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, num_values + 1))
        ax.set_xticklabels(value_names, rotation=45, ha="right")
        subplot_title = f"cluster {c}" if cluster_labels is not None else "all"
        cluster_agreement = format_value_outcome_agreements(
            _agreements_for_cluster(value_outcome_agreements_per_cluster, c))
        if cluster_agreement is not None:
            subplot_title += f"\n{cluster_agreement}"
        ax.set_title(subplot_title)
        ax.set_ylabel(ylabel)
    for ax in axes.flat[len(unique_clusters):]:
        ax.axis("off")
    suptitle_agreement = format_value_outcome_agreements(value_outcome_agreements)
    full_title = f"{title}\n{suptitle_agreement}" if suptitle_agreement is not None else title
    fig.tight_layout(pad=2.5, h_pad=5.0, w_pad=3.0)
    fig.subplots_adjust(hspace=1.0, wspace=0.4, top=0.86 if suptitle_agreement is not None else None)
    fig.suptitle(full_title)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grounding_differences_violin(
    groundings: np.ndarray,
    value_labels: np.ndarray,
    value_names: Iterable[str],
    output_path: str,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "",
    missing_value: float = NO_RATING_MASK,
    value_outcome_agreements: Optional[Dict[str, Optional[np.ndarray]]] = None,
    value_outcome_agreements_per_cluster: Optional[Dict[str, Optional[Dict[Any, np.ndarray]]]] = None,
) -> None:
    """
    Violin plots of the per-value grounding (model-predicted reward) difference
    between the two responses of each preference pair, ordered *independently per
    value*: for value `v`, the response with the higher ground-truth `v` rating goes
    first in the subtraction (so the sign always means "response rated higher on this
    value, minus response rated lower on this value" for that value's own label --
    unlike `plot_grounding_differences_violin_by_overall_score`, which uses a single
    ordering, from the overall preference score, applied to every value column alike).
    One subplot per cluster id, one violin per value inside each subplot.

    `groundings` (shape (num_samples, num_values)) and `cluster_labels` (shape
    (num_samples,)) must be in the original interleaved (option1, option2, option1,
    option2, ...) order produced by the model -- i.e. the same order as
    `others["groundings"]` in `rewards_and_labels_to_logits_and_targets`
    (`logits[:, :-1]`), where consecutive rows (2*p, 2*p+1) form preference pair `p`.
    Note this is NOT (preferred, non-preferred, preferred, non-preferred, ...):
    `response1`/`option1` is not guaranteed to be the preferred response (see
    `score1`/`score2` in the preprocessing scripts) -- that's exactly why per-value
    and per-overall-score orderings can disagree and are both worth plotting.

    `value_labels` (shape (num_pairs, 2, num_values), e.g. `dataset["labels"][:, :,
    :-1]`) are the ground-truth per-value ratings for option1 (`value_labels[:, 0, :]`)
    and option2 (`value_labels[:, 1, :]`), aligned one-row-per-pair with `groundings`'
    pairs. A (pair, value) entry where either rating equals `missing_value` (an
    undefined rating -- ordering is undefined without it) is dropped from that value's
    violin rather than plotted with an arbitrary sign.
    """
    n_pairs = len(groundings) // 2
    option1 = groundings[0:2 * n_pairs:2]
    option2 = groundings[1:2 * n_pairs:2]
    n_pairs = min(n_pairs, len(value_labels))
    option1, option2 = option1[:n_pairs], option2[:n_pairs]
    label1 = np.asarray(value_labels)[:n_pairs, 0, :]
    label2 = np.asarray(value_labels)[:n_pairs, 1, :]

    option1_is_higher = label1 >= label2
    diffs = np.where(option1_is_higher, option1 - option2, option2 - option1)
    invalid = (label1 == missing_value) | (label2 == missing_value)
    diffs = np.where(invalid, np.nan, diffs)

    pair_cluster_labels = np.asarray(cluster_labels)[0:2 * n_pairs:2] if cluster_labels is not None else None
    plot_groundings_violin(
        diffs,
        value_names,
        output_path,
        cluster_labels=pair_cluster_labels,
        title=title,
        ylabel="grounding difference",
        reference_line=0.0,
        value_outcome_agreements=value_outcome_agreements,
        value_outcome_agreements_per_cluster=value_outcome_agreements_per_cluster,
    )


def plot_grounding_differences_violin_by_overall_score(
    groundings: np.ndarray,
    overall_scores: np.ndarray,
    value_names: Iterable[str],
    output_path: str,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "",
    missing_value: float = NO_RATING_MASK,
    value_outcome_agreements: Optional[Dict[str, Optional[np.ndarray]]] = None,
    value_outcome_agreements_per_cluster: Optional[Dict[str, Optional[Dict[Any, np.ndarray]]]] = None,
) -> None:
    """
    Violin plots of the per-value grounding (model-predicted reward) difference
    between the two responses of each preference pair, ordered by the ground-truth
    *overall* preference score -- one ordering per pair (whichever response has the
    higher overall score goes first), applied identically to every value column,
    regardless of that value's own label. This answers "for the response actually
    preferred overall, does the model predict higher per-value rewards across all
    values?" -- unlike `plot_grounding_differences_violin`, which orders each value
    column independently by that value's own ground-truth label.

    `groundings`/`cluster_labels` must be in the interleaved (option1, option2, ...)
    order, same convention as `plot_grounding_differences_violin`. `overall_scores`
    (shape (num_pairs, 2), e.g. `dataset["labels"][:, :, -1]`) are the ground-truth
    overall preference score for option1/option2, aligned one-row-per-pair.
    """
    n_pairs = len(groundings) // 2
    option1 = groundings[0:2 * n_pairs:2]
    option2 = groundings[1:2 * n_pairs:2]
    n_pairs = min(n_pairs, len(overall_scores))
    option1, option2 = option1[:n_pairs], option2[:n_pairs]
    score1 = np.asarray(overall_scores)[:n_pairs, 0]
    score2 = np.asarray(overall_scores)[:n_pairs, 1]

    option1_is_preferred = score1 >= score2
    diffs = np.where(option1_is_preferred[:, None], option1 - option2, option2 - option1)
    invalid = (score1 == missing_value) | (score2 == missing_value)
    diffs = np.where(invalid[:, None], np.nan, diffs)

    pair_cluster_labels = np.asarray(cluster_labels)[0:2 * n_pairs:2] if cluster_labels is not None else None
    plot_groundings_violin(
        diffs,
        value_names,
        output_path,
        cluster_labels=pair_cluster_labels,
        title=title,
        ylabel="grounding difference",
        reference_line=0.0,
        value_outcome_agreements=value_outcome_agreements,
        value_outcome_agreements_per_cluster=value_outcome_agreements_per_cluster,
    )


def plot_label_differences_violin(
    value_labels: np.ndarray,
    overall_scores: np.ndarray,
    value_names: Iterable[str],
    output_path: str,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "",
    missing_value: float = NO_RATING_MASK,
    value_outcome_agreements: Optional[Dict[str, Optional[np.ndarray]]] = None,
    value_outcome_agreements_per_cluster: Optional[Dict[str, Optional[Dict[Any, np.ndarray]]]] = None,
) -> None:
    """
    Violin plots of the *ground-truth dataset* per-value label difference between the
    two responses of each preference pair, ordered by the ground-truth *overall*
    preference score -- one ordering per pair (whichever response has the higher
    overall score goes first), applied identically to every value column, regardless
    of that value's own label (same convention as
    `plot_grounding_differences_violin_by_overall_score`, applied to the ground-truth
    labels instead of the model's predicted groundings). One subplot per cluster id,
    one violin per value inside each subplot.

    There is no fixed "preferred" side by construction -- `option1`/`option2` (and
    correspondingly `value_labels[:, 0, :]`/`value_labels[:, 1, :]`) carry no inherent
    preference ordering; for each pair, whichever option's overall score is higher is
    used as the first term of the subtraction.

    `value_labels` (shape (num_pairs, 2, num_values), e.g. `dataset["labels"][:, :,
    :-1]`) are the ground-truth per-value ratings for option1/option2, and
    `overall_scores` (shape (num_pairs, 2), e.g. `dataset["labels"][:, :, -1]`) are
    the ground-truth overall preference score for option1/option2, aligned
    one-row-per-pair with `cluster_labels`. A pair whose overall score, or whose
    rating for a given value, equals `missing_value` (an undefined rating -- ordering
    is undefined without it) is dropped from that value's violin rather than plotted
    with an arbitrary sign.

    `value_outcome_agreements`/`value_outcome_agreements_per_cluster` -- see
    `plot_groundings_violin` -- are named diagnostics shown above the whole figure and
    inside each subplot's own title, respectively.
    """
    value_names = list(value_names)
    value_labels = np.asarray(value_labels)
    num_values = value_labels.shape[-1]
    score1 = np.asarray(overall_scores)[:, 0]
    score2 = np.asarray(overall_scores)[:, 1]
    label1 = value_labels[:, 0, :]
    label2 = value_labels[:, 1, :]

    option1_is_preferred = score1 >= score2
    diffs = np.where(option1_is_preferred[:, None], label1 - label2, label2 - label1)
    invalid = (
        (score1 == missing_value)[:, None] | (score2 == missing_value)[:, None]
        | (label1 == missing_value) | (label2 == missing_value)
    )
    diffs = np.where(invalid, np.nan, diffs)

    cluster_ids = np.zeros(len(diffs), dtype=int) if cluster_labels is None else np.asarray(cluster_labels)
    unique_clusters = [c for c in sorted(np.unique(cluster_ids).tolist()) if np.sum(cluster_ids == c) > 0]

    columns = min(4, len(unique_clusters))
    rows = (len(unique_clusters) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 5.5 * rows), squeeze=False)
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
        subplot_title = f"cluster {c}" if cluster_labels is not None else "all"
        cluster_agreement = format_value_outcome_agreements(
            _agreements_for_cluster(value_outcome_agreements_per_cluster, c))
        if cluster_agreement is not None:
            subplot_title += f"\n{cluster_agreement}"
        ax.set_title(subplot_title)
        ax.set_ylabel("label difference")
    for ax in axes.flat[len(unique_clusters):]:
        ax.axis("off")
    suptitle_agreement = format_value_outcome_agreements(value_outcome_agreements)
    full_title = f"{title}\n{suptitle_agreement}" if suptitle_agreement is not None else title
    fig.tight_layout(pad=2.5, h_pad=5.0, w_pad=3.0)
    fig.subplots_adjust(hspace=1.0, wspace=0.4, top=0.86 if suptitle_agreement is not None else None)
    fig.suptitle(full_title)
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


# ---------------------------------------------------------------------------
# Clustering analysis
# ---------------------------------------------------------------------------
# Two registries, keyed by metric name; adding a metric only requires a new entry:
#   * CLUSTERING_SIMILARITY_METRICS -- compare two labelings of the same points.
#     `fn(ground_truth_labels, predicted_labels) -> float`. Asymmetric metrics are
#     reported in both directions by `compare_clusterings`.
#   * CLUSTERING_QUALITY_METRICS -- score a single labeling of a feature space.
#     `fn(features, labels) -> float`.
# `analyze_clustering` additionally scores a labeling against reference labelings
# (e.g. KMeans taken as ground truth) with the similarity metrics named in
# CLUSTERING_REFERENCE_METRICS.

from typing import Callable, NamedTuple, Sequence
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix


def _as_float32_array(x) -> np.ndarray:
    # bfloat16/float16 tensors cannot be converted to numpy directly.
    if isinstance(x, th.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _has_multiple_clusters(labels: np.ndarray) -> bool:
    return len(np.unique(labels)) >= 2


def hungarian_accuracy(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """Clustering accuracy (ACC, as in DEC/VaDE): fraction of points correctly labeled
    under the best one-to-one cluster matching (Hungarian algorithm). Symmetric; with
    different cluster counts, the surplus clusters stay unmatched and count as errors."""
    counts = contingency_matrix(ground_truth, predicted)
    rows, cols = linear_sum_assignment(counts, maximize=True)
    return float(counts[rows, cols].sum() / counts.sum())


def majority_mapping_accuracy(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """Many-to-one accuracy (a.k.a. purity): every predicted cluster is mapped to its
    most frequent ground-truth cluster. Asymmetric; favors over-segmented predictions."""
    counts = contingency_matrix(ground_truth, predicted)
    return float(counts.max(axis=0).sum() / counts.sum())


def variation_of_information(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """Variation of Information (Meila, 2007): H(A) + H(B) - 2 I(A; B), in nats. A true
    metric between partitions; 0 means identical partitions, lower is better."""
    counts = contingency_matrix(ground_truth, predicted).astype(np.float64)
    p_rows = counts.sum(axis=1) / counts.sum()
    p_cols = counts.sum(axis=0) / counts.sum()
    h_rows = -np.sum(p_rows * np.log(p_rows))
    h_cols = -np.sum(p_cols * np.log(p_cols))
    return float(h_rows + h_cols - 2.0 * mutual_info_score(None, None, contingency=counts))


def within_cluster_sum_of_squares(features: np.ndarray, labels: np.ndarray) -> float:
    """WCSS (a.k.a. inertia): sum of squared Euclidean distances of every point to its
    cluster centroid. Lower is better; grows with the number of points and dimensions."""
    total = 0.0
    for cluster in np.unique(labels):
        members = features[labels == cluster].astype(np.float64)
        total += float(np.sum((members - members.mean(axis=0)) ** 2))
    return total


def ray_turi_index(features: np.ndarray, labels: np.ndarray) -> float:
    """Ray-Turi index (Ray & Turi, 1999): mean squared distance of points to their
    centroid (WCSS / N) divided by the minimum squared distance between two centroids.
    Lower is better."""
    if not _has_multiple_clusters(labels):
        return float("nan")
    centroids = np.stack([features[labels == c].astype(np.float64).mean(axis=0) for c in np.unique(labels)])
    min_separation = pdist(centroids, "sqeuclidean").min()
    if min_separation == 0:
        return float("nan")
    return within_cluster_sum_of_squares(features, labels) / len(labels) / float(min_separation)


def dunn_index(features: np.ndarray, labels: np.ndarray, chunk_size: int = 2048) -> float:
    """Dunn index (Dunn, 1974): minimum distance between points of different clusters
    divided by the maximum intra-cluster diameter. Higher is better; sensitive to
    outliers. Exact, computed on row chunks of the distance matrix (GPU if available)
    after dropping exact-duplicate (point, label) rows, which change neither extreme."""
    if not _has_multiple_clusters(labels):
        return float("nan")
    _, keep = np.unique(np.column_stack([features, labels.astype(np.float32)]), axis=0, return_index=True)
    device = "cuda" if th.cuda.is_available() else "cpu"
    points = th.as_tensor(features[keep], dtype=th.float32, device=device)
    point_labels = th.as_tensor(labels[keep], device=device)
    min_inter = float("inf")
    max_intra = 0.0
    for start in range(0, len(points), chunk_size):
        distances = th.cdist(points[start:start + chunk_size], points)
        same_cluster = point_labels[start:start + chunk_size, None] == point_labels[None, :]
        max_intra = max(max_intra, distances.masked_fill(~same_cluster, 0.0).max().item())
        min_inter = min(min_inter, distances.masked_fill(same_cluster, float("inf")).min().item())
    if max_intra == 0:
        return float("nan")
    return min_inter / max_intra


def silhouette_index(features: np.ndarray, labels: np.ndarray, max_samples: int = 5000, seed: int = 0) -> float:
    """Mean silhouette coefficient (Rousseeuw, 1987) in [-1, 1], higher is better.
    Estimated on a random subsample of at most `max_samples` points (O(N^2) otherwise)."""
    if not _has_multiple_clusters(labels):
        return float("nan")
    return float(silhouette_score(features, labels, sample_size=min(len(labels), max_samples), random_state=seed))


class SimilarityMetric(NamedTuple):
    fn: Callable[[np.ndarray, np.ndarray], float]
    symmetric: bool


CLUSTERING_SIMILARITY_METRICS: Dict[str, SimilarityMetric] = {
    "nmi": SimilarityMetric(normalized_mutual_info_score, symmetric=True),
    "ami": SimilarityMetric(adjusted_mutual_info_score, symmetric=True),
    "ari": SimilarityMetric(adjusted_rand_score, symmetric=True),
    "rand_index": SimilarityMetric(rand_score, symmetric=True),
    "fowlkes_mallows": SimilarityMetric(fowlkes_mallows_score, symmetric=True),
    "v_measure": SimilarityMetric(v_measure_score, symmetric=True),
    "variation_of_information": SimilarityMetric(variation_of_information, symmetric=True),
    "hungarian_accuracy": SimilarityMetric(hungarian_accuracy, symmetric=True),
    "majority_accuracy": SimilarityMetric(majority_mapping_accuracy, symmetric=False),
    # homogeneity with the ground truth swapped is completeness.
    "homogeneity": SimilarityMetric(homogeneity_score, symmetric=False),
}

CLUSTERING_QUALITY_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "dunn_index": dunn_index,
    "ray_turi_index": ray_turi_index,
    "wcss": within_cluster_sum_of_squares,
    "davies_bouldin_index": lambda x, y: float(davies_bouldin_score(x, y)) if _has_multiple_clusters(y) else float("nan"),
    "calinski_harabasz_index": lambda x, y: float(calinski_harabasz_score(x, y)) if _has_multiple_clusters(y) else float("nan"),
    "silhouette": silhouette_index,
}

CLUSTERING_REFERENCE_METRICS: Tuple[str, ...] = ("hungarian_accuracy", "majority_accuracy")


def compare_clusterings(labels_a, labels_b, name_a: str = "a", name_b: str = "b",
                        metrics: Optional[Dict[str, SimilarityMetric]] = None) -> Dict[str, float]:
    """Similarity between two labelings of the same points. Symmetric metrics are keyed
    by their name; asymmetric ones are computed twice, keyed `<name>_gt_<name_a>`
    (labels_a taken as ground truth) and `<name>_gt_<name_b>`."""
    labels_a = np.asarray(_to_numpy_array(labels_a)).ravel()
    labels_b = np.asarray(_to_numpy_array(labels_b)).ravel()
    if len(labels_a) != len(labels_b):
        raise ValueError(f"Cannot compare clusterings of different sizes: {len(labels_a)} vs {len(labels_b)}")
    results = {}
    for name, metric in (metrics or CLUSTERING_SIMILARITY_METRICS).items():
        if metric.symmetric:
            results[name] = float(metric.fn(labels_a, labels_b))
        else:
            results[f"{name}_gt_{name_a}"] = float(metric.fn(labels_a, labels_b))
            results[f"{name}_gt_{name_b}"] = float(metric.fn(labels_b, labels_a))
    return results


def analyze_clustering(features, labels, reference_labels: Optional[Dict[str, Any]] = None,
                       quality_metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
                       reference_metrics: Sequence[str] = CLUSTERING_REFERENCE_METRICS) -> Dict[str, float]:
    """Internal quality of one labeling of `features` (n_points, dim), plus its agreement
    with each reference labeling in `reference_labels` ({name: labels}, taken as ground
    truth), keyed `<metric>_vs_<name>` (symmetric) or `<metric>_gt_<name>` (asymmetric)."""
    features = _as_float32_array(features)
    labels = np.asarray(_to_numpy_array(labels)).ravel()
    if len(features) != len(labels):
        raise ValueError(f"features ({len(features)}) and labels ({len(labels)}) must be aligned one-per-point.")
    results = {"n_points": len(labels), "n_clusters": len(np.unique(labels))}
    for name, fn in (quality_metrics or CLUSTERING_QUALITY_METRICS).items():
        results[name] = float(fn(features, labels))
    for reference_name, reference in (reference_labels or {}).items():
        reference = np.asarray(_to_numpy_array(reference)).ravel()
        for metric_name in reference_metrics:
            metric = CLUSTERING_SIMILARITY_METRICS[metric_name]
            key = f"{metric_name}_vs_{reference_name}" if metric.symmetric else f"{metric_name}_gt_{reference_name}"
            results[key] = float(metric.fn(reference, labels))
    return results

def orthogonality_loss(vector: th.Tensor, orthonormal_target: Optional[bool] = False) -> th.Tensor:
    if orthonormal_target:
        return th.norm(vector @ vector.T - th.eye(vector.size(0), device=vector.device))
    else:
        return th.norm(vector @ vector.T - th.diag_embed((vector * vector).sum(dim=-1)))