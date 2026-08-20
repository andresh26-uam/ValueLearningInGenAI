
from abc import abstractmethod
from copy import deepcopy
from http.client import NO_CONTENT
from operator import truediv
from re import A
from regex import P
from sympy import Abs
from tokenizers.decoders import CTC
import tqdm
from typing_extensions import Self

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast, ModelOutput
from collections.abc import Iterator
from dataclasses import dataclass

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple
import matplotlib.pyplot as plt
import os

import numpy as np
import torch as th
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, InputExample, PreTrainedModel, TrainingArguments
from transformers.utils import logging
from transformers.cache_utils import Cache
from datasets import Dataset

from sklearn.cluster import KMeans

from vsllib.training_utils import MORMTrainingVariables
from vsllib.defines import CONTEXT_EMBEDDING_FEATURE_NAME, CONTEXT_FEATURE_NAME, MIN_EPSILON, NO_RATING_MASK, SCORE_DIFF_EPSILON, VALUE_LAYER_ACTIVATIONS, ContextImplementations, MOLossFunctions, MOLossFunctionsCategories, MOLossManagement


logger = logging.get_logger(__name__)


def compute_per_centroid_variance(
    data: th.Tensor,
    centroids: th.Tensor,
    assignments: th.Tensor,
    min_var: float = 1e-6,
    return_log: bool = False
) -> th.Tensor:
    """Compute per-centroid variance from data and cluster assignments.
    
    Calculates the variance of data points within each cluster by computing
    the mean squared difference from the centroid for each dimension.
    
    Args:
        data: [B, D] - input data points
        centroids: [K, D] - cluster centroids
        assignments: [B] - hard cluster assignment for each point (values in 0..K-1)
        min_var: minimum variance to prevent log(0) issues
        return_log: if True, return log(variance); else return variance
        
    Returns:
        variance: [K, D] - per-centroid, per-dimension variance
                  or log(variance) if return_log=True
    """
    num_components = centroids.shape[0]
    input_size = centroids.shape[1]
    device = centroids.device
    dtype = centroids.dtype
    
    # Initialize variance accumulator
    variance = th.zeros(
        num_components,
        input_size,
        device=device,
        dtype=dtype
    )
    
    # Compute variance for each component
    for k in range(num_components):
        mask = assignments == k
        if mask.sum() > 0:
            # Get points assigned to this centroid
            points_k = data[mask]  # [n_k, D]
            centroid_k = centroids[k]  # [D]
            
            # Compute squared differences and take mean
            diff_squared = (points_k - centroid_k).pow(2)  # [n_k, D]
            variance[k] = diff_squared.mean(dim=0).clamp(min=min_var)
        else:
            # No points assigned, use minimum variance
            variance[k] = min_var
    
    if return_log:
        return th.log(variance)
    return variance


def construct_layers(input_dim, hidden_sizes, intermediate_activation, dropout, device, dtype, n_outputs, final_activation, final_activation_kwargs):
    layers=[]
    try:
        intermediate_activation = VALUE_LAYER_ACTIVATIONS[
            intermediate_activation]
    except KeyError:
        raise ValueError(
            f"Unsupported intermediate activation: {intermediate_activation}")

    if intermediate_activation is None:
        raise ValueError(
            f"Unsupported intermediate activation: {intermediate_activation}")
    final_size = input_dim
    input_aux = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(input_aux, hidden_size,
                      dtype=dtype, device=device))

        layers.append(intermediate_activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        final_size = hidden_size
        input_aux = hidden_size
    layers.append(nn.Linear(final_size, n_outputs,
                  dtype=dtype, device=device))

    try:
        final_activation = VALUE_LAYER_ACTIVATIONS[final_activation]
    except KeyError:
        raise ValueError(
            f"Unsupported final activation: {final_activation}")
    if final_activation is not None:
        layers.append(final_activation(**final_activation_kwargs))
    return layers

class AlignmentLayer(th.nn.Module):

    @abstractmethod
    def forward(self, grounding: th.Tensor, **kwargs) -> Tuple[th.Tensor, Dict]:
        pass
    @abstractmethod 
    def get_value_system_info(self) -> Any:
        pass

class LinearAlignmentLayer(th.nn.Linear, AlignmentLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None, data=None, n_values=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.linear_bias = bias
        self.n_values = in_features if n_values is None else n_values

        with th.no_grad():
            state_dict = self.state_dict()
            random_vector = th.rand_like(state_dict['weight'])
            state_dict['weight'] = th.nn.functional.sigmoid(
                state_dict['weight']) * random_vector

            self.load_state_dict(state_dict)

    #@th.compile
    def forward(self, grounding: th.Tensor, **kwargs) -> Tuple[th.Tensor, Dict]:
        # assert w_bounded.dtype == self.weight.dtype, f"Expected w_bounded dtype {self.weight.dtype}, but got {w_bounded.dtype}"
        # assert w_bounded.device == self.weight.device, f"Expected w_bounded device {self.weight.device}, but got {w_bounded.device}"
        return th.nn.functional.linear(grounding, self.get_alignment_layer()), {}
        # assert input.shape[-1] == self.n_values, f"Expected output shape to have last dimension {self.n_values}, but got {output.shape}"
        # return output

    def get_alignment_layer(self):
        return self.weight
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))

    def get_value_system_info(self) -> List[float]:
        with th.no_grad():
            return self.get_alignment_layer().detach().view(-1).cpu().tolist()

    
    def value_system_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()
    
    def context_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()

class ConvexAlignmentLayer(LinearAlignmentLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=th.float32, data=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype, data)
        self.set_weights([1.0/self.weight.shape[1]
                         for _ in range(self.weight.shape[1])])
        

    def set_weights(self, weights: tuple):
        with th.no_grad():
            # Convert to tensor with same dtype and device as self.weight
            pure_w = th.tensor(weights, dtype=self.weight.dtype,
                               device=self.weight.device)
            new_weights = th.log(pure_w+1e-8)
            # Reshape to match weight shape
            new_weights = new_weights.view_as(self.weight)
            # Ensure requires_grad matches previous setting
            new_weights.requires_grad = self.weight.requires_grad
            # Update state dict in place
            self.load_state_dict({'weight': new_weights}, strict=False)

            # During HuggingFace from_pretrained, modules may be initialized on meta device.
            # Avoid value assertions that materialize meta tensors.
            if not self.weight.is_meta:
                assert th.allclose(pure_w, th.nn.functional.softmax(
                    self.weight, dim=1)), f"{new_weights} vs {th.nn.functional.softmax(self.weight, dim=1)}"

    @th.compile
    def get_alignment_layer(self) -> th.Tensor:
        return th.nn.functional.softmax(self.weight, dim=1, dtype=self.weight.dtype)


from dataclasses import replace
from dataclasses import fields

@dataclass(frozen=True, eq=False)
class CtxData:    
    context_features: th.Tensor    
    context_logprobs: th.Tensor    
    vs_logprobs: th.Tensor
    vs_assignments: Optional[th.Tensor] = None    
    ctx_assignments: Optional[th.Tensor] = None    
    vs_predicted: Optional[th.Tensor] = None    
    ctx_predicted: Optional[th.Tensor] = None
    ctx_possibilities: Optional[th.Tensor] = None
    vs_possibilities: Optional[th.Tensor] = None

    vs_pred_diff: Optional[th.Tensor] = None
    vs_pred_loss: Optional[th.Tensor] = None

    ctx_pred_diff: Optional[th.Tensor] = None
    ctx_pred_loss: Optional[th.Tensor] = None

    extra_for_custom_loss: Optional[Tuple[th.Tensor]] = None

    @staticmethod
    def from_previous(other: Self, **extra_kwargs) -> Self:
        return replace(other, **extra_kwargs)
    
    def to_dict(self) -> Dict:    
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}
    
    @staticmethod
    def from_dict(dict_info: dict, to_tensor: bool = True) -> Self:
        constructed_fields = dict()
        for k, v in dict_info.items():
            #print(k,v, fields(CtxData))
            if k in [f.name for f in fields(CtxData)]:
                if isinstance(v, np.ndarray) and to_tensor:
                    v = th.tensor(v)
                constructed_fields[k] = v 
            else:
                raise ValueError(f"UNRECOGNIZED FIELD: {k}, VALUE: {v}")
        return CtxData(**constructed_fields)
        
@dataclass(frozen=False)
class CtxStatistics:
    centroids: th.Tensor = None
    deviations: th.Tensor = None
    frequencies: th.Tensor = None
    last_frequencies: th.Tensor = None

    vs_frequencies: th.Tensor = None
    vs_last_frequencies: th.Tensor = None
    update_factor: float = 0.9
    
    def __post_init__(self) -> None:
        if self.centroids is not None:
            assert self.deviations is not None
            assert self.deviations.shape == self.centroids.shape
            

    @property
    def n_used_contexts(self) -> int:
        return sum(self.last_frequencies > 0.0) if self.last_frequencies is not None else 0.0
    
    @property
    def n_used_valuesystems(self) -> int:
        return sum(self.vs_last_frequencies > 0.0) if self.vs_last_frequencies is not None else 0.0
    @property
    def ctx_spread_factor(self) -> int:
        return entropy(self.frequencies) if self.frequencies is not None else 0.0

    @property
    def vs_spread_factor(self) -> int:
            return entropy(self.vs_frequencies) if self.vs_frequencies is not None else 0.0
    
    def to(self, device)->Self:
        self.centroids = self.centroids.to(device)
        self.deviations = self.deviations.to(device)
        self.frequencies = self.frequencies.to(device)
        self.last_frequencies = self.last_frequencies.to(device)
        self.vs_frequencies = self.vs_frequencies.to(device)
        self.vs_last_frequencies = self.vs_last_frequencies.to(device)
        return self
    def to_dict(self) -> Dict:    
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    def cpu(self)->Self:
        return self.to("cpu")
    def update_running_average(self, metrics: Self):
        if self.centroids is None:
            self.update_full(metrics)
        else:
            self.centroids = self.centroids*self.update_factor + metrics.centroids*(1-self.update_factor)
            self.deviations= self.deviations*self.update_factor + metrics.deviations*(1-self.update_factor)
            self.last_frequencies = metrics.frequencies
            self.frequencies = self.frequencies*self.update_factor + metrics.frequencies*(1-self.update_factor)
            self.vs_last_frequencies = metrics.vs_frequencies
            self.vs_frequencies = self.vs_frequencies*self.update_factor + metrics.vs_frequencies*(1-self.update_factor)
    def update_full(self, metrics: Self):
        
        self.centroids = metrics.centroids
        self.deviations = metrics.deviations
        self.frequencies =  metrics.frequencies
        self.last_frequencies = metrics.last_frequencies
        self.vs_frequencies =  metrics.vs_frequencies
        self.vs_last_frequencies = metrics.vs_last_frequencies

class AbstractCtxDependentAlignmentLayer(AlignmentLayer):

    
    

    @abstractmethod
    def value_system_from_context_train(self, hidden_state, context_data: CtxData) -> Tuple[th.Tensor, CtxData]:
        return None
    
    @abstractmethod
    def value_system_from_context_eval(self, hidden_state, context_data: CtxData) -> Tuple[th.Tensor, CtxData]:
        return None

    @abstractmethod
    def calculate_statistics(self, hidden_state: th.Tensor) -> CtxStatistics:
        pass
    @abstractmethod
    def calculate_ctx_data(self, hidden_state: th.Tensor) -> CtxData:
        pass

    
    @abstractmethod
    def get_context_ids_mapped_to_value_system(self, index_context: int) -> List[int]:
        pass 
    @abstractmethod
    def get_value_systems(self) -> Tuple[Iterable[Any], Iterable[th.Tensor], Iterable[int]]:
        pass
    
    def value_system_from_context(self, hidden_state) -> Tuple[th.Tensor, CtxData]:
        assert hidden_state.shape[1] == self.input_shape
        #print("TRAINING??", self.training)
        if self.training:
            _context_stats_before = self.running_context_training_data
            context_data = self.calculate_ctx_data(hidden_state)
        else:
            _context_stats_before = self.running_context_validation_data
            with th.no_grad():
                context_data = self.calculate_ctx_data(hidden_state)
            
        context_stats = self.calculate_statistics(context_data, _context_stats_before)
        if self.training:
            with th.no_grad(): 
                self.running_context_training_data.update_running_average(context_stats)
            
            return self.value_system_from_context_train(hidden_state, context_data)
        else:
            with th.no_grad(): 
                self.running_context_validation_data.update_running_average(context_stats)
            
            return self.value_system_from_context_eval(hidden_state, context_data)
        

    def forward(self, grounding: th.Tensor, hidden_state: th.Tensor, *args, **kwargs) -> Tuple[th.Tensor, Dict]:
        vs_weights, ctx_data = self.value_system_from_context(hidden_state)
        if __debug__:
            if ctx_data.vs_assignments is not None:
                assignments = ctx_data.vs_assignments.detach().flatten()
                unique_assignments, counts = th.unique(assignments, sorted=True, return_counts=True)
                total = assignments.numel()
                summary = []
                for assignment_id, count in zip(unique_assignments.tolist(), counts.tolist()):
                    assignment_mask = assignments == assignment_id
                    assigned_weights = vs_weights[assignment_mask]
                    summary.append(
                        {
                            "vs_assignment": int(assignment_id),
                            "proportion": float(count) / float(total),
                            "weights": assigned_weights.mean(dim=0).detach().cpu().tolist()
                            if assigned_weights.numel() > 0
                            else [],
                        }
                    )
                print("OUTPUT", summary)
            else:
                print("OUTPUT", vs_weights, ctx_data.vs_assignments)
            print("VS_LOGPROBS", ctx_data.vs_logprobs.mean(dim=0))
            print("CTX_LOGPROBS", ctx_data.context_logprobs.mean(dim=0))
        assert vs_weights.shape[0] == grounding.shape[0]
        return (grounding * vs_weights).sum(dim=1, keepdim=True), {"ctx": ctx_data.to_dict()}

    def get_value_system_info(self) -> Dict[str, Dict]:
        value_system_ids, value_systems = self.get_value_systems()
        per_vs_contexts = {

        }
        for vi, vc in zip(value_system_ids, value_systems):
            contexts_to_vc = self.get_context_ids_mapped_to_value_system(vi)
            vs_tuple = transform_weights_to_tuple(vc)
            data={
                "contexts": contexts_to_vc,
                "share_of_ctxdata": float(th.sum(self.running_context_training_data.frequencies[contexts_to_vc]).cpu().numpy()) if self.running_context_training_data.frequencies is not None else "UNK",
                "share_of_vsdata": float(th.sum(self.running_context_training_data.vs_frequencies[contexts_to_vc]).cpu().numpy()) if self.running_context_training_data.vs_frequencies is not None else "UNK"}

            for iv, v in enumerate(vs_tuple):
                data[f"vs_w{iv}"] = v
            per_vs_contexts[f"vs_{vi}"] = data
        per_vs_contexts["used_ctxs"] = self.running_context_training_data.n_used_contexts
        per_vs_contexts["used_vs"] = self.running_context_training_data.n_used_valuesystems
        per_vs_contexts["spread_ctx"] = self.running_context_training_data.ctx_spread_factor
        per_vs_contexts["spread_vs"] = self.running_context_training_data.vs_spread_factor
        return per_vs_contexts
    
    @abstractmethod
    def value_system_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()
    @abstractmethod
    def context_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()
    

    def __init__(self, *args: Any, input_shape: int | Tuple, num_contexts: int, num_value_systems: int, num_values: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_values = num_values
        self.num_contexts = num_contexts
        self.num_value_systems = num_value_systems
        self.running_context_validation_data = CtxStatistics()
        self.running_context_training_data = CtxStatistics()

        if type(input_shape)==int:
            self.input_shape = input_shape
        else:
            raise NotImplementedError("2d (or more) input context shapes not implemeted")



class BasicCtxDependentAlignmentLayer(AbstractCtxDependentAlignmentLayer):

    def _construct_context_logprobabilities(self, input_shape: int | Tuple, ctx_hidden_sizes: list[int] = [], ctx_intermediate_activation: str = "ReLU", dropout = 0.0, device: th.device = None, dtype: th.dtype = None):
        return nn.Sequential(*construct_layers(
            input_dim=self.input_shape,
            hidden_sizes=ctx_hidden_sizes,
            intermediate_activation=ctx_intermediate_activation,
            final_activation="none",
            final_activation_kwargs={},
            n_outputs=self.num_contexts,
            dropout=dropout,
            device=device,
            dtype=dtype,
        ))
    def __init__(self, *args: Any, input_shape: int | Tuple, num_contexts: int, num_value_systems: int, num_values: int, ctx_hidden_sizes: list[int], ctx_intermediate_activation: str = "ReLU", detach_vs_selection_for_value_system_weight_training: bool = False, dropout=0, device: th.device = None, dtype: th.dtype = None, weight_initialization: str = "dirichlet", **kwargs: Any) -> None:
        super().__init__(*args, input_shape=input_shape, num_contexts=num_contexts, num_value_systems=num_value_systems, num_values=num_values,  **kwargs)
        
        
        self.weight_initialization = weight_initialization
        self.context_logprobabilities = self._construct_context_logprobabilities(input_shape, ctx_hidden_sizes, ctx_intermediate_activation, dropout, device, dtype)
        #print("CTX PARAMS", [p.dtype for p in self.context_logprobabilities.parameters()])
        #exit(0)
        self.log_softmaxvs = th.nn.LogSoftmax(dim=1)
        

        # Dirichlet concentration (uniform prior; tune if needed)
        
        if device is not None and th.device(device).type == "meta":
            logweights = th.empty((num_value_systems, num_values), device=device, dtype=dtype)
        else:
            if self.weight_initialization == "dirichlet":
                alpha = th.ones(num_values, device=device, dtype=dtype)
                # Create distribution
                dirichlet = th.distributions.Dirichlet(alpha)
                # Sample: shape (num_contexts, num_values)
                weights = dirichlet.sample((num_value_systems,))  # rows sum to 1
            elif self.weight_initialization == "equal":
                weights = th.ones((num_value_systems, num_values), device=device, dtype=dtype)/num_values
            elif self.weight_initialization == "span":
                weights = sample_example_profiles_scipy(profile_variety=num_value_systems, n_values=num_values)
                weights = th.tensor(np.asarray(weights), dtype=th.float32)
            #weights = sample_example_profiles_scipy(profile_variety=num_value_systems, n_values=num_values)
            #weights = th.tensor(np.asarray(weights), dtype=th.float32)
            
            # weights = th.ones((num_contexts, num_values), device=device, dtype=dtype)/num_values
            
            logweights = th.log(weights)
            
            # Register as parameter
        self.vs_selection_to_logweights_matrix = nn.Parameter(    logweights,    requires_grad=True,)
        self.detach_vs_selection_for_value_system_weight_training = detach_vs_selection_for_value_system_weight_training
        
        #/(self.num_contexts*self.num_value_systems)
        """self.context_to_vs_logprobabilities = nn.Parameter(
            th.rand(self.num_contexts, self.num_value_systems, device=device, dtype=dtype),
            requires_grad=True,
        )"""
    

    def value_system_parameters(self) -> Iterable[nn.Parameter]:
        return (self.vs_selection_to_logweights_matrix,)
    def context_parameters(self) -> Iterable[nn.Parameter]:
        return self.context_logprobabilities.parameters()
    
    def get_value_systems(self) -> Tuple[Iterable[Any], Iterable[th.Tensor]]:
        value_systems = th.nn.functional.softmax(self.vs_selection_to_logweights_matrix, dim=1)
        vs_indices = list(range(self.num_value_systems))
        return vs_indices, value_systems

    def get_context_ids_mapped_to_value_system(self, index_context: int) -> Iterable[Any]:
        return [index_context,]
    

    def calculate_ctx_data(self, hidden_state: th.Tensor) -> Dict:
        #print("CONSTRUCT DTYPE", list(self.context_logprobabilities.parameters())[0].dtype, "HS", hidden_state.dtype)
        #print("CTX PARAMS FORWARD", [p.dtype for p in self.context_logprobabilities.parameters()])
        
        context_logprobs = self.context_logprobabilities(hidden_state)
        
        #raise ValueError("...")
        #print("WHAT", context_logprobs.dtype)
        vs_assignments = th.argmax(context_logprobs, dim=1)
        
        assert vs_assignments.shape == (len(hidden_state),)
        
        return CtxData(
            context_features=hidden_state,
            vs_assignments=vs_assignments,
            ctx_assignments=vs_assignments,
            context_logprobs=context_logprobs,
            vs_logprobs=context_logprobs,
            vs_possibilities=self.vs_selection_to_logweights_matrix,
            ctx_possibilities=self.vs_selection_to_logweights_matrix
        )
        

    def calculate_statistics(self, context_data: CtxData, statistics_before: CtxStatistics= None, update_factor=0.9) -> CtxStatistics:
        with th.no_grad():
            values, counts = th.unique(context_data.ctx_assignments, return_counts=True)
            freqs = counts/len(context_data.ctx_assignments)
            #print("F??", freqs, len(context_data.context_features))
            value_system_indices_all = list(range(self.num_value_systems))
            ctx_indices_all = list(range(self.num_contexts))
            vs_values, vs_counts = th.unique(context_data.vs_assignments, return_counts=True)
            vs_freqs = vs_counts/len(context_data.vs_assignments)

                        
            freqs_all = th.zeros((len(ctx_indices_all),),dtype=th.float32).to(context_data.context_features.device)
            freqs_all[values] = freqs
            vs_freqs_all = th.zeros((len(value_system_indices_all),),dtype=th.float32).to(context_data.context_features.device)
            vs_freqs_all[vs_values] = vs_freqs
            centroids_new = th.stack([th.mean(context_data.context_features[context_data.ctx_assignments==vi],dim=0) for vi in values] )
            deviations_new = th.stack([th.std(context_data.context_features[context_data.ctx_assignments==vi], dim=0) for vi in values] )

            #exit()
            if statistics_before is not None and statistics_before.centroids is not None:
                centroids = deepcopy(statistics_before.centroids)
                deviations = deepcopy(statistics_before.deviations)
            else:
                centroids = th.zeros((self.num_contexts, *(centroids_new[0].shape)), dtype=centroids_new.dtype, device=centroids_new.device)
                deviations = th.zeros((self.num_contexts, *(deviations_new[0].shape)), dtype=deviations_new.dtype, device=deviations_new.device)
            
            centroids[values] = centroids_new
            deviations[values] = deviations_new
            

            return CtxStatistics(
                centroids=centroids.detach(),
                deviations=deviations.detach(),
                frequencies=freqs_all.detach(),
                last_frequencies=freqs_all.detach(),
                vs_frequencies=vs_freqs_all.detach(),
                vs_last_frequencies=vs_freqs_all.detach(),
                update_factor=update_factor
            )

    def value_system_from_context_train(self, hidden_state, context_data: CtxData) -> Tuple[th.Tensor, CtxData]:
        #print("HS DTYPE", hidden_state.dtype)
        #print("DEVICE h", hidden_state.device, "ctx", context_data.context_features.device, "vs", self.context_to_vslogweights_matrix.device)
        #print("Device ", context_data.context_logprobs.device)
        if self.detach_vs_selection_for_value_system_weight_training:
            vs_logprobs = context_data.vs_logprobs.detach()
        else:
            vs_logprobs = context_data.vs_logprobs
            
        
        log_value_systems = self.vs_selection_to_logweights_matrix
        ls_vs_probs = self.log_softmaxvs(vs_logprobs)
        
        ls_vs_weights = self.log_softmaxvs(log_value_systems)
        assert ls_vs_probs.shape==(hidden_state.shape[0], self.num_value_systems)
        assert ls_vs_weights.shape==(self.num_value_systems, self.num_values)

        

        #print(ls_ctx_probs[:,0].shape)
        #print(ls_ctx_probs[:,0].repeat(self.num_values,1).shape)
        #print(ls_vs_weights[0,:].shape)
        """combination = th.stack([
            th.exp(ls_ctx_probs[:,i].repeat(self.num_values,1) + ls_vs_weights[i,:].unsqueeze(1))
         for i in range(self.num_value_systems)]).sum(dim=0).T
""" 
        #print("LS CTX", ls_ctx_probs.device, "LS VS", ls_vs_weights.device)
        vs_predicted = th.exp(ls_vs_probs.T.unsqueeze(2) +   # (num_value_systems, batch, 1)   
                                ls_vs_weights.unsqueeze(1)      # (num_value_systems, 1, num_values)
                                ).sum(dim=0)                       # (batch, num_values)
        #print("vs pred", vs_predicted.device)
        assert vs_predicted.shape == (hidden_state.shape[0], self.num_values)
        assert th.allclose(th.sum(vs_predicted, dim=1), th.ones((hidden_state.shape[0],)).to(vs_predicted.device), atol=1e-5, rtol=0.01)
        
        if __debug__:
            with th.no_grad():
                vs_per_prob_index = th.softmax(log_value_systems, dim=1)
                #print("TYPE PREDICTIONS???? 1", vs_per_prob_index.dtype)
                ctx_probs = th.softmax(vs_logprobs, dim=1)
                #print("TYPE PREDICTIONS???? 2", log_ctx_probs.dtype)
                #print("TYPE PREDICTIONS???? 3", ctx_probs.dtype)
                

                th.testing.assert_close(vs_per_prob_index, th.exp(ls_vs_weights), atol=1e-4,rtol=0.03)
                th.testing.assert_close(ctx_probs, th.exp(ls_vs_probs), atol=1e-4,rtol=0.03)

                should_be = th.softmax(vs_logprobs, dim=1) @ vs_per_prob_index
                #print("TYPE PREDICTIONS???? 4", th.softmax(log_ctx_probs, dim=1).dtype)
                #print("TYPE PREDICTIONS???? 5", should_be.dtype)
                
                
                th.testing.assert_close(vs_predicted, should_be.to(dtype=vs_predicted.dtype), atol=1e-4,rtol=0.03)
                th.testing.assert_close(th.sum(vs_predicted, dim=1), th.ones(hidden_state.shape[0], dtype=vs_predicted.dtype).to(hidden_state.device), atol=1e-4, rtol=0.03)
        enriched_data = CtxData.from_previous(context_data, vs_predicted=vs_predicted, ctx_predicted = context_data.ctx_assignments)
        
        return vs_predicted, enriched_data
    
    def value_system_from_context_eval(self, hidden_state, context_data: CtxData) -> Tuple[th.Tensor, CtxData]:
        
        vs_assignments = context_data.vs_assignments
        vs_predicted = th.softmax(self.vs_selection_to_logweights_matrix[vs_assignments], dim=1)

        if __debug__:
            with th.no_grad():
                assert vs_predicted.shape == (hidden_state.shape[0],self.num_values)
                
                th.testing.assert_close(th.sum(vs_predicted, dim=1), th.ones((vs_predicted.shape[0],)))
        enriched_data = CtxData.from_previous(context_data, vs_predicted=vs_predicted, ctx_predicted = context_data.ctx_assignments)
        return vs_predicted, enriched_data
    
    
class BasicSmoothCtxDependentAlignmentLayer(BasicCtxDependentAlignmentLayer):
    def value_system_from_context_eval(self, hidden_state, context_data: CtxData) -> Tuple[th.Tensor, CtxData]:
        return self.value_system_from_context_train(hidden_state, context_data)


class BasicHarshCtxDependentAlignmentLayer(BasicCtxDependentAlignmentLayer):

    def context_parameters(self) -> Iterable[nn.Parameter]:
        return [ ]#.extend([*super().context_parameters()])
    def value_system_from_context_train(self, hidden_state, context_data: CtxData) -> Tuple[th.Tensor, CtxData]:
        context_logprobs = self.context_logprobabilities(hidden_state)
        vs_assignments = th.argmax(context_logprobs, dim=1)
        vs_predicted = th.softmax(self.vs_selection_to_logweights_matrix[vs_assignments], dim=1)
        #vs_predicted = self.context_to_vslogweights_matrix[vs_assignments]
        return vs_predicted, context_data
class DirectVSCtxDependentAlignmentLayer(BasicCtxDependentAlignmentLayer):

    def __init__(self, *args: Any, input_shape: int | Tuple, num_values: int, ctx_hidden_sizes: List[int], ctx_intermediate_activation: str = "ReLU", dropout=0, device: th.device = None, dtype: th.dtype = None, **kwargs: Any) -> None:
        AbstractCtxDependentAlignmentLayer.__init__(self, *args, input_shape=input_shape, num_contexts=1, num_value_systems=1, num_values=num_values,  **kwargs)
        self.log_vs_prediction = nn.Sequential(*construct_layers(
            input_dim=self.input_shape,
            hidden_sizes=ctx_hidden_sizes,
            intermediate_activation=ctx_intermediate_activation,
            final_activation="none",
            final_activation_kwargs={},
            n_outputs=self.num_values,
            dropout=dropout,
            device=device,
            dtype=dtype,
        ))
        self.softmaxctx = th.nn.Softmax(dim=1)
        self._last_vs_pred = th.ones((self.num_values,))/self.num_values

    def value_system_from_context_train(self, hidden_state, context_data: CtxData) -> Tuple[th.Tensor, CtxData]:
        return context_data.vs_predicted, context_data
    def value_system_from_context_eval(self, hidden_state, context_data: CtxData) -> Tuple[th.Tensor, CtxData]:
        return self.value_system_from_context_train(hidden_state, context_data)
    def calculate_statistics(self, context_data: CtxData, statistics_before: CtxStatistics= None, update_factor=0.9) -> CtxStatistics:
        return CtxStatistics(
            centroids = th.stack([th.mean(context_data.context_features, dim=0),]),
            deviations= th.stack([th.std(context_data.context_features, dim=0),]),
            frequencies=th.tensor([1.0,]),
            vs_frequencies=th.tensor([1.0,]),
            update_factor=update_factor,
        )
    def value_system_parameters(self) -> Iterable[nn.Parameter]:
        return self.log_vs_prediction.parameters()
    def context_parameters(self) -> Iterable[nn.Parameter]:
        return self.value_system_parameters()
    
    def get_value_systems(self) -> Tuple[Iterable[Any], Iterable[th.Tensor]]:
        return [0,], [self._last_vs_pred,]

    def get_context_ids_mapped_to_value_system(self, index_context: int) -> Iterable[Any]:
        return [0,]
    

    def calculate_ctx_data(self, hidden_state: th.Tensor) -> Dict:
        log_vs_pred = self.log_vs_prediction(hidden_state)
        vs_pred = self.softmaxctx(log_vs_pred)
        
        self._last_vs_pred = th.mean(vs_pred, dim=0)
        assert self._last_vs_pred.shape == (self.num_values,)
        #th.testing.assert_close(th.sum(vs_pred, dim=1), th.ones((len(vs_pred)), dtype=th.float32))
        return CtxData(
            context_features=hidden_state,
            vs_predicted=vs_pred,
            vs_assignments=th.range(0, hidden_state.shape[0],),
            ctx_assignments=th.range(0, hidden_state.shape[0],),
            context_logprobs=None,
            vs_logprobs=None

        )

class FastGaussianMixture(nn.Module):

    def __init__(
        self,
        input_size: int,
        num_components: int,
        l_entropy = 1e-1,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.l_entropy = l_entropy
        self.input_size = input_size
        self.num_components = num_components

        self.logits = nn.Parameter(
            th.zeros(num_components, device=device, dtype=dtype)
        )

        self.centroids = nn.Parameter(
            th.randn(
                num_components,
                input_size,
                device=device,
                dtype=dtype
            ) #* 1e-3
        )

        self.log_var = nn.Parameter(
            th.zeros(
                num_components,
                input_size,
                device=device,
                dtype=dtype
            )
        )


    def forward_all(self, x):

        # x: [B,D]

        diff = (
            x[:, None, :]
            -
            self.centroids[None, :, :]
        )
        # [B,K,D]


        inv_var = th.exp(
            -self.log_var
        )


        mahalanobis = (
            diff * diff * inv_var
        ).sum(-1)
        # [B,K]


        log_det = self.log_var.sum(-1)
        # [K]


        norm = (
            self.input_size *
            th.log(
                th.tensor(
                    2 * th.pi,
                    device=x.device,
                    dtype=x.dtype
                )
            )
        )


        component_log_prob = (
            -0.5 *
            (
                mahalanobis
                +
                log_det
                +
                norm
            )
        )
        logits = th.log_softmax(self.logits, dim=0)

        return th.logsumexp(
            component_log_prob
            +
            logits,
            dim=1
        ), component_log_prob, logits

    def forward(self, x: th.Tensor):
        point_logprob, per_component_logprob, component_logprobs = self.forward_all(x)
        #per_component_logprob: [B, K]
        #component_logprobs: [K]
        #assert th.testing.assert_close(th.sum(ind_probs, dim=1) , th.ones((x.shape[0],), device=x.device, dtype=x.dtype), rtol=1e-3, atol=1e-3)

        return per_component_logprob + component_logprobs
        #return per_component_logprob + component_logprobs

    def component_generation_logprob(self, batch: th.Tensor) -> th.Tensor:
        """Compute log-probabilities for each sample under each GMM component.

        Returns a tensor with shape [B, K] where each entry is:
            log p(x_b | component_k) + log p(component_k)
        """

        if batch.ndim != 2 or batch.shape[1] != self.input_size:
            raise ValueError(
                f"Expected batch shape [B, {self.input_size}], got {tuple(batch.shape)}"
            )

        centered = batch[:, None, :] - self.centroids[None, :, :]
        variance = th.exp(self.log_var)
        inv_variance = 1.0 / variance

        quadratic_term = (centered.pow(2) * inv_variance).sum(dim=-1)
        log_det_term = self.log_var.sum(dim=-1)
        normalizer = self.input_size * th.log(
            th.tensor(2.0 * th.pi, device=batch.device, dtype=batch.dtype)
        )

        conditional_logprob = -0.5 * (quadratic_term + log_det_term + normalizer)
        mixture_logprob = th.log_softmax(self.logits, dim=0)

        return conditional_logprob + mixture_logprob.unsqueeze(0)
        
    def set_centroids(self, centroids: th.Tensor):

        with th.no_grad():

            centroids = centroids.to(
                device=self.centroids.device,
                dtype=self.centroids.dtype
            )

            if centroids.shape != self.centroids.shape:
                raise ValueError(
                    f"Expected {self.centroids.shape}, got {centroids.shape}"
                )

            self.centroids.copy_(centroids)

    def initialize_from_data(
            self,
            centroids: th.Tensor,
            data: th.Tensor,
            assignments: th.Tensor,
            min_var: float = 1e-8,
        ):
            """Initialize diagonal-GMM parameters from hard cluster assignments.
    
            Args:
                centroids: [K, D] centroids to copy into the model.
                data: [B, D] data points.
                assignments: [B] hard cluster ids in [0, K-1].
                min_var: diagonal variance floor for stability.
            """
    
            with th.no_grad():
                self.set_centroids(centroids)
    
                data = data.to(device=self.centroids.device, dtype=self.centroids.dtype)
                assignments = assignments.to(device=self.centroids.device, dtype=th.long)
    
                if data.ndim != 2 or data.shape[1] != self.input_size:
                    raise ValueError(
                        f"Expected data shape [B, {self.input_size}], got {tuple(data.shape)}"
                    )
                if assignments.ndim != 1 or assignments.shape[0] != data.shape[0]:
                    raise ValueError("assignments must be [B] with the same B as data")
    
                total_points = data.shape[0]
                if total_points == 0:
                    raise ValueError("Cannot initialize from empty data")
    
                counts = th.bincount(assignments, minlength=self.num_components).to(self.centroids.dtype)
                probs = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-12)
                self.logits.copy_(th.log(probs))
    
                per_centroid_var = th.empty(
                    self.num_components,
                    self.input_size,
                    device=self.centroids.device,
                    dtype=self.centroids.dtype,
                )
    
                fallback_var = th.full(
                    (self.input_size,),
                    min_var,
                    device=self.centroids.device,
                    dtype=self.centroids.dtype,
                )
    
                for k in range(self.num_components):
                    mask = assignments == k
                    n_k = int(mask.sum().item())
    
                    if n_k > 1:
                        points_k = data[mask]
                        centered = points_k - self.centroids[k]
                        var_k = centered.pow(2).mean(dim=0).clamp_min(min_var)
                        per_centroid_var[k] = var_k
                    else:
                        per_centroid_var[k] = fallback_var
    
                self.log_var.copy_(th.log(per_centroid_var))

    def sample(self, n):

        with th.no_grad():

            ids = th.multinomial(
                th.softmax(self.logits, dim=0),
                n,
                replacement=True
            )


            std = th.exp(
                0.5 *
                self.log_var[ids]
            )


            return (
                self.centroids[ids]
                +
                th.randn_like(std) * std,
                ids
            )
    def sample_with_predicted_cluster(self, n: int):

        with th.no_grad():

            mixture_probs = th.softmax(self.logits, dim=0)

            # Generate samples
            ids = th.multinomial(
                mixture_probs,
                n,
                replacement=True
            )

            std = th.exp(0.5 * self.log_var[ids])

            samples = (
                self.centroids[ids]
                + th.randn_like(std) * std
            )

            # Compute log p(x | k) for every component
            diff = samples[:, None, :] - self.centroids[None, :, :]
            inv_var = th.exp(-self.log_var)

            mahalanobis = (diff.pow(2) * inv_var).sum(dim=-1)
            log_det = self.log_var.sum(dim=-1)
            norm = self.input_size * th.log(
                th.tensor(2 * th.pi, device=samples.device, dtype=samples.dtype)
            )

            component_log_prob = -0.5 * (
                mahalanobis + log_det + norm
            )

            # Add log mixture weights
            log_post: th.Tensor = component_log_prob + th.log_softmax(self.logits, dim=0)

            # MAP component
            predicted_ids = log_post.argmax(dim=-1)
            log_prob, _ = th.max(log_post, dim=-1)

            return samples, log_prob, predicted_ids
class BasicGmmCtxDependentAlignmentLayer(BasicCtxDependentAlignmentLayer):

    def __init__(self, input_shape: int | Tuple, num_contexts: int, num_value_systems: int, num_values: int, ctx_hidden_sizes: list[int], ctx_intermediate_activation: str = "ReLU", dropout=0, device: th.device = None, dtype: th.dtype = None, detach_vs_selection_for_value_system_weight_training: bool = False, detach_context_selection_for_value_system_selection: bool = False, weight_initialization: str = "dirichlet", **kwargs: Any) -> None:
        
        super().__init__(input_shape=input_shape, num_contexts=num_contexts, num_value_systems=num_value_systems, num_values=num_values, ctx_hidden_sizes=ctx_hidden_sizes, ctx_intermediate_activation=ctx_intermediate_activation, dropout=dropout, dtype=dtype, 
                         detach_vs_selection_for_value_system_weight_training=detach_vs_selection_for_value_system_weight_training,  weight_initialization=weight_initialization, **kwargs)
        self.detach_context_selection_for_value_system_selection=detach_context_selection_for_value_system_selection
        self.context_to_vs_logprobabilities = nn.Parameter(
                            th.randn(
                                (num_contexts,num_value_systems),
                                device=device,
                                dtype=dtype
                            )*2, requires_grad=True
                        )
            
    def initialize_gmm(self, centroids: th.Tensor, data: th.Tensor, assignments: th.Tensor):
        with th.no_grad():
            self.context_logprobabilities.initialize_from_data(centroids, data, assignments)
    def _construct_context_logprobabilities(self, input_shape: int | Tuple, ctx_hidden_sizes: list[int] = [], ctx_intermediate_activation: str = "ReLU", dropout = 0.0, device: th.device = None, dtype: th.dtype = None):
        return FastGaussianMixture(input_size=input_shape, num_components=self.num_contexts, device=device, dtype=dtype)

    def context_parameters(self) -> Iterable[nn.Parameter]:
        return (*self.context_logprobabilities.parameters(), self.context_to_vs_logprobabilities,)
        
    

    def calculate_ctx_data(self, hidden_state: th.Tensor) -> Dict:
            #print("CONSTRUCT DTYPE", list(self.context_logprobabilities.parameters())[0].dtype, "HS", hidden_state.dtype)
            #print("CTX PARAMS FORWARD", [p.dtype for p in self.context_logprobabilities.parameters()])
            self.context_logprobabilities: FastGaussianMixture
            assert isinstance(self.context_logprobabilities, FastGaussianMixture)
            gmm_logprob, per_component_logprob, component_logprobs = self.context_logprobabilities.forward_all(hidden_state)
            #per_component_logprob = per_component_logprob/th.max(th.abs(per_component_logprob))
            #SOFT: context_logprobs = th.log_softmax(per_component_logprob, dim=1) + component_logprobs
            #HARD: context_logprobs = per_component_logprob + component_logprobs
            context_logprobs = th.log_softmax(per_component_logprob + component_logprobs, dim=1)
            if self.detach_context_selection_for_value_system_selection:
                context_logprobs = context_logprobs.detach()
                
            assert context_logprobs.shape == (hidden_state.shape[0], self.num_contexts)
            assert hidden_state.shape[1] == self.context_logprobabilities.input_size
            assert hidden_state[0].norm() <= 1.0001
            #print("SHOULD BE", th.log(per_component_logprob.exp() * component_logprobs.exp()))
            #print("IT GOES:", context_logprobs)
            #assert th.allclose(th.log(per_component_logprob.exp() * component_logprobs.exp()), context_logprobs, atol=1e-3, rtol=0.03)
            
            ctx_assignments = th.argmax(context_logprobs, dim=1)
            #context_logprobs_with_default = th.cat([th.tensor((-th.sum(context_logprobs, dim=1)+1.0).unsqueeze(0), dtype=context_logprobs.dtype, device=context_logprobs.device), context_logprobs], dim=1 )
            #assert context_logprobs_with_default.shape == (context_logprobs.shape[0], context_logprobs.shape[1] +1)
            #context_logprobs = th.log_softmax(per_component_logprob, dim=1) + component_logprobs
            #raise ValueError("...")
            #print("WHAT", context_logprobs.dtype)
            assert context_logprobs.shape == (hidden_state.shape[0], self.num_contexts)

            # p(v|x) = sum_c p(v|c) p(c|x), computed stably in log-space.
            log_p_context_given_x = context_logprobs
            log_p_vs_given_context = th.log_softmax(self.context_to_vs_logprobabilities, dim=1)

            vs_logprobs = th.logsumexp(
                log_p_context_given_x.unsqueeze(-1) + log_p_vs_given_context.unsqueeze(0),
                dim=1,
            )

            if __debug__:
                with th.no_grad():
                    probs_aux = th.softmax(context_logprobs.detach(), dim=1) @ th.softmax(self.context_to_vs_logprobabilities, dim=1)
                    vs_logprobs_aux = th.log(probs_aux.clamp_min(1e-30))

                #print(vs_logprobs.shape)
                assert vs_logprobs.shape == (hidden_state.shape[0], self.num_value_systems)
                assert vs_logprobs_aux.shape == (hidden_state.shape[0], self.num_value_systems)

                
                assert th.allclose(vs_logprobs, vs_logprobs_aux, atol=1e-4, rtol=0.03), f"VS logprobs mismatch: {vs_logprobs[0:5]} vs {vs_logprobs_aux[0:5]}"
            vs_assignments = th.argmax(vs_logprobs, dim=1)
            
            assert vs_assignments.shape == (len(hidden_state),)
            
            return CtxData(
                context_features=hidden_state,
                vs_assignments=vs_assignments,
                ctx_assignments=ctx_assignments,
                context_logprobs=context_logprobs,
                vs_logprobs=vs_logprobs,
                vs_possibilities=self.vs_selection_to_logweights_matrix,
                ctx_possibilities=self.context_logprobabilities.centroids,
                extra_for_custom_loss=(gmm_logprob, per_component_logprob, component_logprobs)
                
            )
    

class MORMForClassificationConfig(PretrainedConfig):
    model_type = "morm_for_sequence_classification"
    has_no_defaults_at_init = True

    @property
    def loss_management(self) -> MOLossManagement:
        loss_func_enum = MOLossFunctions(self.loss_func_type)
        return MOLossManagement(loss_func_enum, self.loss_func_type_kwargs)

    def __init__(
        self,
        # This will be set properly in the model init based on the tokenizer
        context_implementation: str = ContextImplementations.NO_CONTEXT.value,
        vs_weight_initialization: Literal['dirichlet',
                                     'span', 'equal'] = "dirichlet",
        do_initialization: bool = True,
        ctx_coefficient: float = 0.0,
        vs_selection_coefficient: float = 0.0,
        training_initialization_data_size: int|str = "all",
        sharp_context_classification: bool = True,
        detach_context_selection_for_value_system_selection: bool = False,
        detach_vs_selection_for_value_system_weight_training: bool = False,
        pad_token_id: int = "UNKNOWN",
        num_values: int = 3,
        input_size: int = "infer", 
        input_size_vs: int = "infer",
        hidden_sizes: list[int] = [1024, 1024, 1024],
        vs_layer_hidden_sizes: list[int] = [],
        vs_layer_dropout: float = 0.0,
        vs_layer_intermediate_activation: str = "ReLU",
        value_layer_dropout: float = 0.1,
        value_layer_intermediate_activation: str = "ReLU",
        value_layer_final_activation: str = "none",
        layer_normalization: Literal['LayerNorm',
                                     'BatchNorm', 'none'] = 'LayerNorm',
                            
        reward_diff_threshold: float = 50.0,
        assume_qualitative_labels: bool = True,
        discordance_epsilon=MIN_EPSILON,
        activate_discordance_epsilon_for_loss: bool = False,
        check_undefined_label: bool = True,
        grounding_loss_tendency_update_ratio: float = 0.001,
        update_tendencies_every_n_steps: int = 1,
        use_validation_for_tendencies: bool = False,
        rew_center_coefficient: float = 0.0,
        gradient_accumulation_steps: int = 2,
        use_metrics_or_losses_for_lagrange_updates: str = "metrics",
        use_exponential_moving_average_or_optimum_targets: str = "optimum",
        grad_on_only_worst_value: bool = False,
        zero_constraint: bool = True,
        lambda_decay: float = 0.0,
        gather_train_metrics: bool = False,
        use_ideal_grounding_model: bool = False,
        dtype: str = "float32",
        base_model_name_or_path: Optional[str] = None,
        base_model_trust_remote_code: bool = True,
        base_model_num_labels: int = 1,
        use_base_model_heads: bool = False,
        base_model_reward_heads_module_name: str = None,
        base_model_value_system_module_name: str = None,
        base_model_reward_head_indices: list = None,
        loss_func_type: str = MOLossFunctions.DEFAULT.value,
        loss_func_kwargs: dict = None,
        lr_grounding: Optional[float] = None,
        lr_value_system: Optional[float] = None,
        lr_context: Optional[float] = None,
        lr_lambda: Optional[float] = None,
        max_contexts: Optional[float]=5,
        max_value_systems: Optional[float]=3,
        **kwargs,
    ):
        assert num_values > 0, "num_values must be greater than 0"
        # assert len(hidden_sizes) > 0, "hidden_sizes must be a non-empty list"

        if value_layer_intermediate_activation not in VALUE_LAYER_ACTIVATIONS.keys():
            raise ValueError(
                f"value_layer_intermediate_activation must be one of {list(VALUE_LAYER_ACTIVATIONS.keys())}, but got {value_layer_intermediate_activation}")
        if value_layer_final_activation not in VALUE_LAYER_ACTIVATIONS.keys():
            raise ValueError(
                f"value_layer_final_activation must be one of {list(VALUE_LAYER_ACTIVATIONS.keys())}, but got {value_layer_final_activation}")

        if layer_normalization not in ['LayerNorm', 'BatchNorm', 'none']:
            raise ValueError(
                f"layer_normalization must be one of 'LayerNorm', 'BatchNorm', 'none', but got {layer_normalization}")

        default_id2label = {
            index: f"VALUE_{index}" for index in range(num_values)
        }
        default_id2label[num_values] = "VALUE_SYSTEM"
        id2label = kwargs.pop("id2label", default_id2label)
        label2id = kwargs.pop(
            "label2id", {label: index for index, label in id2label.items()})

        """if pad_token_id == "UNKNOWN":
            raise ValueError("pad_token_id must be set to a valid integer value corresponding to the tokenizer's pad token ID. It is currently set to 'UNKNOWN', which is not valid. Please set it to the correct value when initializing the config.")
        """
        self.vs_layer_hidden_sizes = vs_layer_hidden_sizes
        self.vs_layer_dropout = vs_layer_dropout
        self.vs_layer_intermediate_activation = vs_layer_intermediate_activation
        self.vs_weight_initialization = vs_weight_initialization
        self.context_implementation = context_implementation
        self.sharp_context_classification = sharp_context_classification
        self.do_initialization = do_initialization
        self.vs_selection_coefficient = vs_selection_coefficient
        self.ctx_coefficient = ctx_coefficient
        self.max_contexts = max_contexts
        self.detach_context_selection_for_value_system_selection = detach_context_selection_for_value_system_selection
        self.detach_vs_selection_for_value_system_weight_training = detach_vs_selection_for_value_system_weight_training

        self.max_value_systems = max_value_systems
        self.pad_token_id = pad_token_id
        self.num_values = num_values
        self.gather_train_metrics = gather_train_metrics
        self.hidden_sizes = hidden_sizes
        self.value_layer_dropout = value_layer_dropout
        self.value_layer_intermediate_activation = value_layer_intermediate_activation
        self.activate_discordance_epsilon_for_loss = activate_discordance_epsilon_for_loss
        self.value_layer_final_activation = value_layer_final_activation
        self.reward_diff_threshold = reward_diff_threshold
        self.assume_qualitative_labels = assume_qualitative_labels
        self.check_undefined_label = check_undefined_label
        self.grounding_loss_tendency_update_ratio = grounding_loss_tendency_update_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_metrics_or_losses_for_lagrange_updates = use_metrics_or_losses_for_lagrange_updates
        self.use_exponential_moving_average_or_optimum_targets = use_exponential_moving_average_or_optimum_targets
        self.grad_on_only_worst_value = grad_on_only_worst_value
        self.update_tendencies_every_n_steps = update_tendencies_every_n_steps
        self.use_validation_for_tendencies = use_validation_for_tendencies
        self.zero_constraint = zero_constraint
        self.rew_center_coefficient = rew_center_coefficient
        self.discordance_epsilon = discordance_epsilon
        self.input_size = input_size
        self.input_size_vs = input_size_vs
        self.training_initialization_data_size = training_initialization_data_size
        if isinstance(dtype, th.dtype):
            self.dtype = str(dtype).replace("torch.", "")
        else:
            self.dtype = str(dtype)
        self.use_ideal_grounding_model = use_ideal_grounding_model
        self.layer_normalization = layer_normalization
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_trust_remote_code = bool(base_model_trust_remote_code)
        self.base_model_num_labels = int(base_model_num_labels)
        self.use_base_model_heads = use_base_model_heads
        self.base_model_reward_heads_module_name = base_model_reward_heads_module_name
        self.base_model_value_system_module_name = base_model_value_system_module_name
        # Store loss_func_type as string value for JSON serialization compatibility
        if isinstance(loss_func_type, MOLossFunctions):
            self.loss_func_type = loss_func_type.value
        else:
            self.loss_func_type = loss_func_type
        self.loss_func_type_kwargs = loss_func_kwargs if loss_func_kwargs is not None else {}

        loss_manage = self.loss_management # requires self. loss_functype and loss_functypekwargs.

        if not loss_manage.should_apply_grad_on_grounding_parameters():
            self.lr_grounding = 0.0
        else:
            assert lr_grounding is not None and lr_grounding > 0.0, f"Loss function type {loss_func_type} requires applying gradients on grounding parameters, but lr_grounding is set to {lr_grounding}. Please set lr_grounding to a positive value to enable optimization of grounding parameters."
            self.lr_grounding = lr_grounding

        if not loss_manage.should_apply_grad_on_value_system_weights():
            self.lr_value_system = 0.0
        else:
            assert lr_value_system is not None and lr_value_system > 0.0, f"Loss function type {loss_func_type} requires applying gradients on value system parameters, but lr_value_system is set to {lr_value_system}. Please set lr_value_system to a positive value to enable optimization of value system parameters."
            self.lr_value_system = lr_value_system
        
        if not loss_manage.should_apply_grad_on_context_parameters():
            self.lr_context = 0.0
        else:
            assert lr_context is not None and lr_context > 0.0, f"Loss function type {loss_func_type} requires applying gradients on context parameters, but lr_context is set to {lr_context}. Please set lr_context to a positive value to enable optimization of value system parameters."
            self.lr_context = lr_context
        

        self.lambda_decay = lambda_decay
        if not loss_manage.should_apply_grad_on_lagrange_multipliers():
            self.lr_lambda = 0.0
            self.lambda_decay = 0.0
        else:
            if lr_lambda is None:
                lr_lambda = lr_value_system
            assert lr_lambda is not None and lr_lambda > 0.0, f"Loss function type {loss_func_type} requires applying gradients on Lagrange multipliers, but lr_lambda is set to {lr_lambda}. Please set lr_lambda to a positive value to enable optimization of Lagrange multipliers."
            self.lr_lambda = lr_lambda

        self.base_model_reward_head_indices = base_model_reward_head_indices if base_model_reward_head_indices is not None else "use_base_model_value_system_module_name"

        
        super().__init__(num_labels=num_values + 1,
                         id2label=id2label, label2id=label2id, **kwargs)




LossFuncType = Callable[[th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor],
                         MORMForClassificationConfig, MORMTrainingVariables], th.Tensor]


def parse_loss_function(config: MORMForClassificationConfig) -> LossFuncType:
    return mo_loss_function

def apply_discordance_epsilon_to_logits(missing_mask: th.Tensor, targets, bt: th.Tensor, activate_discordance_epsilon_for_loss: bool = True, discordance_epsilon: Optional[float] = 0.0):
        if activate_discordance_epsilon_for_loss and discordance_epsilon is not None:
                with th.no_grad():
                    discordance = th.full_like(bt, fill_value=0.0)
                    discordance = discordance.masked_fill(targets < 0.5, -discordance_epsilon)
                    discordance = discordance.masked_fill(targets > 0.5, discordance_epsilon)
                    if missing_mask is not None:
                        discordance = discordance.masked_fill(missing_mask, 0.0)
                return bt-discordance
        else:
            return bt
        
def accuracy_rewards_labels(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, threshold=50.0, assume_qualitative_labels=False, check_undefined_label=True, missing_mask=None, assume_torch=True, discordance_epsilon=MIN_EPSILON) -> th.Tensor:

    logits, targets, others = reward_pairs_and_scores_to_logits_and_targets(reward1, reward2, scores1, scores2, reward_diff_threshold=threshold,
                                                                            assume_qualitative_labels=assume_qualitative_labels, check_undefined_label=check_undefined_label, 
                                                                            assume_torch=assume_torch)
    return accuracy_logits(logits, targets, missing_mask=others.get("missing_mask", missing_mask), assume_torch=assume_torch, discordance_epsilon=discordance_epsilon)


def accuracy_logits_smooth(logits: th.Tensor, target_probs: th.Tensor, missing_mask=None, assume_torch=True) -> th.Tensor:
    with th.no_grad():
        
        missing_mask = get_missing_rating_mask(
            target_probs) if missing_mask is None else missing_mask
        if missing_mask is not None:
            all_defined_cases = ~missing_mask
            logits_of_smoothing_equal_cases = logits[all_defined_cases]
            targets_of_smoothing_equal_cases = target_probs[all_defined_cases]
        else:
            all_defined_cases = True
            logits_of_smoothing_equal_cases = logits
            targets_of_smoothing_equal_cases = target_probs


        if assume_torch:
            f = th.nn.functional.sigmoid(logits_of_smoothing_equal_cases)
            score = 1.0-th.abs(f - targets_of_smoothing_equal_cases)
            # assert th.all(score <= 1.0) and th.all(score >= 0.0), f"Score values must be between 0 and 1.0, but got min {th.min(score)}, max {th.max(score)}"
        else:
            f = 1.0/(1.0+np.exp(-logits_of_smoothing_equal_cases))
            score = 1.0-np.abs(f - targets_of_smoothing_equal_cases)
            # assert np.all(score <= 1.0) and np.all(score >= 0.0), f"Score values must be between 0 and 1.0, but got min {np.min(score)}, max {np.max(score)}"

        if assume_torch:
            mask = all_defined_cases.float()
            factor = mask.sum(dim=0)
        else:
            mask = all_defined_cases.astype(float)
            factor = mask.sum(axis=0)
        mask[all_defined_cases] = score

        if assume_torch:
            positive_cases = mask.sum(dim=0)
        else:
            positive_cases = mask.sum(axis=0)

        return positive_cases/factor


def print_logits_target_mismatches(logits: th.Tensor, target_probs: th.Tensor, logits1: th.Tensor, target1: th.Tensor, all_defined_cases, assume_torch=True) -> None:
    """Debug helper: print where `logits1` differs from `target1` and show values.

    Kept as a standalone function so callers can enable/disable it easily.
    """
    try:
        if assume_torch:
            mismatch = (logits1 != target1) & all_defined_cases
            if mismatch.any():
                idx = mismatch.nonzero(as_tuple=False)
                if idx.numel() > 0:
                    if logits.ndim == 1:
                        rows = idx.squeeze(1)
                        vals_logits = logits[rows].cpu().numpy()
                        vals_target_probs = target_probs[rows].cpu().numpy()
                        vals_logits1 = logits1[rows].cpu().numpy()
                        vals_target1 = target1[rows].cpu().numpy()
                        print(f"accuracy_logits mismatch at indices: {rows.cpu().numpy()}")
                        print("logits:", vals_logits)
                        print("target_probs:", vals_target_probs)
                        print("logits1:", vals_logits1, "target1:", vals_target1)
                        input()
                    else:
                        rows = idx[:, 0]
                        cols = idx[:, 1]
                        vals_logits = logits[rows, cols].cpu().numpy()
                        vals_target_probs = target_probs[rows, cols].cpu().numpy()
                        vals_logits1 = logits1[rows, cols].cpu().numpy()
                        vals_target1 = target1[rows, cols].cpu().numpy()
                        print("accuracy_logits mismatches at (row,col):", list(zip(rows.cpu().numpy(), cols.cpu().numpy())))
                        print("logits:", vals_logits)
                        print("target_probs:", vals_target_probs)
                        print("logits1:", vals_logits1, "target1:", vals_target1)
                        input()
        else:
            # numpy branch
            mismatch = (logits1 != target1) & all_defined_cases
            if mismatch.any():
                idx = np.nonzero(mismatch)
                print("accuracy_logits mismatches at indices:", idx)
                print("logits values:", logits[idx])
                print("target_probs values:", target_probs[idx])
                print("logits1:", logits1[idx], "target1:", target1[idx])
                input()
    except Exception as e:
        # Do not raise from the debug helper; report and continue
        try:
            print("print_logits_target_mismatches failed:", e)
        except Exception:
            pass


def accuracy_logits(logits: th.Tensor, target_probs: th.Tensor, missing_mask=None, assume_torch=True, discordance_epsilon=MIN_EPSILON, hard_classification=True) -> th.Tensor:

    with th.no_grad():
        score_diff_epsilon = 1.0/(1.0+np.exp(-discordance_epsilon)) -0.5 if discordance_epsilon > 0 else 0.0
        discordance_epsilon = max(discordance_epsilon, 1e-5)

        missing_mask = get_missing_rating_mask(
            target_probs) if missing_mask is None else missing_mask
        if missing_mask is not None:
            all_defined_cases = ~missing_mask
        else:
            all_defined_cases = th.ones_like(target_probs, dtype=th.bool) if assume_torch else np.ones_like(target_probs, dtype=bool)
       
        # & (target_probs != NO_RATING_MASK))

        if hard_classification:
            logits1 = (logits > discordance_epsilon) 
            logits0 = (logits < -discordance_epsilon) 
            target1 = (target_probs > 0.5 + score_diff_epsilon) 
            target0 = (target_probs < 0.5 - score_diff_epsilon)

            mask1_1 = logits1 & target1
            mask1_2 = logits0 & target0
            
            equal_cases_logits = ~(logits1  | logits0) #if not hard_classification else (logits <= discordance_epsilon) & (logits >= -discordance_epsilon)
            equal_cases_targets = ~(target1  |target0) #if not hard_classification else (target_probs <= 0.5 + score_diff_epsilon) & (target_probs >= 0.5 - score_diff_epsilon)
            mask1_3 = equal_cases_logits & equal_cases_targets

            #print_logits_target_mismatches(logits, target_probs, equal_cases_logits, equal_cases_targets, all_defined_cases, assume_torch=assume_torch)
            mask05_1 = equal_cases_logits & ~equal_cases_targets
            mask05_2 = equal_cases_targets & ~equal_cases_logits

            mask05 = (mask05_1 | mask05_2) & all_defined_cases
            mask1 = (mask1_1 | mask1_2  | mask1_3) & all_defined_cases

        else:
            repr1 = (logits > 0) & (target_probs > 0.5)
            repr2 = (logits < 0) & (target_probs < 0.5)
            equal_targets = (target_probs >= 0.5 - score_diff_epsilon) & (target_probs <= 0.5 + score_diff_epsilon)
            equal_logits = (logits >= -discordance_epsilon) & (logits <= discordance_epsilon)
            repr3 = equal_targets & equal_logits
            mask1 = (repr1 | repr2 | repr3) & all_defined_cases
            mask05 = all_defined_cases & ((equal_targets & ~equal_logits) | (~equal_targets & equal_logits))

        if assume_torch:
            mask = mask1.float()
            mask[mask05 & ~mask1] = 0.5   
            factor = (all_defined_cases).float().sum(dim=0)
            positive_cases = mask.sum(dim=0)
        else:
            mask = mask1.astype(float)
            mask[mask05 & ~mask1] = 0.5   
            factor = (all_defined_cases).astype(float).sum(axis=0)
            positive_cases = mask.sum(axis=0)
        accuracy = positive_cases / factor

    if len(logits.shape) >= 2:
        assert accuracy.shape == (
            logits.shape[-1],), f"Expected loss shape {(logits.shape[-1],)}, got {accuracy.shape}"
    else:
        assert accuracy.shape == (
        ), f"Expected loss shape (), got {accuracy.shape}"
    return accuracy


def logits_BT(x: th.Tensor, y: th.Tensor, threshold=50.0, check_undefined_label=False, missing_mask=None, assume_torch=True) -> th.Tensor:
    
    returns_diff = x - y
    
    if check_undefined_label:
        if missing_mask is None:
            missing_mask = get_missing_rating_mask(x, y)
    else:
        if missing_mask is not None:
            raise ValueError(
                "check_undefined_label should be True to use missing_mask or get_missing_rating_mask")

    if assume_torch:
        returns_diff = th.clip(returns_diff, -threshold, threshold)
    else:
        returns_diff = np.clip(returns_diff, -threshold, threshold)
    if missing_mask is not None and check_undefined_label:
        if assume_torch:
            returns_diff.masked_fill_(missing_mask, NO_RATING_MASK)
        else:
            returns_diff[missing_mask] = NO_RATING_MASK
    """if assume_torch:
        assert th.max(returns_diff[~missing_mask]) <= threshold and th.min(returns_diff[~missing_mask]) >= - \
            threshold, f"Clipping failed: max {th.max(returns_diff[~missing_mask])}, min {th.min(returns_diff[~missing_mask])}, threshold {threshold}"
    """
    return returns_diff


def get_missing_rating_mask(x_or_probs, y=None):
    if y is not None:
        missing_mask = (x_or_probs == NO_RATING_MASK) | (y == NO_RATING_MASK)
    else:
        missing_mask = (x_or_probs == NO_RATING_MASK)

    return missing_mask


def rewards_and_labels_to_logits_and_targets(logits, labels=None, assume_torch=True, config: MORMForClassificationConfig = None):
    others_logits = None
    if isinstance(logits, tuple):
        assert config.context_implementation != ContextImplementations.NO_CONTEXT
        
        logits_, others_logits = logits[0], logits[1]
    else:
        logits_ = logits
    bsz = logits_.size(0)

    jidx = th.arange(0, bsz, 2, device=logits_.device)
    kidx = jidx + 1

    rewards_1 = logits_[jidx]
    rewards_2 = logits_[kidx]

    if labels is not None:
        labels_1 = labels[jidx]
        labels_2 = labels[kidx]
    else:
        labels_1 = None
        labels_2 = None

    logits_new, target_probs, others = reward_pairs_and_scores_to_logits_and_targets(
        rewards_1, rewards_2, labels_1, labels_2, 
        reward_diff_threshold=config.reward_diff_threshold, 
        assume_qualitative_labels=config.assume_qualitative_labels, 
        check_undefined_label=config.check_undefined_label, 
        assume_torch=assume_torch)
    if others_logits is not None:
        others.update(others_logits)
    return logits_new, target_probs, others

from vsllib.utils import entropy, kmeans_clustering, print_tensor_and_grad_fn, sample_example_profiles_exact, sample_example_profiles_scipy, transform_weights_to_tuple

def scores_to_target_probs(scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold: int = 50.0, assume_qualitative_labels=False, check_undefined_label=True, missing_mask=None, assume_torch=True) -> th.Tensor:

    with th.no_grad():

        if assume_qualitative_labels:
            # model probability of first one being preferred.
            mask_greater = scores1 > scores2
            mask_less = scores1 < scores2
            mask_equal = ~(mask_greater | mask_less)
            if assume_torch:
                target_probs = mask_greater.float()
            else:
                target_probs = mask_greater.astype(float)
            target_probs[mask_equal] = 0.5

            
        
        else:
            log = logits_BT(scores1, scores2, threshold=reward_diff_threshold,
                            check_undefined_label=check_undefined_label, missing_mask=missing_mask, assume_torch=assume_torch)
            if assume_torch:
                target_probs = th.sigmoid(log)
            else:
                target_probs = 1 / (1 + np.exp(-log))
        mask = True
        if check_undefined_label:
            # If either score is NO_RATING_MASK, set target_prob to 0.5 (indicating no preference)
            mask = get_missing_rating_mask(
                scores1, scores2) if missing_mask is None else missing_mask
            if assume_torch:
                target_probs.masked_fill_(mask, NO_RATING_MASK)
                #assert th.allclose(target_probs[mask_less], th.zeros_like(target_probs[mask_less]))
                #assert th.allclose(target_probs[mask_greater], th.ones_like(target_probs[mask_greater]))
                #assert th.allclose(target_probs[mask_equal], 0.5 * th.ones_like(target_probs[mask_equal]))
            else:
                target_probs[mask] = NO_RATING_MASK

    """if assume_qualitative_labels:
        assert np.allclose(target_probs[mask_less & mask], np.zeros_like(target_probs[mask_less & mask]))
        assert np.allclose(target_probs[mask_greater & mask], np.ones_like(target_probs[mask_greater & mask]))
        assert np.allclose(target_probs[mask_equal & mask], 0.5 * np.ones_like(target_probs[mask_equal & mask]))
    """
    return target_probs



def grounding_loss_logits(logits_p: th.Tensor, target_probs_p: th.Tensor, rew_sum: th.Tensor = None, return_metrics: bool = False, check_undefined_label=True, rew_center_coefficient=0.0, missing_mask: th.Tensor = None, no_grad_on_indexes: Optional[list[int]] = None, discordance_epsilon=MIN_EPSILON, activate_disc_epsilon_for_loss=False) -> th.Tensor:
    """Multi-objective Cross-entropy loss: target_probs(1,2)*log(exp(r1) / (exp(r1) + exp(r2)))- (1-target_probs(1,2))*log(exp(r2) / (exp(r1) + exp(r2)))"""

    missing_mask = get_missing_rating_mask(
        target_probs_p) if check_undefined_label and missing_mask is None else missing_mask

    if check_undefined_label:
        logits = logits_p.masked_fill(missing_mask, 0.0)
        target_probs = target_probs_p.masked_fill(missing_mask, 0.5)
    else:
        logits = logits_p
        target_probs = target_probs_p
    
    if __debug__:
        assert target_probs.shape == logits.shape, f"Target probabilities shape {target_probs.shape} does not match logits shape {logits.shape}"
        assert not th.any(logits.isnan()) and not th.any(
            logits.isinf()), f"Logits contain NaN or Inf values: {logits}"
        assert not th.any(target_probs.isnan()) and not th.any(target_probs.isinf(
        )), f"Target probabilities contain NaN or Inf values: {target_probs}"
        assert th.all(target_probs >= 0.0) and th.all(
            target_probs <= 1.0), f"Target probabilities should be in [0, 1], but got {target_probs}"

    if no_grad_on_indexes:
        loss = th.empty_like(logits)
        detached_idx = th.as_tensor(
            no_grad_on_indexes, device=logits.device, dtype=th.long)
        # not_detached_idx = th.tensor([i for i in range(logits.shape[-1]) if i not in no_grad_on_indexes], device=logits.device, dtype=th.long)
        assert detached_idx.numel() > 0, "no_grad_on_indexes should be non-empty when provided"
        assert th.all((detached_idx >= 0) & (detached_idx < logits.shape[-1])).item(), (
            f"no_grad_on_indexes contains invalid indices for last dimension size {logits.shape[-1]}: {no_grad_on_indexes}"
        )
        logits[..., detached_idx] = logits[...,
                                           detached_idx].detach().requires_grad_(False)
    target_probs = target_probs.detach()
    
    if activate_disc_epsilon_for_loss and discordance_epsilon is not None:
        with th.no_grad():
            
            discordance = th.full_like(logits, fill_value=0.0)
            discordance = discordance.masked_fill(target_probs < 0.5, -discordance_epsilon)
            discordance = discordance.masked_fill(target_probs > 0.5, discordance_epsilon)
            if missing_mask is not None:
                discordance = discordance.masked_fill(missing_mask, 0.0)
        logits_app = logits-discordance
    else:
        logits_app = logits
    

    loss = th.nn.functional.binary_cross_entropy_with_logits(
        # /sum(weights)
        logits_app, target_probs, reduction='none')
    
    with th.no_grad():
        loss_best = th.nn.functional.binary_cross_entropy(
            # /sum(weights)
            target_probs, target_probs, reduction='none')
    loss = loss - loss_best
    # assert loss.shape == logits.shape, f"Expected loss shape {(logits.shape[0],)}, got {loss.shape}"
    mean = th.mean(loss, dim=-2)
    if rew_center_coefficient != 0 and rew_sum is not None:
        centering = th.mean((rew_sum)**2, dim=-2)
        # assert centering.shape == mean.shape, f"Expected centering shape {mean.shape}, got {centering.shape}"
        mean += rew_center_coefficient * centering
    # assert mean.shape == (logits.shape[-1],), f"Expected loss shape {(logits.shape[-1],)}, got {loss.shape}"
    if return_metrics:
        metrics = {}
        metrics['coherences'] = accuracy_logits(
            logits_p, target_probs_p, missing_mask=missing_mask, discordance_epsilon=discordance_epsilon)
        metrics['avg_coherence'] = metrics['coherences'].mean().item()
        return mean, metrics
    return mean


def value_system_loss_logits(logits_p: th.Tensor, target_probs_p: th.Tensor, rew_sum: th.Tensor = None, return_metrics: bool = False, check_undefined_label=True, rew_center_coefficient=0.0, missing_mask: th.Tensor = None, discordance_epsilon=MIN_EPSILON, activate_disc_epsilon_for_loss=False) -> th.Tensor:
    missing_mask = get_missing_rating_mask(
        target_probs_p) if check_undefined_label and missing_mask is None else missing_mask

    if check_undefined_label:
        logits = logits_p.masked_fill(missing_mask, 0.0)
        target_probs = target_probs_p.masked_fill(missing_mask, 0.5)
    else:
        logits = logits_p
        target_probs = target_probs_p
    target_probs = target_probs.detach()
    if activate_disc_epsilon_for_loss and discordance_epsilon is not None:
        with th.no_grad():
            
            discordance = th.full_like(logits, fill_value=0.0)
            discordance = discordance.masked_fill(target_probs < 0.5, -discordance_epsilon)
            discordance = discordance.masked_fill(target_probs > 0.5, discordance_epsilon)
            if missing_mask is not None:
                discordance = discordance.masked_fill(missing_mask, 0.0)
        logits_app = logits-discordance
    else:
        logits_app = logits

    loss = th.nn.functional.binary_cross_entropy_with_logits(
        # /sum(weights)
        # + rew_center_coefficient*th.mean((reward1 + reward2)**2, dim=-2)
        logits_app, target_probs, reduction='none')
    with th.no_grad():
        loss_best = th.nn.functional.binary_cross_entropy(
            # /sum(weights)
            target_probs, target_probs, reduction='none')

    loss = (loss - loss_best).mean()
    # assert loss.shape == reward1.shape, f"Expected loss shape {(reward1.shape[0],)}, got {loss.shape}"
    if rew_center_coefficient != 0:
        loss += rew_center_coefficient * th.mean((rew_sum)**2)

    if return_metrics:
        metrics = {}
        metrics['representativeness'] = accuracy_logits(
            logits_p, target_probs_p, missing_mask=missing_mask, discordance_epsilon=discordance_epsilon)
        return loss, metrics
    return loss


def context_loss_logits(config: MORMForClassificationConfig, logits_p: th.Tensor, target_probs_p: th.Tensor, ctx: CtxData, gr_rew_sum: th.Tensor=None, return_metrics: bool = False, check_undefined_label=True, rew_center_coefficient=0.0, missing_mask: th.Tensor = None, discordance_epsilon=MIN_EPSILON, activate_disc_epsilon_for_loss=False, sharp_classification=False) -> th.Tensor:
   

    if ctx.vs_logprobs is not None:
        bsz = ctx.vs_logprobs.size(0)

        jidx = th.arange(0, bsz, 2, device=ctx.vs_logprobs.device)
        kidx = jidx + 1

        assert ctx.extra_for_custom_loss[0].shape == (bsz,), f"Expected extra_for_custom_loss[0] shape {(bsz, )}, got {ctx.extra_for_custom_loss[0].shape}"
        logs1 = ctx.extra_for_custom_loss[0][jidx]
        logs2 = ctx.extra_for_custom_loss[0][kidx]
        assert th.allclose(logs1, logs2), f"Expected vs_logprobs to be the same for each pair, but got {logs1} and {logs2}"
        if ContextImplementations(config.context_implementation) == ContextImplementations.GMM:
            """
            extra_for_custom_loss = (gmm_logprob,per_component_logprob,component_logprobs)
                
            """
            if __debug__:
                with th.no_grad():
                    context_logprobabilities = ctx.extra_for_custom_loss[1][jidx] + ctx.extra_for_custom_loss[2]
                    assert th.allclose(logs1, th.logsumexp(context_logprobabilities, dim=1)), f"Expected vs_logprobs to be the same for each pair, but got {logs1} and {logs2}"
            ctx_prediction_loss = -logs1.mean() # TODO entropy reg...
        
        #print("VS P L", vs_prediction_loss, ctx.vs_logprobs.shape)
        return ctx_prediction_loss, CtxData.from_previous(ctx, ctx_pred_loss=ctx_prediction_loss )
    else:
        return 0, ctx
    
def value_system_selection_loss_logits(logits_p: th.Tensor, target_probs_p: th.Tensor, ctx: CtxData, gr_rew_sum: th.Tensor=None, return_metrics: bool = False, check_undefined_label=True, rew_center_coefficient=0.0, missing_mask: th.Tensor = None, discordance_epsilon=MIN_EPSILON, activate_disc_epsilon_for_loss=False, sharp_classification=False) -> th.Tensor:
    missing_mask = get_missing_rating_mask(
        target_probs_p)
    
    if check_undefined_label:
        logits = logits_p.masked_fill(missing_mask, 0.0)
        target_probs = target_probs_p.masked_fill(missing_mask, 0.5)
    else:
        logits = logits_p
        target_probs = target_probs_p
    target_probs = target_probs.detach()
    
    if ctx.vs_logprobs is not None:
        bsz = ctx.vs_logprobs.size(0)

        jidx = th.arange(0, bsz, 2, device=ctx.vs_logprobs.device)
        kidx = jidx + 1

        logs1 = ctx.vs_logprobs[jidx]
        assert logs1.shape[-1] == ctx.vs_possibilities.shape[0]
        
        logs2 = ctx.vs_logprobs[kidx]
        
    
        """ sharp seems good."""
        with th.no_grad():
            gr = logits[..., 0:-1] 
            
            predicted_logits_with_each_vs = gr @ ctx.vs_possibilities.T # This is [Batch size, NumValueSystems]
            gr_rew_sum_for_each_vs = gr_rew_sum @ ctx.vs_possibilities.T if gr_rew_sum is not None else None
            assert predicted_logits_with_each_vs.shape == (logits_p.shape[0], ctx.vs_possibilities.shape[0])
            
            #assert th.allclose(ls_p.exp().sum(dim=-1), th.ones_like(ls_p[..., 0])), f"Log softmax sum is not 1.0, got {ls_p.sum(dim=-1)}"
            #assert th.allclose(th.softmax(predicted_logits_with_each_vs, dim=-1).sum(dim=-1), th.ones_like(ls_p[..., 0])), f"Softmax sum is not 1.0, got {th.softmax(predicted_logits_with_each_vs, dim=-1).sum(dim=-1)}"
            
            #exit(0)
        target_probs_vs = target_probs_p[..., -1]
        missing_mask_vs = missing_mask[..., -1] if missing_mask is not None else None
        if sharp_classification:
            with th.no_grad():
                losses = []
                for i in range(predicted_logits_with_each_vs.shape[-1]):
                    pred_logits_i = predicted_logits_with_each_vs[..., i]
                    
                    pred_logits_i_app = apply_discordance_epsilon_to_logits(missing_mask_vs, target_probs_vs, pred_logits_i, discordance_epsilon=discordance_epsilon, activate_discordance_epsilon_for_loss=activate_disc_epsilon_for_loss)
                
                    loss = th.nn.functional.binary_cross_entropy_with_logits(pred_logits_i_app, target_probs_vs, reduction='none')
                    if rew_center_coefficient != 0 and gr_rew_sum is not None:
                        gr_rew_sum_for_each_vs_i = gr_rew_sum_for_each_vs[..., i]
                        centering = th.mean((gr_rew_sum_for_each_vs_i)**2)
                        loss += rew_center_coefficient * centering
                    losses.append(loss)
                losses = th.stack(losses, dim=-1)
                assert losses.shape == (logits_p.shape[0],predicted_logits_with_each_vs.shape[-1],), f"Expected losses shape {(logits_p.shape[0],predicted_logits_with_each_vs.shape[-1],)}, got {losses.shape}"

                best_one_hot = th.zeros_like(losses, device=losses.device)
                best_indices = th.argmin(losses, dim=-1)
                best_one_hot.scatter_(-1, best_indices.unsqueeze(-1), 1.0)
            assert th.allclose(logs1, logs2), f"Expected vs_logprobs to be the same for each pair, but got {logs1} and {logs2}"
            vs_prediction_loss = th.nn.functional.binary_cross_entropy_with_logits(logs1, best_one_hot.detach(), reduction='mean')
            vs_pred_diff = th.abs(th.softmax(logs1, dim=-1) - best_one_hot.detach()).mean()
            
        else:
            ls_p = th.nn.functional.logsigmoid(predicted_logits_with_each_vs)
            logsumexp = th.logsumexp(ls_p + th.nn.functional.log_softmax(logs1, dim=-1), dim=-1)
            probs_per_vs_via_log = th.exp(logsumexp)
            
            if __debug__:
                probs_per_vs =  (th.softmax(logs1, dim=-1) * th.sigmoid(predicted_logits_with_each_vs)).sum(dim=-1) 
                assert th.allclose(probs_per_vs_via_log, probs_per_vs, atol=1e-5), f"Expected log_probs_per_vs and probs_per_vs to be close, but got {probs_per_vs_via_log} and {probs_per_vs}"
            with th.no_grad():
                vs_pred_diff = th.abs(probs_per_vs_via_log - target_probs_vs).mean()
                
            
            vs_prediction_loss = th.nn.functional.binary_cross_entropy_with_logits(probs_per_vs_via_log, target_probs_vs, reduction='mean')

        #print("VS P L", vs_prediction_loss, ctx.vs_logprobs.shape)
        return vs_prediction_loss, CtxData.from_previous(ctx, vs_pred_loss=vs_prediction_loss, vs_pred_diff=vs_pred_diff )
    else:
        return 0, ctx
    
def reward_pairs_and_scores_to_logits_and_targets(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold=50.0, assume_qualitative_labels=False, check_undefined_label=True, assume_torch=True) -> tuple[th.Tensor, th.Tensor, Dict[str, Any]]:
    # assert check_undefined_label
    missing_mask = get_missing_rating_mask(
        scores1, scores2) if check_undefined_label else None
    logits_p = logits_BT(reward1, reward2, threshold=reward_diff_threshold, missing_mask=missing_mask,
                         assume_torch=assume_torch, check_undefined_label=missing_mask is not None)
    target_probs_p = scores_to_target_probs(scores1, scores2, reward_diff_threshold=reward_diff_threshold, assume_qualitative_labels=assume_qualitative_labels,
                                            check_undefined_label=missing_mask is not None, missing_mask=missing_mask, assume_torch=assume_torch)
    
    target_probs_quantitative = scores_to_target_probs(scores1, scores2, reward_diff_threshold=reward_diff_threshold, assume_qualitative_labels=not assume_qualitative_labels,
                                            check_undefined_label=missing_mask is not None, missing_mask=missing_mask, assume_torch=assume_torch)

    rew_sum = reward1 + reward2
    rew_sum = rew_sum.masked_fill(
        missing_mask, 0.0) if missing_mask is not None and check_undefined_label else rew_sum

    others = {
        'missing_mask': missing_mask,
        'rew_sum': rew_sum,
    }
    others["target_probs_quantitative"] = target_probs_quantitative if assume_qualitative_labels else target_probs_p
    others["target_probs_qualitative"] = target_probs_p if assume_qualitative_labels else target_probs_quantitative
    return logits_p, target_probs_p.detach(), others


def grounding_loss(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor = None, scores2: th.Tensor = None, reward_diff_threshold: float = 50.0, return_metrics: bool = False, assume_qualitative_labels=False, check_undefined_label=True, rew_center_coefficient=0.0, discordance_epsilon=MIN_EPSILON, activate_disc_epsilon_for_loss=False) -> th.Tensor:
    """Multi-objective Cross-entropy loss: target_probs(1,2)*log(exp(r1) / (exp(r1) + exp(r2)))- (1-target_probs(1,2))*log(exp(r2) / (exp(r1) + exp(r2)))"""

    logits_p, target_probs_p, others = reward_pairs_and_scores_to_logits_and_targets(
        reward1, reward2, scores1, scores2, reward_diff_threshold, assume_qualitative_labels, check_undefined_label)

    missing_mask = others['missing_mask']
    rew_sum = others['rew_sum']

    # assert len(reward1.shape) == 2 and reward1.shape[-1] == scores1.shape[-1], f"Expected reward1 shape (batch_size, num_values) and scores1 shape (batch_size, num_values), but got {reward1.shape} and {scores1.shape}"

    return grounding_loss_logits(logits_p, target_probs_p, rew_sum=rew_sum, return_metrics=return_metrics, check_undefined_label=check_undefined_label, rew_center_coefficient=rew_center_coefficient, missing_mask=missing_mask, discordance_epsilon=discordance_epsilon, activate_disc_epsilon_for_loss=activate_disc_epsilon_for_loss)


def value_system_loss(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold=50.0, assume_qualitative_labels=False, check_undefined_label=False, return_metrics=False, rew_center_coefficient=0.0, discordance_epsilon=MIN_EPSILON, activate_disc_epsilon_for_loss=False) -> th.Tensor:
    logits_p, target_probs_p, others = reward_pairs_and_scores_to_logits_and_targets(
        reward1, reward2, scores1, scores2, reward_diff_threshold, assume_qualitative_labels, check_undefined_label)

    missing_mask = others['missing_mask']
    rew_sum = others['rew_sum']

    return value_system_loss_logits(logits_p, target_probs_p, rew_sum=rew_sum, return_metrics=return_metrics, 
                                    check_undefined_label=check_undefined_label, 
                                    rew_center_coefficient=rew_center_coefficient, 
                                    missing_mask=missing_mask, 
                                    discordance_epsilon=discordance_epsilon, 
                                    activate_disc_epsilon_for_loss=activate_disc_epsilon_for_loss)


def mo_loss_function(logits, labels, others=None, ideal_logits=None, config: MORMForClassificationConfig = None, training_variables: MORMTrainingVariables = None, **kwargs):

    logits, labels, others_from_rewards = rewards_and_labels_to_logits_and_targets(
        logits, labels, assume_torch=True, config=config)
    missing_mask = others_from_rewards.get('missing_mask', None)
    grounding_mask = missing_mask[..., 0:-1] if missing_mask is not None else None
    vs_mask = missing_mask[..., -1] if missing_mask is not None else None

    epoch = kwargs.get('epoch', None)
    vs_selection_coefficient = config.vs_selection_coefficient
    ctx_coefficient = config.ctx_coefficient
    
    rew_sum = others_from_rewards.get('rew_sum', None)
    if rew_sum is not None:
        grounding_rew_sum = rew_sum[..., 0:-1]
        vs_rew_sum = rew_sum[..., -1]
    else:
        grounding_rew_sum = None
        vs_rew_sum = None

    if ideal_logits is not None:
        ideal_logits, _, _ = rewards_and_labels_to_logits_and_targets(
            ideal_logits, None, assume_torch=True, config=config)

    use_metrics = training_variables is not None and training_variables.use_metrics_or_losses == 'metrics'
    if config is not None:
        assert logits.shape[-1] == config.num_values + 1
        # We want to compute metrics at every step, even if not used for lagrange updates, for better monitoring and analysis.
        use_metrics = use_metrics or config.gather_train_metrics

    if config.loss_management.requires_grad_for_all_grounding_losses(epoch=epoch):
        gr_loss = grounding_loss_logits(logits[..., 0:-1], labels[..., 0:-1], rew_sum=grounding_rew_sum, missing_mask=grounding_mask,
                                            check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, 
                                            rew_center_coefficient=config.rew_center_coefficient, 
                                            discordance_epsilon=config.discordance_epsilon, 
                                            activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)

    elif config.loss_management.requires_grad_for_only_some_grounding_losses(epoch=epoch):
        # gr_loss = grounding_loss(rewards_1[...,0:-1], rewards_2[...,0:-1], scores1=labels_1[...,0:-1], scores2=labels_2[...,0:-1], reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
       
        value_indices = config.loss_func_type_kwargs.get(
            'value_indices', config.base_model_reward_head_indices)
        if value_indices is None:
            value_indices = list(range(logits.shape[-1] - 1))
        not_grad_indices = [i for i in range(
            logits.shape[-1] - 1) if i not in value_indices]
        gr_loss = grounding_loss_logits(
            logits[..., 0:-1],
            labels[..., 0:-1],
            rew_sum=grounding_rew_sum,
            missing_mask=grounding_mask,
            check_undefined_label=config.check_undefined_label,
            return_metrics=use_metrics,
            rew_center_coefficient=config.rew_center_coefficient,
            no_grad_on_indexes=not_grad_indices,
            discordance_epsilon=config.discordance_epsilon, 
            activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss
        )
    else:
        with th.no_grad():
            gr_loss = grounding_loss_logits(logits[..., 0:-1], labels[..., 0:-1], rew_sum=grounding_rew_sum, missing_mask=grounding_mask,
                                        check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient, discordance_epsilon=config.discordance_epsilon, 
                                        activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)

    if ideal_logits is not None:
        gr_loss_ideal = grounding_loss_logits(ideal_logits[..., 0:-1], labels[..., 0:-1], rew_sum=grounding_rew_sum, missing_mask=grounding_mask,
                                              check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient, discordance_epsilon=config.discordance_epsilon, 
                                              activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)

    if config.loss_management.requires_grad_for_value_system_loss(epoch=epoch):
        vs_loss = value_system_loss_logits(logits[..., -1], labels[..., -1], rew_sum=vs_rew_sum, missing_mask=vs_mask,
                                           check_undefined_label=config.check_undefined_label, 
                                           return_metrics=use_metrics, 
                                           rew_center_coefficient=config.rew_center_coefficient, 
                                           discordance_epsilon=config.discordance_epsilon, 
                                           activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)
    else:
        with th.no_grad():
            # vs_loss = value_system_loss(rewards_1[...,-1],rewards_2[...,-1], scores1=labels_1[..., -1], scores2=labels_2[..., -1] , reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
            vs_loss = value_system_loss_logits(logits[..., -1], labels[..., -1], rew_sum=vs_rew_sum, missing_mask=vs_mask,
                                               check_undefined_label=config.check_undefined_label, 
                                               return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient, 
                                               discordance_epsilon=config.discordance_epsilon, 
                                               activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)
    
    vs_selection_loss = 0
    ctx_loss = 0
    
    ctx_data = None
    if ContextImplementations(config.context_implementation) != ContextImplementations.NO_CONTEXT and vs_selection_coefficient > 0.0:
        if 'ctx' not in others.keys():
                raise ValueError("Program expected a CtxData object returned by the forward method.")
        ctx_data = CtxData.from_dict(others['ctx'])
        if config.loss_management.requires_grad_for_value_system_selection_loss(epoch=epoch):
            #assert ContextImplementations(config.context_implementation) != ContextImplementations.NO_CONTEXT
            vs_selection_loss, ctx_data = value_system_selection_loss_logits(logits, labels, ctx=ctx_data,
                                                sharp_classification=config.sharp_context_classification,
                                                  gr_rew_sum=grounding_rew_sum, missing_mask=vs_mask,
                                                check_undefined_label=config.check_undefined_label, 
                                                return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient, 
                                                discordance_epsilon=config.discordance_epsilon, 
                                                activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)

        else:
            with th.no_grad():
                vs_selection_loss, ctx_data = value_system_selection_loss_logits(logits, labels, sharp_classification=config.sharp_context_classification,
                                                    ctx=ctx_data, gr_rew_sum=grounding_rew_sum, missing_mask=vs_mask,
                                                check_undefined_label=config.check_undefined_label, 
                                                return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient, 
                                                discordance_epsilon=config.discordance_epsilon, 
                                                activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)
    if ContextImplementations(config.context_implementation) != ContextImplementations.NO_CONTEXT and ctx_coefficient > 0.0:
        ctx_data = CtxData.from_dict(others['ctx']) if ctx_data is None else ctx_data
        if 'ctx' not in others.keys():
                raise ValueError("Program expected a CtxData object returned by the forward method.")
        if config.loss_management.requires_grad_for_context_loss(epoch=epoch):
            #assert ContextImplementations(config.context_implementation) != ContextImplementations.NO_CONTEXT
            ctx_loss, ctx_data = context_loss_logits(config, logits, labels, ctx=ctx_data,
                                                sharp_classification=config.sharp_context_classification,
                                                    gr_rew_sum=grounding_rew_sum, missing_mask=vs_mask,
                                                check_undefined_label=config.check_undefined_label, 
                                                return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient, 
                                                discordance_epsilon=config.discordance_epsilon, 
                                                activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)

        else:
            with th.no_grad():
                ctx_loss, ctx_data = context_loss_logits(config, logits, labels, sharp_classification=config.sharp_context_classification,
                                                            ctx=ctx_data, gr_rew_sum=grounding_rew_sum, missing_mask=vs_mask,
                                                        check_undefined_label=config.check_undefined_label, 
                                                        return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient, 
                                                        discordance_epsilon=config.discordance_epsilon, 
                                                        activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)
        
    with th.no_grad():
        if use_metrics:
            metrics_grounding: dict = gr_loss[1]
            metrics_value_system: dict = vs_loss[1]
            # join the two dicts
            metrics = {**metrics_grounding, **metrics_value_system}
            if ideal_logits is not None:
                metrics_grounding_ideal: dict = gr_loss_ideal[1]
                metrics = {**metrics, **{f"{k}_ideal": v for k,
                                         v in metrics_grounding_ideal.items()}}
    if use_metrics:
        vs_loss = vs_loss[0]
        gr_loss = gr_loss[0]
        gr_loss_ideal = gr_loss_ideal[0] if ideal_logits is not None else None

    if th.is_grad_enabled():
        ctx_data: CtxData
        if ContextImplementations(config.context_implementation) == ContextImplementations.GMM:
            print("CTXP", ctx_data.context_logprobs[0:4])
            print("VSP", ctx_data.vs_logprobs[0:4])
            print("CTXPexp", th.exp(ctx_data.context_logprobs)[0:4])
            print("CTXPsoft", th.softmax(ctx_data.context_logprobs, dim=1)[0:4])
            print("CTX_LOSS", ctx_loss)
            print("VS_SELECTION_LOSS", vs_selection_loss)
            print("VS_LOSS", vs_loss)
            #exit(0)
    vs_loss = vs_loss + vs_selection_loss*vs_selection_coefficient +ctx_loss*ctx_coefficient    
    #vs_loss = vs_selection_loss + 0.0000001*vs_loss
    if th.is_grad_enabled() and training_variables is not None:
        with th.no_grad():
            grl = gr_loss.detach()
            vsl = vs_loss.detach()
            grli = gr_loss_ideal.detach() if ideal_logits is not None else None
            training_variables.record_grounding_loss(
                gr_loss_detached=grl, vs_loss_detached=vsl, gr_loss_ideal_detached=grli)
    if use_metrics and training_variables is not None:
        if vs_selection_loss == 0.0:
            metrics["vs_selection_loss"] = 0.0
        else:
            metrics["vs_selection_loss"] = vs_selection_loss.detach().item()
        if ctx_loss == 0.0:
            metrics["ctx_loss"] = 0.0
        else:
            metrics["ctx_loss"] = ctx_loss.detach().item()
        # assert "representativeness" in metrics.keys() and "coherences" in metrics.keys(), f"Expected metrics to contain 'representativeness' and 'coherences', but got {metrics.keys()}"
        training_variables.record_metrics(metrics, metric_type="train")

    if ideal_logits is not None:
        return th.cat([gr_loss, gr_loss_ideal, vs_loss.reshape(-1)])
    else:
        return th.cat([gr_loss, vs_loss.reshape(-1)])


def mo_compute_loss_func(outputs, labels, config=None, training_variables=None, **kwargs):
    # assert config is not None, "Config must be provided to mo_compute_loss_func"
    # assert training_variables is not None, "Training variables must be provided to mo_compute_loss_func"
    # assert outputs.logits.device == labels.device, "Devices do not match"
   
    id_logits = getattr(outputs, "ideal_logits", None)
    others = getattr(outputs, "other", None)
    return parse_loss_function(config)(outputs.logits, labels, others=others, ideal_logits=id_logits, config=config, training_variables=training_variables, **kwargs)


@dataclass
class SequenceClassifierOutputWithPastAndOthers(SequenceClassifierOutputWithPast):
    other: Any = None

@dataclass
class SequenceClassifierOutputWithPastAndIdeal(SequenceClassifierOutputWithPastAndOthers):
    ideal_logits: th.Tensor = None
    other: Any = None



@dataclass
class ClassifierOutputWithPast(ModelOutput):
    logits: th.Tensor
    loss: th.Tensor = None
    other: Any = None

@dataclass
class ClassifierOutputWithPastAndIdeal(ClassifierOutputWithPast):
    ideal_logits: Optional[th.Tensor] = None
    
    

class MultiValueRewardHead(nn.Module):
    def __init__(self, value_heads: nn.ModuleList, normalization: nn.Module, optimized_head_indices: Optional[list[int]] = None):
        super().__init__()
        self.value_heads = value_heads
        self.normalization = normalization
        self.optimized_head_indices = set(
            optimized_head_indices if optimized_head_indices is not None else list(range(len(value_heads))))
        for head_i in range(len(value_heads)):
            if head_i in self.optimized_head_indices:
                self.value_heads[head_i].requires_grad_(True)
            else:
                self.value_heads[head_i].requires_grad_(False)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        yield from self.normalization.parameters(recurse=recurse)

        for head_i, head in enumerate(self.value_heads):
            if head_i in self.optimized_head_indices:
                yield from head.parameters(recurse=recurse)

    def forward(self, hidden_state: th.Tensor) -> th.Tensor:
        rewards_list = []
        for head_i, head in enumerate(self.value_heads):
            if head_i in self.optimized_head_indices:
                rewards_list.append(head(hidden_state))
            else:
                with th.no_grad():
                    rewards_list.append(head(hidden_state))

        rewards = th.cat(rewards_list, dim=-1)
        return self.normalization(rewards)

    
    def reference_weight(self) -> th.Tensor:
        # Used for dtype/device consistency assertions.
        for layer in self.value_heads[0]:
            if isinstance(layer, nn.Linear):
                return layer.weight
        raise ValueError("Expected at least one Linear layer in value head")

    def extra_repr(self) -> str:
        return_str = ""
        return_str += f"Normalization: {self.normalization}\n Heads:\n"
        for head_i, head in enumerate(self.value_heads):
            if head_i in self.optimized_head_indices:
                optimized_str = "optimized"
            else:
                optimized_str = "frozen"
            head_str = f"head_{head_i} ({optimized_str}): {head}"
            return_str += head_str + "\n"
        return return_str

class MORMForClassification(PreTrainedModel):
    config_class = MORMForClassificationConfig
    supports_gradient_checkpointing = False
    
    @property
    def grounding_features_name(self) -> str:
        return 'grounding_features'

    @property
    def vs_features_name(self) -> Optional[str]:
        return CONTEXT_FEATURE_NAME


    classifier_ouput_class = ClassifierOutputWithPast
    classifier_output_class_ideal = ClassifierOutputWithPastAndIdeal
    def __init__(self, config: MORMForClassificationConfig, *args, **kwargs):
        super().__init__(config)
        self.supports_gradient_checkpointing = False
        self.config = config
        self.reward_heads: Optional[MultiValueRewardHead] = None
        self.reward_heads_ideal: Optional[MultiValueRewardHead] = None
        self.value_system_layer: Optional[LinearAlignmentLayer] = None
        self.training_variables: Optional[MORMTrainingVariables] = None

        self.config = config
        print(f"Initializing MORMForSequenceClassification")
        self.use_base_model_heads = config.use_base_model_heads
        self.base_model_reward_head_indices = config.base_model_reward_head_indices
        self.base_model_rewards_attr_name = config.base_model_reward_heads_module_name
        self.base_model_score_attr_name = config.base_model_value_system_module_name
        
        self.num_values = config.num_values
        self.use_ideal_grounding_model = config.use_ideal_grounding_model
        self.forward_ideal_grounding = self.use_ideal_grounding_model

        
        self.init_networks(config,  *args, **kwargs)

        self.post_init()

        # TODO: Apparetly this is much faster. See https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        self.zero_grad(set_to_none=True)

    def init_networks(self, config, *args, **kwargs) -> None:

        # TODO: Apparetly this is much faster. See https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        

        model_device = "cuda:0" if th.cuda.is_available() else "cpu"
        
        self.create_reward_networks(config, model_device)

        # if config.training_variables_dtype == "float32" else th.float16 if config.training_variables_dtype == "float16" else self._resolve_torch_dtype(config.training_variables_dtype)
        self.create_training_variables(config, model_device)


    def _set_train_mode(self, train_mode: bool):
        for param in self.grounding_parameters():
            param.requires_grad = train_mode

        for param in self.value_system_parameters():
            param.requires_grad = train_mode
        
        for param in self.context_parameters():
            param.requires_grad = train_mode
        # This is set to True inside training_variables in prepare_for_optimizer_step method.
        self.training_variables.requires_grad_(False)
        
    def train_initialization(self, train_subdataset: Dataset, args: TrainingArguments, eval_set: Dataset = None, total_dataset_size: int = None):
        
        prev_config_loss_func_type = self.config.loss_func_type
        self.config.loss_func_type = MOLossFunctions.DEFAULT.value
        dataset_ctxs = np.array(train_subdataset.select_columns([self.vs_features_name])[self.vs_features_name])
        print("Dataset contexts shape:", dataset_ctxs.shape)
        print("Dataset contexts sample:", dataset_ctxs[0],  np.linalg.norm(dataset_ctxs[0], ord=2) )
        kmeans = kmeans_clustering(dataset_ctxs, K= self.config.max_contexts)
            
        pred = kmeans.predict(dataset_ctxs)
        centroids = kmeans.cluster_centers_
        print("INIT", pred[0:10], pred.shape)
        
        
        #random_assignment = np.random.choice(self.value_system_layer.num_contexts, size=len(dataset_ctxs))
        #print(random_assignment[0:10], random_assignment.shape)
        pred_one_hot = th.eye(self.value_system_layer.num_contexts, requires_grad=False)[pred]
        dataset_ctxs_th = th.tensor(train_subdataset.select_columns([self.vs_features_name])[self.vs_features_name], requires_grad=False)

        self.value_system_layer : BasicCtxDependentAlignmentLayer

        #self.train_context_log_probs_network_to_predict_KmeansClusters(pred_one_hot, dataset_ctxs_th)
        if ContextImplementations(self.config.context_implementation) in [ContextImplementations.BASIC_HARSH, ContextImplementations.DIRECT_VS, ContextImplementations.BASIC_SMOOTH]:
            self.train_context_log_probs_network_to_predict_KmeansClusters(pred_one_hot, dataset_ctxs_th)
        # "Pretrain" the network to assign to each cluster the best value system. (given current initialization)
        elif ContextImplementations(self.config.context_implementation) in [ContextImplementations.GMM, ContextImplementations.NESTED_GMM]:
            assert isinstance(self.value_system_layer, BasicGmmCtxDependentAlignmentLayer), f"Expected value_system_layer to be an instance of BasicGmmCtxDependentAlignmentLayer, but got {type(self.value_system_layer)}"
            self.value_system_layer.initialize_gmm(th.tensor(centroids, dtype=th.float32), dataset_ctxs_th, th.tensor(pred, requires_grad=False, dtype=th.long))
            self.pre_train_grounding_and_vs_selection_matrix_for_gmm(kmeans, train_subdataset, pred, dataset_ctxs_th, args=args, eval_set=eval_set, total_dataset_size=total_dataset_size)
        else:    
            # TODO: Missing mask needed.
            self.train_context_log_probs_so_that_each_cluster_aligns_with_a_value_system(kmeans, train_subdataset, pred, dataset_ctxs_th, args=args, eval_set=eval_set, total_dataset_size=total_dataset_size)
        
        self.config.loss_func_type = prev_config_loss_func_type
    def train_context_log_probs_network_to_predict_KmeansClusters(self, pred_one_hot: np.ndarray, dataset_ctxs_th: th.Tensor):
        self.train()
        optimizer = th.optim.AdamW(self.value_system_layer.context_logprobabilities.parameters(), lr=self.config.lr_context, weight_decay=0.003)
        lossfun = th.nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = 1000.0
        pbar = tqdm.tqdm(range(500))
        for t in pbar:
            optimizer.zero_grad()
            f = self.value_system_layer.context_logprobabilities(dataset_ctxs_th)
            loss = lossfun(f, pred_one_hot)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        print("Finished context network initialization")
        self.eval()


    def train_context_log_probs_so_that_each_cluster_aligns_with_a_value_system(self, kmeans: KMeans, train_subdataset: Dataset, pred: np.ndarray, dataset_ctxs_th: th.Tensor, args: TrainingArguments, eval_set: Dataset = None, total_dataset_size: int = None):
        with th.no_grad():
                dataset_grounding_features1 = th.tensor(train_subdataset.select_columns([self.grounding_features_name + "_1"])[self.grounding_features_name + "_1"])
                dataset_grounding_features2 = th.tensor(train_subdataset.select_columns([self.grounding_features_name + "_2"])[self.grounding_features_name + "_2"])
                dataset_score = th.tensor(train_subdataset.select_columns(["labels"])["labels"])[..., -1]
                dataset_gr_scores = th.tensor(train_subdataset.select_columns(["labels"])["labels"])[..., 0:-1]
                
                dataset_gr_scores = scores_to_target_probs(dataset_gr_scores[:,0, :], dataset_gr_scores[:,1, :], assume_qualitative_labels=self.config.assume_qualitative_labels, check_undefined_label=self.config.check_undefined_label, missing_mask=None, assume_torch=True)
                
                missing_mask_gr = dataset_gr_scores == NO_RATING_MASK 
                
                dataset_score=scores_to_target_probs(dataset_score[:,0], dataset_score[:,1], assume_qualitative_labels=self.config.assume_qualitative_labels, check_undefined_label=self.config.check_undefined_label, missing_mask=None, assume_torch=True)
                missing_mask = dataset_score==NO_RATING_MASK
                #dataset_score_valid = dataset_score.masked_fill(missing_mask, 0.5)
                dataset_score_gr_valid = dataset_gr_scores.masked_fill(missing_mask_gr, 0.5)
                pred_valid = pred[~missing_mask]

        self.train()
        #optimizer = th.optim.AdamW((self.value_system_layer.context_to_vslogweights_matrix,*self.grounding_parameters()), lr=0.005, weight_decay=0.01)
        optimizer = th.optim.AdamW((*self.grounding_parameters(),), lr=self.config.lr_grounding, weight_decay=0.003)
         
            
        loss = 1000.0
            
        pbar = tqdm.tqdm(range(500))
        grounding_features_per_cluster_1 = [dataset_grounding_features1[~missing_mask][pred_valid==c] for c in range(len(kmeans.cluster_centers_))]
        grounding_features_per_cluster_2 = [dataset_grounding_features2[~missing_mask][pred_valid==c] for c in range(len(kmeans.cluster_centers_))]
        targets = [dataset_score[~missing_mask][pred_valid==c] for c in range(len(kmeans.cluster_centers_))]
        ctx_features_per_cluster = [dataset_ctxs_th[~missing_mask][pred_valid==c] for c in range(len(kmeans.cluster_centers_))]
        
        print("Grounding initialization")
        for t in pbar:
            loss_total = 0
            optimizer.zero_grad()
            
            
                    
            rewards1 = self.reward_heads(dataset_grounding_features1)
            rewards2 = self.reward_heads(dataset_grounding_features2)
            targets = dataset_score_gr_valid
            #s1, other1 = self.value_system_layer.forward(rewards1, hidden_state=context_features_in_c)
            #s2, other2 = self.value_system_layer.forward(rewards2, hidden_state=context_features_in_c)
                #print(s1, s2)
            bt = logits_BT(rewards1, rewards2, check_undefined_label=self.config.check_undefined_label, missing_mask=missing_mask if self.config.check_undefined_label else None, assume_torch=True)
            rew_sum = rewards1+rewards2
            assert bt.shape == (len(dataset_ctxs_th), self.num_values)
                #print(th.sum((th.abs(th.sigmoid(-bt)[0:10]-targets_in_c[0:10]))), "wtf")
            for vi in range(self.num_values):
                loss = th.nn.functional.binary_cross_entropy_with_logits(bt[:, vi], targets[: ,vi])
                if self.config.rew_center_coefficient > 0.0:
                    loss += self.config.rew_center_coefficient*th.mean((rew_sum[:, vi])**2)
                loss_total = loss + loss_total
                
            loss_total.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss_total.item()})

        
        self.train()
        #optimizer = th.optim.AdamW((self.value_system_layer.context_to_vslogweights_matrix,*self.grounding_parameters()), lr=0.005, weight_decay=0.01)
        optimizer = th.optim.AdamW((*self.value_system_layer.context_logprobabilities.parameters(),), lr=self.config.lr_value_system, weight_decay=0.001)
         
            
        loss = 1000.0
        len_dataset = len(dataset_grounding_features1)
        pbar = tqdm.tqdm(range(int(len_dataset//args.train_batch_size*0.1*args.num_train_epochs)))
        grounding_features_per_cluster_1 = [dataset_grounding_features1[~missing_mask][pred_valid==c] for c in range(len(kmeans.cluster_centers_))]
        grounding_features_per_cluster_2 = [dataset_grounding_features2[~missing_mask][pred_valid==c] for c in range(len(kmeans.cluster_centers_))]
        targets = [dataset_score[~missing_mask][pred_valid==c] for c in range(len(kmeans.cluster_centers_))]
        ctx_features_per_cluster = [dataset_ctxs_th[~missing_mask][pred_valid==c] for c in range(len(kmeans.cluster_centers_))]
        
        
        for t in pbar:
            loss_total = 0
            optimizer.zero_grad()
            for c in range(len(kmeans.cluster_centers_)):
                grounding_features_in_c_1 = grounding_features_per_cluster_1[c]
                grounding_features_in_c_2 = grounding_features_per_cluster_2[c]
                context_features_in_c =  ctx_features_per_cluster[c]
                    #print("CTX, ", context_features_in_c[0:10])
                targets_in_c = targets[c]
                    
                rewards1 = self.reward_heads(grounding_features_in_c_1)
                rewards2 = self.reward_heads(grounding_features_in_c_2)

                s1, other1 = self.value_system_layer.forward(rewards1, hidden_state=context_features_in_c)
                s2, other2 = self.value_system_layer.forward(rewards2, hidden_state=context_features_in_c)
                    #print(s1, s2)
                bt = logits_BT(s1.squeeze(1), s2.squeeze(1), check_undefined_label=self.config.check_undefined_label, missing_mask=missing_mask if self.config.check_undefined_label else None, assume_torch=True)
                rew_sum = rewards1+rewards2
                assert bt.shape == targets_in_c.shape
                    #print(th.sum((th.abs(th.sigmoid(-bt)[0:10]-targets_in_c[0:10]))), "wtf")
                loss = th.nn.functional.binary_cross_entropy_with_logits(bt, targets_in_c)
                if self.config.rew_center_coefficient > 0.0:
                    loss += self.config.rew_center_coefficient*th.mean((rew_sum)**2)
                loss_total=loss+loss_total
            loss_total.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss_total.item()})
        #print("PARAMS CTX", list(self.value_system_layer.context_logprobabilities.parameters())[0:2])
        #print("MATRIX2", self.value_system_layer.context_to_vslogweights_matrix)
        #print("MATRIX2", self.value_system_layer.get_value_systems())
        
        self.train(False)


    def pre_train_grounding_and_vs_selection_matrix_for_gmm(self, kmeans: KMeans, train_subdataset: Dataset, pred: np.ndarray, dataset_ctxs_th: th.Tensor, args: TrainingArguments, eval_set: Dataset = None, total_dataset_size: int = None):
            with th.no_grad():
                    dataset_grounding_features1 = th.tensor(train_subdataset.select_columns([self.grounding_features_name + "_1"])[self.grounding_features_name + "_1"])
                    dataset_grounding_features2 = th.tensor(train_subdataset.select_columns([self.grounding_features_name + "_2"])[self.grounding_features_name + "_2"])
                    dataset_score = th.tensor(train_subdataset.select_columns(["labels"])["labels"])[..., -1]
                    dataset_gr_scores = th.tensor(train_subdataset.select_columns(["labels"])["labels"])[..., 0:-1]
                    
                    dataset_gr_scores = scores_to_target_probs(dataset_gr_scores[:,0, :], dataset_gr_scores[:,1, :], assume_qualitative_labels=self.config.assume_qualitative_labels, check_undefined_label=self.config.check_undefined_label, missing_mask=None, assume_torch=True)
                    
                    missing_mask_gr = dataset_gr_scores == NO_RATING_MASK 
                    
                    dataset_score=scores_to_target_probs(dataset_score[:,0], dataset_score[:,1], assume_qualitative_labels=self.config.assume_qualitative_labels, check_undefined_label=self.config.check_undefined_label, missing_mask=None, assume_torch=True)
                    missing_mask = dataset_score==NO_RATING_MASK
                    #dataset_score_valid = dataset_score.masked_fill(missing_mask, 0.5)
                    dataset_score_gr_valid = dataset_gr_scores.masked_fill(missing_mask_gr, 0.5)
                    pred_valid = pred[~missing_mask]
    
            self.train()
            #optimizer = th.optim.AdamW((self.value_system_layer.context_to_vslogweights_matrix,*self.grounding_parameters()), lr=0.005, weight_decay=0.01)
            optimizer = th.optim.AdamW((*self.grounding_parameters(),), lr=self.config.lr_grounding, weight_decay=args.weight_decay)
             
                
            loss = 1000.0
            len_dataset = len(dataset_grounding_features1)
            
            iterations_total = int((total_dataset_size//args.train_batch_size)*args.num_train_epochs*0.1)
            print("Iterations total:", iterations_total)
            
                
            pbar = tqdm.tqdm(range(iterations_total))
            batch_size = min(args.train_batch_size, len_dataset)
            if batch_size <= 0:
                raise ValueError("No valid examples available for value-system pretraining")
            batches_per_epoch = (len_dataset + batch_size - 1) // batch_size
            
            print("Grounding initialization")
            for t in pbar:
                loss_total = 0
                accuracy_total = 0
                optimizer.zero_grad()
                with th.no_grad():
                    epoch_step = t % batches_per_epoch
                    if epoch_step == 0 and t >= 0:
                        perm = th.randperm(
                            len_dataset, device=dataset_grounding_features1.device
                        )
                    start = epoch_step * batch_size
                    end = min(start + batch_size, len_dataset)
                    batch_indices = perm[start:end]
                    batch_indices = batch_indices.reshape((len(batch_indices),))
                rewards1 = self.reward_heads(dataset_grounding_features1[batch_indices])
                rewards2 = self.reward_heads(dataset_grounding_features2[batch_indices])

                targets = dataset_score_gr_valid[batch_indices]
                assert targets.shape == (len(batch_indices), self.num_values)
                #s1, other1 = self.value_system_layer.forward(rewards1, hidden_state=context_features_in_c)
                #s2, other2 = self.value_system_layer.forward(rewards2, hidden_state=context_features_in_c)
                    #print(s1, s2)
                bt_normal = logits_BT(rewards1, rewards2, check_undefined_label=self.config.check_undefined_label, missing_mask=missing_mask if self.config.check_undefined_label else None, assume_torch=True)
                assert bt_normal.shape == (len(batch_indices), self.num_values)
                assert targets.shape == (len(batch_indices), self.num_values)
                assert missing_mask_gr[batch_indices].shape == (len(batch_indices), self.num_values)
                bt = apply_discordance_epsilon_to_logits(missing_mask_gr[batch_indices], targets, bt_normal, discordance_epsilon=self.config.discordance_epsilon, activate_discordance_epsilon_for_loss=self.config.activate_discordance_epsilon_for_loss)
                rew_sum = rewards1+rewards2
                
                    #print(th.sum((th.abs(th.sigmoid(-bt)[0:10]-targets_in_c[0:10]))), "wtf")
                for vi in range(self.num_values):
                    loss = th.nn.functional.binary_cross_entropy_with_logits(bt[:, vi], targets[: ,vi])
                    if self.config.rew_center_coefficient > 0.0:
                        loss += self.config.rew_center_coefficient*th.mean((rew_sum)**2)
                    loss_total = loss + loss_total
                    with th.no_grad():
                        accuracy = ((th.sigmoid(bt_normal[:, vi]) > 0.5) == (targets[:, vi] > 0.5)).float().mean()
                        accuracy_total = accuracy + accuracy_total if t > 0 else accuracy
                loss_total.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss_total.item(), 'accuracy': (accuracy_total/float(self.num_values)).item()})
    
            
            self.train()

            assert isinstance(self.value_system_layer, BasicGmmCtxDependentAlignmentLayer), f"Expected value_system_layer to be an instance of BasicGmmCtxDependentAlignmentLayer, but got {type(self.value_system_layer)}"
            #optimizer = th.optim.AdamW((self.value_system_layer.context_to_vslogweights_matrix,*self.grounding_parameters()), lr=0.005, weight_decay=0.01)
            print("Pretraining value system selection matrix")
            freeze_ctx = False
            lr = self.config.lr_value_system
            weight_decay = args.weight_decay if args.weight_decay is not None else 0.001
            entropy_coef = 0.0
            if freeze_ctx:
                assert self.value_system_layer.context_to_vs_logprobabilities.shape == (self.value_system_layer.context_to_vs_logprobabilities.shape[0], self.value_system_layer.context_to_vs_logprobabilities.shape[0])
                with th.no_grad():
                    self.value_system_layer.context_to_vs_logprobabilities.copy_(th.eye(self.value_system_layer.context_to_vs_logprobabilities.shape[0], device=self.value_system_layer.context_to_vs_logprobabilities.device, dtype=self.value_system_layer.context_to_vs_logprobabilities.dtype, requires_grad=True)*1000.0)
                optimizer = th.optim.AdamW(( self.value_system_layer.vs_selection_to_logweights_matrix,), lr=lr, weight_decay=weight_decay)
                            
            else:
                optimizer = th.optim.AdamW(( self.value_system_layer.vs_selection_to_logweights_matrix, self.value_system_layer.context_to_vs_logprobabilities,), lr=lr, weight_decay=weight_decay)
             
                
            loss = 1000.0
            pbar = tqdm.tqdm(range(iterations_total))
            valid_grounding_features1 = dataset_grounding_features1[~missing_mask]
            len_dataset = len(valid_grounding_features1)
            valid_grounding_features2 = dataset_grounding_features2[~missing_mask]
            valid_targets = dataset_score[~missing_mask]
            valid_context_features = dataset_ctxs_th[~missing_mask]
            batch_size = min(args.train_batch_size, len_dataset)
            if batch_size <= 0:
                raise ValueError("No valid examples available for value-system pretraining")
            batches_per_epoch = (len_dataset + batch_size - 1) // batch_size

            
            
            for t in pbar:
                loss_total = 0
                with th.no_grad():
                    epoch_step = t % batches_per_epoch
                    if epoch_step == 0 and t >= 0:
                        perm2 = th.randperm(
                            len_dataset, device=valid_grounding_features1.device
                        )
                    start = epoch_step * batch_size
                    end = min(start + batch_size, len_dataset)
                    batch_indices = perm2[start:end]
                    rewards1 = self.reward_heads(valid_grounding_features1[batch_indices])
                    rewards2 = self.reward_heads(valid_grounding_features2[batch_indices])
                optimizer.zero_grad()
                
                targets = valid_targets[batch_indices]
                
                context_features = valid_context_features[batch_indices]
                
                assert context_features.shape == (len(batch_indices), dataset_ctxs_th.shape[1])
                assert rewards1.shape == (len(batch_indices), self.num_values)
                assert rewards2.shape == (len(batch_indices), self.num_values)
                s1, other1 = self.value_system_layer.forward(rewards1, hidden_state=context_features)
                s2, other2 = self.value_system_layer.forward(rewards2, hidden_state=context_features)
                    #print(s1, s2)
                
                vs_rewards1 = s1.squeeze(1)
                vs_rewards2 = s2.squeeze(1)
                bt_normal = logits_BT(s1.squeeze(1), s2.squeeze(1), check_undefined_label=self.config.check_undefined_label, missing_mask=missing_mask if self.config.check_undefined_label else None, assume_torch=True)
                assert bt_normal.shape == (len(batch_indices), )
                assert targets.shape == (len(batch_indices), )
                assert missing_mask[batch_indices].shape == (len(batch_indices), )
                bt = apply_discordance_epsilon_to_logits(missing_mask[batch_indices], targets, bt_normal, discordance_epsilon=self.config.discordance_epsilon, activate_discordance_epsilon_for_loss=self.config.activate_discordance_epsilon_for_loss)
                rew_sum = vs_rewards1+vs_rewards2
                assert bt.shape == targets.shape
                    #print(th.sum((th.abs(th.sigmoid(-bt)[0:10]-targets_in_c[0:10]))), "wtf")
                
                loss_total = th.nn.functional.binary_cross_entropy_with_logits(bt, targets) #- self.value_system_layer.context_to_vs_logprobabilities
                if self.config.rew_center_coefficient > 0.0:
                    loss_total += self.config.rew_center_coefficient*th.mean((rew_sum)**2)
                #substract the entropy of the context_to_vs_logprobabilities matrix to encourage diversity in value system selection.
                entropy = entropy_coef * th.mean(th.sum(th.softmax(self.value_system_layer.context_to_vs_logprobabilities, dim=1) * th.log_softmax(self.value_system_layer.context_to_vs_logprobabilities, dim=1)))
                loss_total = loss_total - entropy
                
                loss_total.backward()
                with th.no_grad():
                    accuracy = ((th.sigmoid(bt_normal) > 0.5) == (targets > 0.5)).float().mean()
                    
                optimizer.step()
                pbar.set_postfix({'loss': loss_total.item(), "accuracy": accuracy.item(), "entropy": entropy.item()})

                # Every 100 steps, draw the self.value_system_layer.context_to_vs_logprobabilities matrix with matplotlib and save it to a file.
                
                if t % 100 == 0:
                    
                    os.makedirs("pretrain_plots", exist_ok=True)
                    self.plot_matrices(t, filename=f"___context_to_vs_logprobabilities_step_{t}_DIRICHL_freeze_{freeze_ctx}_wd_{weight_decay}_lr_{lr}_entropy_coef_{entropy_coef}_acc_{accuracy}")
            #print("PARAMS CTX", list(self.value_system_layer.context_logprobabilities.parameters())[0:2])
            #print("MATRIX2", self.value_system_layer.context_to_vslogweights_matrix)
            #print("MATRIX2", self.value_system_layer.get_value_systems())
            
            self.train(False)

    def plot_matrices(self, t: int, filename):
        context_to_vs = th.softmax(
                        self.value_system_layer.context_to_vs_logprobabilities, dim=1
                    ).detach()
        vs_weights = th.softmax(
                        self.value_system_layer.vs_selection_to_logweights_matrix, dim=1
                    ).detach()
        selected_vs = th.argmax(context_to_vs, dim=1)
        selected_vs_weights = vs_weights[selected_vs].cpu().numpy()
        averaged_vs_weights = (context_to_vs @ vs_weights).cpu().numpy()
        context_to_vs = context_to_vs.cpu().numpy()
        vs_weights = vs_weights.cpu().numpy()

        figure, axes = plt.subplots(1, 4, figsize=(32, 8))
        axes[0].imshow(context_to_vs, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
        axes[0].set_title("Context to Value System")
        axes[0].set_xlabel("Value Systems")
        axes[0].set_ylabel("Contexts")
        axes[1].imshow(vs_weights, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
        axes[1].set_title("Value System Weights")
        axes[1].set_xlabel("Values")
        axes[1].set_ylabel("Value Systems")
        axes[2].imshow(selected_vs_weights, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
        axes[2].set_title("Selected Value System per Context")
        axes[2].set_xlabel("Values")
        axes[2].set_ylabel("Contexts")
        axes[3].imshow(averaged_vs_weights, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
        axes[3].set_title("Averaged Value System per Context")
        axes[3].set_xlabel("Values")
        axes[3].set_ylabel("Contexts")
        figure.suptitle(f"Value System Probabilities (Step {t})")
        figure.colorbar(axes[0].images[0], ax=axes[0])
        figure.colorbar(axes[1].images[0], ax=axes[1])
        figure.colorbar(axes[2].images[0], ax=axes[2])
        figure.colorbar(axes[3].images[0], ax=axes[3])
        figure.tight_layout()
        figure.savefig(filename + ".png")
        # Save the matrices as numpy arrays for later analysis
        np.save(filename + "_context_to_vs.npy", context_to_vs)
        np.save(filename + "_vs_weights.npy", vs_weights)
        np.save(filename + "_selected_vs_weights.npy", selected_vs_weights)
        np.save(filename + "_averaged_vs_weights.npy", averaged_vs_weights)
        plt.close(figure)

    

    

    def create_training_variables(self, config: MORMForClassificationConfig, model_device: str) -> None:
        training_variables_dtype = th.float32
        self.training_variables = MORMTrainingVariables(n_values=config.num_values, initial_lambda=1.0,
                                                        device=model_device, dtype=training_variables_dtype, grounding_loss_tendency_update_ratio=config.grounding_loss_tendency_update_ratio,
                                                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                                                        update_tendencies_every_n_steps=config.update_tendencies_every_n_steps,
                                                        use_validation_for_tendencies=config.use_validation_for_tendencies,
                                                        use_metrics_or_losses=config.use_metrics_or_losses_for_lagrange_updates,
                                                        use_exponential_moving_average_or_optimum_targets=config.use_exponential_moving_average_or_optimum_targets,
                                                        grad_on_only_worst_value=config.grad_on_only_worst_value,
                                                        zero_constraint=config.zero_constraint,
                                                        lambda_decay=config.lambda_decay
                                                        )
        self._set_train_mode(train_mode=True)
        

        self.loss_function = partial(parse_loss_function(
            config), training_variables=self.training_variables, config=config)

    def construct_value_system_layer(self, config: MORMForClassificationConfig, device, dtype):
        if ContextImplementations(config.context_implementation) == ContextImplementations.NO_CONTEXT:
            return ConvexAlignmentLayer(
                config.num_values, 1, device=device, dtype=dtype)
        elif ContextImplementations(config.context_implementation) == ContextImplementations.BASIC:
            return BasicCtxDependentAlignmentLayer(
                input_shape=config.input_size_vs,
                #detach_context_selection_for_value_system_selection=config.detach_context_selection_for_value_system_selection,
                detach_vs_selection_for_value_system_weight_training=config.detach_vs_selection_for_value_system_weight_training,
                weight_initialization = config.vs_weight_initialization,
                num_contexts=config.max_contexts,
                num_value_systems=config.max_value_systems,
                ctx_hidden_sizes=config.vs_layer_hidden_sizes,
                ctx_intermediate_activation=config.vs_layer_intermediate_activation,
                num_values=config.num_values, dropout=config.vs_layer_dropout, device=device, dtype=dtype)
        elif ContextImplementations(config.context_implementation) == ContextImplementations.BASIC_SMOOTH:
            return BasicSmoothCtxDependentAlignmentLayer(
                input_shape=config.input_size_vs,
                #detach_context_selection_for_value_system_selection=config.detach_context_selection_for_value_system_selection,
                detach_vs_selection_for_value_system_weight_training=config.detach_vs_selection_for_value_system_weight_training,
                weight_initialization = config.vs_weight_initialization,
                num_contexts=config.max_contexts,
                num_value_systems=config.max_value_systems,
                ctx_hidden_sizes=config.vs_layer_hidden_sizes,
                ctx_intermediate_activation=config.vs_layer_intermediate_activation,
                num_values=config.num_values, dropout=config.vs_layer_dropout, device=device, dtype=dtype)
        elif ContextImplementations(config.context_implementation) == ContextImplementations.GMM:
            
            return BasicGmmCtxDependentAlignmentLayer(
                input_shape=config.input_size_vs,
                detach_context_selection_for_value_system_selection=config.detach_context_selection_for_value_system_selection,
                detach_vs_selection_for_value_system_weight_training=config.detach_vs_selection_for_value_system_weight_training,
                weight_initialization = config.vs_weight_initialization,
                num_contexts=config.max_contexts,
                num_value_systems=config.max_value_systems,
                ctx_hidden_sizes=config.vs_layer_hidden_sizes,
                ctx_intermediate_activation=config.vs_layer_intermediate_activation,
                num_values=config.num_values, dropout=config.vs_layer_dropout, device=device, dtype=dtype)
        
        elif ContextImplementations(config.context_implementation) == ContextImplementations.BASIC_HARSH:
            return BasicHarshCtxDependentAlignmentLayer(
                input_shape=config.input_size_vs,
                #detach_context_selection_for_value_system_selection=config.detach_context_selection_for_value_system_selection,
                detach_vs_selection_for_value_system_weight_training=config.detach_vs_selection_for_value_system_weight_training,
                weight_initialization = config.vs_weight_initialization,
                num_contexts=config.max_contexts,
                num_value_systems=config.max_value_systems,
                ctx_hidden_sizes=config.vs_layer_hidden_sizes,
                ctx_intermediate_activation=config.vs_layer_intermediate_activation,
                num_values=config.num_values, dropout=config.vs_layer_dropout, device=device, dtype=dtype
            )
        elif ContextImplementations(config.context_implementation) == ContextImplementations.DIRECT_VS:
            return DirectVSCtxDependentAlignmentLayer(
                input_shape=config.input_size_vs,
                #detach_context_selection_for_value_system_selection=config.detach_context_selection_for_value_system_selection,
                #detach_vs_selection_for_value_system_weight_training=config.detach_vs_selection_for_value_system_weight_training,
                #weight_initialization = config.vs_weight_initialization,
                ctx_hidden_sizes=config.vs_layer_hidden_sizes,
                ctx_intermediate_activation=config.vs_layer_intermediate_activation,
                num_values=config.num_values, dropout=config.vs_layer_dropout, device=device, dtype=dtype)
        else:
            raise NotImplementedError(f"This type of context implementation is not implemented yet. {config.context_implementation}")
    def create_reward_networks(self, config: MORMForClassificationConfig, model_device: str):
        if len(config.hidden_sizes) > 0:
                # This constructs one NN per value to avoid that changing one value loss parameter chagnes also affect others.
            constructor = self.construct_reward_head
        else:
                # In this case, the parameters of different value dimensions do not conflict.
            constructor = partial(
                    self.construct_value_layer, n_outputs=self.num_values, add_normalization=config.layer_normalization != 'none')
            
        self.reward_heads = constructor(config, input_size=config.input_size, model_device=model_device)
        print(f"Reward heads: {self.reward_heads}")
        
        if config.use_ideal_grounding_model:
            self.reward_heads_ideal = constructor(config, input_size=config.input_size, model_device=model_device)
        self.value_system_layer =  self.construct_value_system_layer(config, model_device, dtype=self.reward_heads.parameters().__next__().dtype) #config.input_size_vs
        print(self.value_system_layer)
        self.reward_heads_ideal = None if not config.use_ideal_grounding_model else self.reward_heads_ideal
        # self.score_weight_head: ConvexAlignmentLayer = ConvexAlignmentLayer(num_values, 1)
        
    def construct_value_layer(self, config: MORMForClassificationConfig, input_size: int, model_device=None, n_outputs: int = 1, add_normalization: bool = False) -> nn.Sequential:
        
        model_dtype = self._resolve_torch_dtype(config.dtype)
        
        layers = construct_layers(input_dim=input_size, 
                                  n_outputs=n_outputs, 
                                  hidden_sizes=config.hidden_sizes, 
                                  intermediate_activation=config.value_layer_intermediate_activation, 
                                  device=model_device, 
                                  dtype=model_dtype, 
                                  dropout=config.value_layer_dropout,
                                  final_activation=config.value_layer_final_activation, 
                                  final_activation_kwargs={})
        if add_normalization:
            layers.append(self.construct_value_normalization(config, model_device))
        return nn.Sequential(*layers)

    def construct_value_normalization(self, config: MORMForClassificationConfig, model_device=None) -> nn.Module:
        # Normalize across value dimensions to keep reward channels on a comparable scale.
        
        model_dtype = self._resolve_torch_dtype(config.dtype)

        if config.layer_normalization == 'LayerNorm':
            return nn.LayerNorm(config.num_values, dtype=model_dtype, device=model_device)
        if config.layer_normalization == 'BatchNorm':
            return nn.BatchNorm1d(config.num_values, dtype=model_dtype, device=model_device)
        if config.layer_normalization == 'none':
            return nn.Identity()
        raise ValueError(
            f"Unsupported normalization: {config.layer_normalization}")

    def construct_reward_head(self, config: MORMForClassificationConfig, input_size: int, model_device=None) -> MultiValueRewardHead:
        value_heads = nn.ModuleList([
            self.construct_value_layer(config, input_size, model_device)
            for _ in range(config.num_values)
        ])
        normalization = self.construct_value_normalization(config, model_device)
        return MultiValueRewardHead(value_heads=value_heads, normalization=normalization, optimized_head_indices=config.loss_func_type_kwargs.get('value_indices', None))



    @staticmethod
    def _resolve_torch_dtype(dtype_value: Any) -> th.dtype:
        if isinstance(dtype_value, th.dtype):
            return dtype_value
        if dtype_value is None:
            return th.float32
        dtype_name = str(dtype_value).replace("torch.", "")
        if not hasattr(th, dtype_name):
            raise ValueError(f"Unsupported dtype '{dtype_value}' in config.")
        resolved = getattr(th, dtype_name)
        if not isinstance(resolved, th.dtype):
            raise ValueError(
                f"Resolved dtype '{dtype_name}' is not a torch.dtype.")
        return resolved

    @staticmethod
    def _module_device(module: nn.Module) -> th.device:
        return next(module.parameters()).device

    @staticmethod
    def _module_dtype(module: nn.Module) -> th.dtype:
        return next(module.parameters()).dtype
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        # TODO: Apparetly this is much faster. See https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        set_to_none = True
        super().zero_grad(set_to_none)
        if self.reward_heads is not None:
            self.reward_heads.zero_grad(set_to_none)
        if self.use_ideal_grounding_model:
            self.reward_heads_ideal.zero_grad(set_to_none)
        if self.value_system_layer is not None:
            self.value_system_layer.zero_grad(set_to_none)
        self.training_variables.zero_grad(set_to_none)
    def train(self, mode: bool = True):
        self._set_train_mode(train_mode=mode)
        self.forward_ideal_grounding = mode and self.use_ideal_grounding_model

        return super().train(mode)
    
    def set_value_system_layer(self, layer: nn.Module):
        self.value_system_layer = layer

    def grounding_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        params = []
        if self.config.loss_management.should_apply_grad_on_grounding_parameters():
            if self.reward_heads is not None:
                params.extend(self.reward_heads.parameters(recurse=recurse))
        return iter(params)
    
    def context_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        params = []
        if self.config.loss_management.should_apply_grad_on_context_parameters():
            if self.reward_heads is not None:
                params.extend(self.value_system_layer.context_parameters())
        return iter(params)

    def value_system_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        params = []
        if self.config.loss_management.should_apply_grad_on_value_system_weights():
            params.extend(
                    self.value_system_layer.value_system_parameters())
        return iter(params)
    
    def score(self, grounding_features, score_mode='normal', vs_features=None) -> th.Tensor:
        if score_mode == 'ideal' and self.use_ideal_grounding_model:
            reward_heads = self.reward_heads_ideal
            raise ValueError(f"Unexpected loss function type {self.config.loss_func_type} that does not fit into any grounding loss category, cannot determine whether to apply grad on grounding parameters or not.")
        else:
            reward_heads = self.reward_heads

        if self.config.loss_management.should_apply_grad_on_grounding_parameters():
            rewards = reward_heads(grounding_features)
            assert rewards.shape[-1] == self.num_values
        else:
            #raise ValueError(f"Unexpected loss function type {self.config.loss_func_type} that does not fit into any grounding loss category, cannot determine whether to apply grad on grounding parameters or not.")
            with th.no_grad():
                rewards = reward_heads(grounding_features)

        if self.value_system_layer is not None:
            if self.config.loss_management.should_apply_grad_on_value_system_weights():
                vs_reward, other = self.value_system_layer.forward(rewards, hidden_state=vs_features)
                assert vs_reward.shape[0] == len(rewards)
            else:
                #raise ValueError(f"Unexpected loss function type {self.config.loss_func_type} that does not fit into any grounding loss category, cannot determine whether to apply grad on grounding parameters or not.")
                with th.no_grad():
                    vs_reward, other = self.value_system_layer.forward(rewards, hidden_state=vs_features)
                
            all_rewards = th.cat([rewards, vs_reward], dim=-1)
        else:
            all_rewards = rewards
        return all_rewards, other
    
    def parameters(self, recurse: bool = True) -> Iterator[th.nn.Parameter]:
        # Condition 2: Reward heads
        if self.config.loss_management.should_apply_grad_on_grounding_parameters():
            
            yield from self.grounding_parameters()
            
        # Condition 3: Value system layer
        if self.config.loss_management.should_apply_grad_on_value_system_weights():
            
            yield from self.value_system_parameters(recurse=recurse)
        
        if self.config.loss_management.should_apply_grad_on_context_parameters():
            
            yield from self.context_parameters(recurse=recurse)
        # yield from self.training_variables.parameters(recurse=recurse)
        
        # Condition 4: Ideal grounding model
        if self.use_ideal_grounding_model and self.config.loss_management.should_apply_grad_on_grounding_parameters():
            
            yield from self.reward_heads_ideal.parameters(recurse=recurse)

    def base_forward(self, *args, **kwargs) -> th.Tensor:
        grounding_features = kwargs.pop(self.grounding_features_name)
        
        vs_features = kwargs.pop(self.vs_features_name, None)
       
        all_rewards, other = self.score(grounding_features, vs_features=vs_features)
        assert all_rewards.shape[-1] == self.num_values + 1, f"Shape: {all_rewards.shape}"
        if self.forward_ideal_grounding:
            grounding_ideal = self.reward_heads_ideal(grounding_features)
            return self.classifier_output_class_ideal(logits=all_rewards, ideal_logits=grounding_ideal, other=other)
        else:
            return self.classifier_ouput_class(logits=all_rewards, other=other)
    
    def forward(self, *args, **kwargs):
        """perfect_debug_forward = True #DEBUG ONLY.
        if perfect_debug_forward: 
            #print(kwargs.keys())
            grfeatures = kwargs.pop(self.grounding_features_name)
            all_rewards = self.score(grfeatures)
            logits = kwargs.pop("labels", None)
            missing = get_missing_rating_mask(logits)
            logits = logits  + th.tensor([0.0], requires_grad=True) + all_rewards*0.1 - all_rewards*0.1 # Just to have a tensor that requires grad for testing.  
            logits = logits.masked_fill(missing, float("-inf"))
            return SequenceClassifierOutputWithPastAndOthers(logits=logits)"""
        
        return self.base_forward(*args,**kwargs)
        
class MORMForSequenceClassification(MORMForClassification):
    config_class = MORMForClassificationConfig
    base_model_prefix = "full_model"
    supports_gradient_checkpointing = True
    config : MORMForClassificationConfig

    classifier_ouput_class = SequenceClassifierOutputWithPastAndOthers 
    classifier_ouput_class_ideal = SequenceClassifierOutputWithPastAndIdeal

    @property
    def grounding_features_name(self) -> str:
        return 'embedding'

    @property
    def vs_features_name(self) -> Optional[str]:
        return CONTEXT_EMBEDDING_FEATURE_NAME
    
    
    """def train_initialization(self, train_subdataset: Dataset):
        pass"""

    def _build_base_model_from_config(self, config: MORMForClassificationConfig) -> AutoModelForSequenceClassification:
        """
        Build the base model from the config, ensuring that it is a AutoModelForSequenceClassification and extracting necessary information for reward head construction.
        """
        model_name_or_path = getattr(config, "base_model_name_or_path", None)
        if not model_name_or_path:
            raise ValueError(
                "Config is missing base_model_name_or_path. "
                "Set this field when constructing MORMForSequenceClassificationConfig."
            )

        torch_dtype = self._resolve_torch_dtype(
            getattr(config, "dtype", th.float32))
        if torch_dtype == "float16":
            torch_dtype = th.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = th.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = th.float32
        trust_remote_code = bool(
            getattr(config, "base_model_trust_remote_code", True))
        base_cfg = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "config": base_cfg,
            "trust_remote_code": trust_remote_code,
            "dtype": torch_dtype,
        }

        if bool(getattr(config, "use_base_model_heads", False)):
            return AutoModelForSequenceClassification.from_config(**model_kwargs)

        base_cfg.num_labels = int(getattr(config, "base_model_num_labels", 1))
        base = AutoModelForSequenceClassification.from_config(
            **model_kwargs).base_model
        if hasattr(base, "base_model"):
            base = base.base_model
        return base

    def parameters(self, recurse: bool = True) -> Iterator[th.nn.Parameter]:
        # Override parameters to only return reward head and value system parameters for optimization.
        # Condition 1: Base model heads
        if self.use_base_model_heads and self.config.loss_management.should_apply_grad_on_grounding_or_value_system_params():
            yield from self.full_model.parameters(recurse=recurse)

        yield from MORMForClassification.parameters(self, recurse=recurse)
    
    

    def _select_reward_indices(self, rewards: th.Tensor) -> th.Tensor:
        indices = self.base_model_reward_head_indices
        if indices is None:
            return rewards
        if len(indices) == 0:
            raise ValueError(
                "base_model_reward_head_indices cannot be an empty list when using base model outputs.")
        idx = th.tensor(indices, device=rewards.device, dtype=th.long)
        if rewards.ndim == 1:
            return rewards[idx]
        return th.index_select(rewards, dim=-1, index=idx)

    def _extract_logits_from_base_output(self, output: Any) -> th.Tensor:
        rewards_attr = self.base_model_rewards_attr_name
        score_attr = self.base_model_score_attr_name

        rewards = getattr(output, rewards_attr, None)
        score = getattr(output, score_attr, None)
        if score is None:
            raise ValueError(
                f"Base model output does not have score attribute '{score_attr}'.")

        if score.ndim == 0:
            score = score.unsqueeze(0).unsqueeze(-1)
        elif score.ndim == 1:
            score = score.unsqueeze(-1)
        elif score.ndim > 2:
            score = score.reshape(score.shape[0], -1)

        if rewards is None or self.base_model_reward_head_indices == "use_base_model_value_system_module_name":

            rewards = score.repeat(1, self.num_values)
        else:
            rewards = self._select_reward_indices(rewards)

        assert rewards.shape[0] == score.shape[
            0], f"Batch size of rewards and score must match, but got {rewards.shape[0]} and {score.shape[0]}"
        assert rewards.shape[-1] == self.num_values, f"Expected rewards to have last dimension {self.num_values}, but got shape {rewards.shape}"
        assert score.shape[
            -1] == 1, f"Expected score to have last dimension 1 after processing, but got shape {score.shape}"

        return th.cat([rewards, score], dim=-1)

    @staticmethod
    def infer_model_inputs_sizes(base_model: AutoModelForSequenceClassification) -> Tuple[int, Optional[int]]:
        # SequenceClassification wrappers often expose the classifier head input width here.
        if hasattr(base_model, "score") and hasattr(base_model.score, "in_features"):
            return int(base_model.score.in_features), int(base_model.score.in_features)

        cfg = getattr(base_model, "config", None)
        for attr in ("hidden_size", "d_model", "n_embd", "dim"):
            value = getattr(cfg, attr, None)
            if value is not None:
                return int(value), int(value)

        # Last fallback for models with custom configs but standard embedding modules.
        input_emb = base_model.get_input_embeddings() if hasattr(
            base_model, "get_input_embeddings") else None
        if input_emb is not None and hasattr(input_emb, "embedding_dim"):
            return int(input_emb.embedding_dim), int(input_emb.embedding_dim)
        if input_emb is not None and hasattr(input_emb, "weight"):
            return int(input_emb.weight.shape[-1]), int(input_emb.weight.shape[-1])

        raise ValueError(
            "Could not infer hidden size for reward heads. Expected one of: "
            "score.in_features, config.hidden_size/d_model/n_embd/dim, or input embedding width."
        )

    


    def init_networks(self, config: MORMForClassificationConfig, *args, **kwargs) -> None:
        base_model = kwargs.get('base_model', None)
        if base_model is None:
            base_model = self._build_base_model_from_config(config)
            
        print(f"Base model loaded: {base_model.__class__.__name__}")
        self.full_model = base_model
        self.supports_gradient_checkpointing = hasattr(
            self.full_model, "gradient_checkpointing_enable")
        model_device = self._module_device(self.full_model)
        input_size, input_size_vs = MORMForSequenceClassification.infer_model_inputs_sizes(base_model)
        assert input_size == config.input_size
        assert input_size_vs == config.input_size_vs
        
        # In base-model mode, consume reward/score attributes from base model outputs.
        if self.use_base_model_heads:
            if not self.base_model_rewards_attr_name:
                raise ValueError(
                    "use_base_model_heads=True requires base_model_reward_heads_module_name to specify the output reward attribute name.")
            if not self.base_model_score_attr_name:
                raise ValueError(
                    "use_base_model_heads=True requires base_model_value_system_module_name to specify the output score attribute name.")
            self.reward_heads = None
            self.value_system_layer = None
            self.reward_heads_ideal = None
        else:
            
            self.create_reward_networks(config, model_device)
        
        # if config.training_variables_dtype == "float32" else th.float16 if config.training_variables_dtype == "float16" else self._resolve_torch_dtype(config.training_variables_dtype)
        self.create_training_variables(config, model_device)

    def _set_train_mode(self, train_mode: bool = True) -> None:
        # Freeze pretrained weights and train only the custom reward/value-system heads.
        possibly_change_train_mode_in_base_model = self.use_base_model_heads and self.config.loss_management.should_apply_grad_on_grounding_or_value_system_params()
        train_mode_base_model = False
        if possibly_change_train_mode_in_base_model:
            train_mode_base_model = train_mode
        
        self.full_model.train(train_mode_base_model)

        MORMForClassification._set_train_mode(self,train_mode)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.full_model, "is_gradient_checkpointing", False))


    def grounding_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        params = []
        if self.config.loss_management.should_apply_grad_on_grounding_parameters():
            if self.reward_heads is not None:
                params.extend(self.reward_heads.parameters(recurse=recurse))
            elif self.use_base_model_heads:
                # In base model mode, we assume all parameters require grad, but we only want to return the reward head parameters for optimization.
                params.extend(self.full_model.parameters(recurse=recurse))
        return iter(params)

    def value_system_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        params = []
        if self.config.loss_management.should_apply_grad_on_value_system_weights():
            if self.value_system_layer is not None:
                params.extend(
                    self.value_system_layer.parameters(recurse=recurse))
            elif self.use_base_model_heads:
                # In base model mode, we assume all parameters require grad, but we only want to return the reward head parameters for optimization.
                params.extend(self.full_model.parameters(recurse=recurse))
        return iter(params)

    

    

    def forward(self, *args, **kwargs) -> SequenceClassifierOutputWithPastAndOthers | SequenceClassifierOutputWithPastAndIdeal:
        

        if self.use_base_model_heads:
            kwargs.pop("labels", None)
            kwargs.pop("embedding", None)
            kwargs.pop(CONTEXT_EMBEDDING_FEATURE_NAME, None)

            kwargs.pop("return_loss", None)
            # TODO this might not work with other models
            kwargs.pop("num_items_in_batch", None)
            base_output = self.full_model(*args, **kwargs, return_dict=True)
            pooled_logits = self._extract_logits_from_base_output(base_output)
            del base_output
            # print("LABELS SHAPE", labels.shape)

            return self.classifier_ouput_class(
                logits=pooled_logits,
                #past_key_values=getattr(base_output, "past_key_values", None),
                #hidden_states=getattr(base_output, "hidden_states", None),
                #attentions=getattr(base_output, "attentions", None),
            )

        if self.grounding_features_name in kwargs and (self.vs_features_name is None or self.vs_features_name in kwargs.keys() ):
            return self.base_forward(*args,**kwargs)
        else:
            # sq = GenericForSequenceClassification.forward(self, *args, **kwargs)
            kwargs["score_mode"] = "normal"
            sq = self.generic_forward(*args, **kwargs)
            if self.forward_ideal_grounding:
                # = GenericForSequenceClassification.forward(self, *args, **kwargs)
                kwargs["score_mode"] = "ideal"
                ideal_sq = self.generic_forward(*args, **kwargs)
                kwargs["score_mode"] = "normal"
                return self.classifier_ouput_class_ideal(logits=sq.logits, ideal_logits=ideal_sq.logits)
            else:
                return sq

    def generic_forward( # Same as GenericForSequenceClassification forward, (version 5.3.0 transformers)
        self,
        input_ids: th.LongTensor | None = None,
        attention_mask: th.Tensor | None = None,
        position_ids: th.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        labels: th.LongTensor | None = None,
        use_cache: bool | None = None,
        score_mode = "normal",
        **kwargs: Dict[str, Any],
    ) -> SequenceClassifierOutputWithPastAndOthers:
        transformer_outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits, other = self.score(hidden_states, score_mode=score_mode) # Change here.

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(
                logits.device, th.int32)
            token_indices = th.arange(
                input_ids.shape[-1], device=logits.device, dtype=th.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[th.arange(
            batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:

            loss = self.loss_function(
                logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config, training_variables=self.training_variables) # slight change here.

        return self.classifier_ouput_class(
            loss=loss,
            other=other,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
