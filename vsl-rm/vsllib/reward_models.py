
from collections.abc import Iterator
from dataclasses import dataclass

import enum
from functools import partial
import re
import time
from typing import Any, Callable, Dict, Literal, Optional, Unpack

import numpy as np
import torch as th
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.utils import logging
from transformers.cache_utils import Cache

logger = logging.get_logger(__name__)

#from transformers.modeling_layers import GenericForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast

from vsllib.defines import NO_RATING_MASK
from vsllib.utils import MORMTrainingVariables, print_tensor_and_grad_fn
class LinearAlignmentLayer(th.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None, data=None, n_values=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.linear_bias = bias
        self.n_values = in_features if n_values is None else n_values # TODO: This might not be the case in future works...

        with th.no_grad():
            state_dict = self.state_dict()
            random_vector = th.rand_like(state_dict['weight'])
            state_dict['weight'] = th.nn.functional.sigmoid(
                state_dict['weight']) * random_vector

            self.load_state_dict(state_dict)

    

    def forward(self, input: th.Tensor) -> th.Tensor:
        w_bounded, b_bounded = self.get_alignment_layer()
        assert w_bounded.dtype == self.weight.dtype, f"Expected w_bounded dtype {self.weight.dtype}, but got {w_bounded.dtype}"
        assert w_bounded.device == self.weight.device, f"Expected w_bounded device {self.weight.device}, but got {w_bounded.device}"
        output = th.nn.functional.linear(input, w_bounded)
        assert input.shape[-1] == self.n_values, f"Expected output shape to have last dimension {self.n_values}, but got {output.shape}"

        return output

    def get_alignment_layer(self):
        w_bounded = self.weight
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))
        b_bounded = 0.0
        if self.linear_bias:
            b_bounded = self.bias
        return w_bounded, b_bounded

    def get_weights(self):
        with th.no_grad():
            w_bounded, b_bounded = self.get_alignment_layer()
            return w_bounded.detach().clone().view(-1).cpu().tolist()
    """def copy(self):
        with th.no_grad():
            new = self.__class__(in_features=self.in_features, out_features=self.out_features, bias=self.linear_bias, device=self.weight.device, dtype=self.weight.dtype)
            new.load_state_dict(deepcopy(self.state_dict()))
        return new"""


class ConvexAlignmentLayer(LinearAlignmentLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=th.float32, data=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype, data)
        self.set_weights([1/self.weight.shape[1] for _ in range(self.weight.shape[1])])

    
    def set_weights(self, weights: tuple):
        with th.no_grad():
            # Convert to tensor with same dtype and device as self.weight
            pure_w = th.tensor(weights, dtype=self.weight.dtype, device=self.weight.device)
            new_weights = th.log(pure_w+1e-8)
            # Reshape to match weight shape
            new_weights = new_weights.view_as(self.weight)
            # Ensure requires_grad matches previous setting
            new_weights.requires_grad = self.weight.requires_grad
            # Update state dict in place
            self.load_state_dict({'weight': new_weights}, strict=False)
            
            assert th.allclose(pure_w, th.nn.functional.softmax(self.weight, dim=1, dtype=self.weight.dtype)), f"{new_weights} vs {th.nn.functional.softmax(self.weight, dim=1, dtype=self.weight.dtype)}"

    def get_alignment_layer(self):
        w_bounded = th.nn.functional.softmax(self.weight, dim=1, dtype=self.weight.dtype)
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))
        b_bounded = 0.0
        
        return w_bounded, b_bounded


from transformers.configuration_utils import PretrainedConfig
from typing import List

class MOLossFunctions(enum.Enum):
    DEFAULT = "DEFAULT"
    ONLY_GROUNDING = "ONLY_GROUNDING"
    ONLY_VALUE_SYSTEM = "ONLY_VALUE_SYSTEM"
    ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS = "ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS"
    ONLY_VALUES_IN_KWARGS = "ONLY_VALUES_IN_KWARGS"
    EVALUATION_ONLY = "EVALUATION_ONLY"

    

class MOLossFunctionsCategories():
    REQUIRES_GRAD_ON_EVERYTHING = [MOLossFunctions.DEFAULT]
    REQUIRES_GRAD_ON_VALUE_SYSTEM_WEIGHTS = [MOLossFunctions.ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS, MOLossFunctions.ONLY_VALUE_SYSTEM, MOLossFunctions.DEFAULT]
    REQUIRES_GRAD_ON_SOME_GROUNDING = [MOLossFunctions.ONLY_VALUES_IN_KWARGS]
    REQUIRES_GRAD_ON_ALL_GROUNDING = [MOLossFunctions.ONLY_GROUNDING, MOLossFunctions.ONLY_VALUE_SYSTEM, MOLossFunctions.DEFAULT]
    REQUIRES_GRAD_ON_SOME_OR_ALL_GROUNDING = REQUIRES_GRAD_ON_SOME_GROUNDING + REQUIRES_GRAD_ON_ALL_GROUNDING
    
    REQUIRES_GRAD_ON_VALUE_SYSTEM_WEIGHTS_ALONE = [MOLossFunctions.ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS]

    
    NEEDS_NO_GRAD_ON_VALUE_SYSTEM_WEIGHTS = [MOLossFunctions.ONLY_GROUNDING, MOLossFunctions.ONLY_VALUES_IN_KWARGS, MOLossFunctions.EVALUATION_ONLY]

    NEEDS_NO_GRAD_EVER = [MOLossFunctions.EVALUATION_ONLY]

    NEEDS_NO_GRAD_ON_ALL_GROUNDINGS = REQUIRES_GRAD_ON_VALUE_SYSTEM_WEIGHTS_ALONE + NEEDS_NO_GRAD_EVER

    NEEDS_NO_GRAD_ON_LAGRANGE_MULTIPLIERS = [MOLossFunctions.ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS, MOLossFunctions.EVALUATION_ONLY, MOLossFunctions.ONLY_VALUE_SYSTEM]

class MORMForSequenceClassificationConfig(PretrainedConfig):
    model_type = "morm_for_sequence_classification"
    has_no_defaults_at_init = True



    
        
    def __init__(
        self,
        pad_token_id: int,
        num_values: int = 3,
        hidden_sizes: list[int] = [1024,1024,1024],
        value_layer_dropout: float = 0.1,
        value_layer_intermediate_activation: str = "ReLU",
        value_layer_final_activation: str = "none",
        layer_normalization: Literal['LayerNorm', 'BatchNorm', 'none'] = 'LayerNorm',
        reward_diff_threshold: float = 50.0,
        assume_qualitative_labels: bool = False,
        check_undefined_label: bool = True,
        grounding_loss_tendency_update_ratio: float = 0.001,
        rew_center_coefficient: float = 0.0,
        gradient_accumulation_steps: int = 2,
        metrics_accumulation_steps: int = 2,
        use_metrics_or_losses_for_lagrange_updates: str = "metrics",
        grad_on_only_worst_value: bool = False,
        zero_constraint: bool = True,
        lambda_decay: float = 0.0,
        use_ideal_grounding_model: bool = False,
        dtype: th.Type = th.float16,
        use_base_model_heads: bool = False,
        base_model_reward_heads_module_name: str = None,
        base_model_value_system_module_name: str = None,
        base_model_reward_head_indices: list = None,
        loss_func_type: str = MOLossFunctions.DEFAULT,
        loss_func_kwargs: dict = None,
        **kwargs,
    ):
        assert num_values > 0, "num_values must be greater than 0"
        assert len(hidden_sizes) > 0, "hidden_sizes must be a non-empty list"

        if value_layer_intermediate_activation not in ['ReLU', 'SiLU', 'Tanh', 'Softplus']:
             raise ValueError(f"value_layer_intermediate_activation must be one of 'ReLU', 'SiLU', 'Tanh', 'Softplus', but got {value_layer_intermediate_activation}")
        if value_layer_final_activation not in ['ReLU', 'SiLU', 'Tanh', 'Softplus', 'none']:
             raise ValueError(f"value_layer_final_activation must be one of 'ReLU', 'SiLU', 'Tanh', 'Softplus', 'none', but got {value_layer_final_activation}")

        if layer_normalization not in ['LayerNorm', 'BatchNorm', 'none']:
            raise ValueError(f"layer_normalization must be one of 'LayerNorm', 'BatchNorm', 'none', but got {layer_normalization}")
        

        default_id2label = {
            index: f"VALUE_{index}" for index in range(num_values)
        }
        default_id2label[num_values] = "VALUE_SYSTEM"
        id2label = kwargs.pop("id2label", default_id2label)
        label2id = kwargs.pop("label2id", {label: index for index, label in id2label.items()})
        self.pad_token_id = pad_token_id

        super().__init__(num_labels=num_values + 1, id2label=id2label, label2id=label2id, **kwargs)
        
        self.num_values = num_values
        self.hidden_sizes = hidden_sizes
        self.value_layer_dropout = value_layer_dropout
        self.value_layer_intermediate_activation = value_layer_intermediate_activation
        self.value_layer_final_activation = value_layer_final_activation
        self.reward_diff_threshold = reward_diff_threshold
        self.assume_qualitative_labels = assume_qualitative_labels
        self.check_undefined_label = check_undefined_label
        self.grounding_loss_tendency_update_ratio=grounding_loss_tendency_update_ratio
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.metrics_accumulation_steps = metrics_accumulation_steps
        self.use_metrics_or_losses_for_lagrange_updates = use_metrics_or_losses_for_lagrange_updates
        self.grad_on_only_worst_value = grad_on_only_worst_value
        self.zero_constraint = zero_constraint
        self.rew_center_coefficient = rew_center_coefficient
        self.dtype = dtype
        self.lambda_decay = lambda_decay
        self.use_ideal_grounding_model = use_ideal_grounding_model
        self.layer_normalization = layer_normalization
        self.use_base_model_heads = use_base_model_heads
        self.base_model_reward_heads_module_name = base_model_reward_heads_module_name
        self.base_model_value_system_module_name = base_model_value_system_module_name
        self.loss_func_type = MOLossFunctions(loss_func_type)
        self.loss_func_type_kwargs = loss_func_kwargs if loss_func_kwargs is not None else {}
        self.base_model_reward_head_indices = base_model_reward_head_indices if base_model_reward_head_indices is not None else list(range(num_values))

LossFuncType = Callable[[th.Tensor, th.Tensor, th.Tensor, th.Tensor, MORMForSequenceClassificationConfig, MORMTrainingVariables, Any], th.Tensor]
def parse_loss_function(config: MORMForSequenceClassificationConfig) -> LossFuncType:
        return mo_loss_function

def accuracy_rewards_labels(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, epsilon=1.0e-2, threshold=50.0, assume_qualitative_labels=False, check_undefined_label=True, missing_mask=None, assume_torch=True) -> th.Tensor:
    logits, targets, others = reward_pairs_and_scores_to_logits_and_targets(reward1, reward2, scores1, scores2, reward_diff_threshold=threshold, assume_qualitative_labels=assume_qualitative_labels, check_undefined_label=check_undefined_label, missing_mask=missing_mask, assume_torch=assume_torch)
    return accuracy_logits(logits, targets, epsilon=epsilon, missing_mask=others.get("missing_mask", missing_mask), assume_torch=assume_torch)
def accuracy_logits(logits: th.Tensor, target_probs: th.Tensor, epsilon=1.0e-2, missing_mask=None, assume_torch=True) -> th.Tensor:
    with th.no_grad():

        rep_mask1 = (logits > 0) & (target_probs > 0.5)
        rep_mask2 = (logits < 0) & (target_probs < 0.5)
        rep_mask3 = (target_probs == 0.5) & ((logits <= epsilon) & (logits >= -epsilon))
        all_defined_cases = ~get_missing_rating_mask(target_probs)  if missing_mask is None else ~missing_mask
        mask = (rep_mask1 | rep_mask2 | rep_mask3 ) & all_defined_cases # IT SHOULD BE OR BECAUSE INDEFINITE IS OK!!!!
        
        
        if assume_torch:
            
            accuracy = mask.float().sum(dim=0).div_((all_defined_cases).float().sum(dim=0))
        else:
            accuracy = mask.astype(float).sum(axis=0)/(all_defined_cases).astype(float).sum(axis=0)

    if len(logits.shape) >= 2:
        assert accuracy.shape == (logits.shape[-1],), f"Expected loss shape {(logits.shape[-1],)}, got {accuracy.shape}"
    else:
        assert accuracy.shape == (), f"Expected loss shape (), got {accuracy.shape}"
    return accuracy


def logits_BT(x: th.Tensor, y: th.Tensor, threshold=50.0, check_undefined_label=False, missing_mask=None, assume_torch=True) -> th.Tensor:
    # print("DIFF", th.max(x - y))
    returns_diff = x - y
    if check_undefined_label:
        if missing_mask is None:
            missing_mask = get_missing_rating_mask(x, y)
            
        

        """if th.any(missing_mask).item():
            
            both_missing_mask = (x == NO_RATING_MASK) & (y == NO_RATING_MASK)
            if th.any(both_missing_mask).item():
                x.masked_fill_(both_missing_mask, 0.0)
                y.masked_fill_(both_missing_mask, 0.0)

            # If one side is missing, copy the other side so the pair becomes neutral.
            x_missing_mask = x == NO_RATING_MASK
            if th.any(x_missing_mask).item():
                x[x_missing_mask] = y[x_missing_mask]

            y_missing_mask = y == NO_RATING_MASK
            if th.any(y_missing_mask).item():
                y[y_missing_mask] = x[y_missing_mask]"""

        # IMPORTANT: Because we use a bradley terry model, the scale does not matter. The model will learn indifference, perhaps force it... This is not ideal...
    
    if assume_torch:
        returns_diff = th.clip(returns_diff, -threshold, threshold)
    else:
        returns_diff = np.clip(returns_diff, -threshold, threshold)
    if missing_mask is not None:
        returns_diff[missing_mask] = NO_RATING_MASK
    if assume_torch:
        assert th.max(returns_diff[~missing_mask]) <= threshold and th.min(returns_diff[~missing_mask]) >= - \
            threshold, f"Clipping failed: max {th.max(returns_diff[~missing_mask])}, min {th.min(returns_diff[~missing_mask])}, threshold {threshold}"
    
    return returns_diff

def get_missing_rating_mask(x_or_probs, y=None):
    if y is not None:
        missing_mask = (x_or_probs == NO_RATING_MASK) | (y == NO_RATING_MASK)
    else:
        missing_mask = (x_or_probs == NO_RATING_MASK)
    return missing_mask


def rewards_and_labels_to_logits_and_targets(logits, labels=None, assume_torch=True, config: MORMForSequenceClassificationConfig = None):
    bsz = logits.size(0)

    jidx = th.arange(0, bsz, 2, device=logits.device)
    kidx = jidx + 1

    rewards_1 = logits[jidx]
    rewards_2 = logits[kidx]

    if labels is not None:
        labels_1 = labels[jidx]
        labels_2 = labels[kidx]
    else:
        labels_1 = None
        labels_2 = None

    logits_new, target_probs, others = reward_pairs_and_scores_to_logits_and_targets(rewards_1, rewards_2, labels_1, labels_2, reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label, assume_torch=assume_torch)
    return logits_new, target_probs, others

def scores_to_target_probs(scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold: int=50.0, assume_qualitative_labels=False, check_undefined_label=True, missing_mask=None, assume_torch=True) -> th.Tensor:
    with th.no_grad():
        
        if assume_qualitative_labels:
            assert th.max(scores1) == 1.0
            assert th.min(scores1) == 0.0
            target_probs: th.Tensor = scores1# model probability of first one being preferred.
        else:
            log = logits_BT(scores1, scores2, threshold=reward_diff_threshold, check_undefined_label=check_undefined_label, missing_mask=missing_mask, assume_torch=assume_torch)
            if assume_torch:
                target_probs = th.sigmoid(log)
            else:
                target_probs = 1 / (1 + np.exp(-log))
        if check_undefined_label: 
                # If either score is NO_RATING_MASK, set target_prob to 0.5 (indicating no preference)
                mask = get_missing_rating_mask(scores1, scores2) if missing_mask is None else missing_mask
                if assume_torch:
                    target_probs.masked_fill_(mask, NO_RATING_MASK)
                else:
                    target_probs[mask] = NO_RATING_MASK
    return target_probs


def grounding_loss_logits(logits_p: th.Tensor, target_probs_p: th.Tensor, rew_sum: th.Tensor=None, return_metrics: bool=False, check_undefined_label=True, rew_center_coefficient=0.0, missing_mask: th.Tensor=None, no_grad_on_indexes: Optional[list[int]] = None) -> th.Tensor:
    """Multi-objective Cross-entropy loss: target_probs(1,2)*log(exp(r1) / (exp(r1) + exp(r2)))- (1-target_probs(1,2))*log(exp(r2) / (exp(r1) + exp(r2)))"""
    # label = 1: reward1 should be higher.
    # label = 0: reward2 should be higher.
    # label = 0.5: no preference.
    missing_mask = get_missing_rating_mask(target_probs_p) if check_undefined_label and missing_mask is None else missing_mask
    
    if check_undefined_label:
        logits = logits_p.masked_fill(missing_mask, 0.0)
        target_probs = target_probs_p.masked_fill(missing_mask, 0.5)
    else:
        logits = logits_p
        target_probs = target_probs_p
        
    assert target_probs.shape == logits.shape, f"Target probabilities shape {target_probs.shape} does not match logits shape {logits.shape}"	
    assert not th.any(logits.isnan()) and not th.any(logits.isinf()), f"Logits contain NaN or Inf values: {logits}"
    assert not th.any(target_probs.isnan()) and not th.any(target_probs.isinf()), f"Target probabilities contain NaN or Inf values: {target_probs}"	
    assert th.all(target_probs >= 0.0) and th.all(target_probs<= 1.0), f"Target probabilities should be in [0, 1], but got {target_probs}"

    
    if no_grad_on_indexes:
        loss = th.empty_like(logits)
        detached_idx = th.as_tensor(no_grad_on_indexes, device=logits.device, dtype=th.long)
        not_detached_idx = th.tensor([i for i in range(logits.shape[-1]) if i not in no_grad_on_indexes], device=logits.device, dtype=th.long)
        assert detached_idx.numel() > 0, "no_grad_on_indexes should be non-empty when provided"
        assert th.all((detached_idx >= 0) & (detached_idx < logits.shape[-1])).item(), (
            f"no_grad_on_indexes contains invalid indices for last dimension size {logits.shape[-1]}: {no_grad_on_indexes}"
        )
        logits[..., detached_idx] = logits[..., detached_idx].detach().requires_grad_(False)

        """pf = time.perf_counter()
        with th.no_grad():
            loss[..., detached_idx] = th.nn.functional.binary_cross_entropy_with_logits(
                # /sum(weights)
                logits[..., detached_idx], target_probs[..., detached_idx], reduction='none') 
        loss[..., not_detached_idx] = th.nn.functional.binary_cross_entropy_with_logits(
                # /sum(weights)
                logits[..., not_detached_idx], target_probs[..., not_detached_idx], reduction='none') 
        pf2 = time.perf_counter()"""
        pf = time.perf_counter()
        loss = th.nn.functional.binary_cross_entropy_with_logits(
                # /sum(weights)
                logits, target_probs, reduction='none') 
        pf2 = time.perf_counter()

        #print(f"Loss computation time with no_grad_on_indexes: {pf2 - pf:.4f} seconds")
        #input("...")
    else:
        loss = th.nn.functional.binary_cross_entropy_with_logits(
                # /sum(weights)
                logits, target_probs, reduction='none')
    assert loss.shape == logits.shape, f"Expected loss shape {(logits.shape[0],)}, got {loss.shape}"
    mean = th.mean(loss, dim=-2)
    if rew_center_coefficient != 0 and rew_sum is not None:
        centering = th.mean((rew_sum)**2, dim=-2)
        assert centering.shape == mean.shape, f"Expected centering shape {mean.shape}, got {centering.shape}"
        mean += rew_center_coefficient * centering
    assert mean.shape == (logits.shape[-1],), f"Expected loss shape {(logits.shape[-1],)}, got {loss.shape}"
    if return_metrics:
        metrics = {}
        metrics['coherences'] = accuracy_logits(logits_p, target_probs_p, missing_mask=missing_mask)
        metrics['avg_coherence'] = metrics['coherences'].mean().item()
        return mean, metrics
    return mean


def value_system_loss_logits(logits_p: th.Tensor, target_probs_p: th.Tensor, rew_sum: th.Tensor=None, return_metrics: bool=False, check_undefined_label=True, rew_center_coefficient=0.0, missing_mask: th.Tensor=None) -> th.Tensor:
    missing_mask = get_missing_rating_mask(target_probs_p) if check_undefined_label and missing_mask is None else missing_mask
    
    if check_undefined_label:
        logits = logits_p.masked_fill(missing_mask, 0.0)
        target_probs = target_probs_p.masked_fill(missing_mask, 0.5)
    else:
        logits = logits_p
        target_probs = target_probs_p

    loss = th.nn.functional.binary_cross_entropy_with_logits(
                # /sum(weights)
                logits, target_probs.detach(), reduction='mean') #+ rew_center_coefficient*th.mean((reward1 + reward2)**2, dim=-2) 
    #assert loss.shape == reward1.shape, f"Expected loss shape {(reward1.shape[0],)}, got {loss.shape}"
    if rew_center_coefficient != 0:
        loss += rew_center_coefficient * th.mean((rew_sum)**2)

    if return_metrics:
        metrics = {}
        metrics['representativeness'] = accuracy_logits(logits_p, target_probs_p, missing_mask=missing_mask)
        return loss, metrics
    return loss


def reward_pairs_and_scores_to_logits_and_targets(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold=50.0, assume_qualitative_labels=False, check_undefined_label=True, assume_torch=True) -> tuple[th.Tensor, th.Tensor, Dict[str, Any]]:
    missing_mask = get_missing_rating_mask(scores1, scores2) if check_undefined_label else None
    logits_p = logits_BT(reward1, reward2, threshold=reward_diff_threshold, missing_mask=missing_mask, assume_torch=assume_torch)
    target_probs_p = scores_to_target_probs(scores1, scores2, reward_diff_threshold, assume_qualitative_labels, check_undefined_label, missing_mask=missing_mask, assume_torch=assume_torch)
    rew_sum = reward1 + reward2
    rew_sum = rew_sum.masked_fill(missing_mask, 0.0) if missing_mask is not None else rew_sum

    others = {
        'missing_mask': missing_mask,
        'rew_sum': rew_sum
    }
    return logits_p, target_probs_p, others
def grounding_loss(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor=None, scores2: th.Tensor=None, reward_diff_threshold: float=50.0, return_metrics: bool=False, assume_qualitative_labels=False, check_undefined_label=True, rew_center_coefficient=0.0) -> th.Tensor:
    """Multi-objective Cross-entropy loss: target_probs(1,2)*log(exp(r1) / (exp(r1) + exp(r2)))- (1-target_probs(1,2))*log(exp(r2) / (exp(r1) + exp(r2)))"""
    # label = 1: reward1 should be higher.
    # label = 0: reward2 should be higher.
    # label = 0.5: no preference.
    logits_p, target_probs_p, others = reward_pairs_and_scores_to_logits_and_targets(reward1, reward2, scores1, scores2, reward_diff_threshold, assume_qualitative_labels, check_undefined_label)

    missing_mask = others['missing_mask']
    rew_sum = others['rew_sum']

    assert len(reward1.shape) == 2 and reward1.shape[-1] == scores1.shape[-1], f"Expected reward1 shape (batch_size, num_values) and scores1 shape (batch_size, num_values), but got {reward1.shape} and {scores1.shape}"
    
    return grounding_loss_logits(logits_p, target_probs_p, rew_sum=rew_sum, return_metrics=return_metrics, check_undefined_label=check_undefined_label, rew_center_coefficient=rew_center_coefficient, missing_mask=missing_mask)


def value_system_loss(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold=50.0, assume_qualitative_labels=False, check_undefined_label=False, return_metrics=False, rew_center_coefficient=0.0) -> th.Tensor:
    logits_p, target_probs_p, others = reward_pairs_and_scores_to_logits_and_targets(reward1, reward2, scores1, scores2, reward_diff_threshold, assume_qualitative_labels, check_undefined_label)

    missing_mask = others['missing_mask']
    rew_sum = others['rew_sum']

    return value_system_loss_logits(logits_p, target_probs_p, rew_sum=rew_sum, return_metrics=return_metrics, check_undefined_label=check_undefined_label, rew_center_coefficient=rew_center_coefficient, missing_mask=missing_mask)


def mo_loss_function(logits, labels, pooled_logits, ideal_logits=None, config: MORMForSequenceClassificationConfig =None, training_variables: MORMTrainingVariables =None, accelerator=None, **kwargs):
    
    assert logits is pooled_logits, "Expected logits and pooled_logits to be the same, but got different tensors. Please ensure that the model's forward function returns the same tensor for both logits and pooled_logits, or adjust the mo_loss_function accordingly."
    
    logits, labels, others = rewards_and_labels_to_logits_and_targets(logits, labels, assume_torch=True, config=config) 
    missing_mask = others['missing_mask']
    assert logits.shape == labels.shape, f"Expected logits shape {logits.shape} to match labels shape {labels.shape}"
    assert missing_mask.shape == logits.shape, f"Expected missing_mask shape {missing_mask.shape} to match logits shape {logits.shape}"
    grounding_mask = missing_mask[...,0:-1]
    vs_mask = missing_mask[...,-1]

    rew_sum = others.get('rew_sum', None)
    if rew_sum is not None:
        grounding_rew_sum = rew_sum[...,0:-1]
        vs_rew_sum = rew_sum[...,-1]
    else:
        grounding_rew_sum = None
        vs_rew_sum = None

    if ideal_logits is not None:
        ideal_logits, _, _ = rewards_and_labels_to_logits_and_targets(ideal_logits, None, assume_torch=True, config=config)

    if config is not None: assert logits.shape[-1] == config.num_values + 1
    #assert labels.shape == pooled_logits.shape, f"Labels shape {labels.shape} does not match pooled logits shape {pooled_logits.shape}"
    use_metrics = training_variables is not None and training_variables.use_metrics_or_losses == 'metrics'

    if config.loss_func_type in MOLossFunctionsCategories.NEEDS_NO_GRAD_ON_ALL_GROUNDINGS:
        with th.no_grad():
            gr_loss = grounding_loss_logits(logits[...,0:-1], labels[...,0:-1], rew_sum=grounding_rew_sum, missing_mask=grounding_mask,check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
            
    elif config.loss_func_type in   MOLossFunctionsCategories.REQUIRES_GRAD_ON_SOME_GROUNDING:
        #gr_loss = grounding_loss(rewards_1[...,0:-1], rewards_2[...,0:-1], scores1=labels_1[...,0:-1], scores2=labels_2[...,0:-1], reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
        value_indices = config.loss_func_type_kwargs.get('value_indices', config.base_model_reward_head_indices)
        if value_indices is None:
            value_indices = list(range(logits.shape[-1] - 1))
        not_grad_indices = [i for i in range(logits.shape[-1] - 1) if i not in value_indices]
        gr_loss = grounding_loss_logits(
            logits[...,0:-1],
            labels[...,0:-1],
            rew_sum=grounding_rew_sum,
            missing_mask=grounding_mask,
            check_undefined_label=config.check_undefined_label,
            return_metrics=use_metrics,
            rew_center_coefficient=config.rew_center_coefficient,
            no_grad_on_indexes=not_grad_indices,
        )
    else:
        gr_loss = grounding_loss_logits(logits[...,0:-1], labels[...,0:-1], rew_sum=grounding_rew_sum, missing_mask=grounding_mask,check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
        
    if ideal_logits is not None:
        gr_loss_ideal = grounding_loss_logits(ideal_logits[...,0:-1], labels[...,0:-1], rew_sum=grounding_rew_sum, missing_mask=grounding_mask,check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
    
    if config.loss_func_type in MOLossFunctionsCategories.NEEDS_NO_GRAD_ON_VALUE_SYSTEM_WEIGHTS:
        with th.no_grad():
            #vs_loss = value_system_loss(rewards_1[...,-1],rewards_2[...,-1], scores1=labels_1[..., -1], scores2=labels_2[..., -1] , reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
            vs_loss = value_system_loss_logits(logits[..., -1], labels[..., -1], rew_sum=vs_rew_sum, missing_mask=vs_mask, check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
    else:
        vs_loss = value_system_loss_logits(logits[..., -1], labels[..., -1], rew_sum=vs_rew_sum, missing_mask=vs_mask, check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)

    with th.no_grad():
        if use_metrics:
            metrics_grounding: dict = gr_loss[1]
            metrics_value_system: dict = vs_loss[1]
            # join the two dicts
            metrics = {**metrics_grounding, **metrics_value_system}
            if ideal_logits is not None:
                metrics_grounding_ideal: dict = gr_loss_ideal[1]
                metrics = {**metrics, **{f"{k}_ideal": v for k, v in metrics_grounding_ideal.items()}}
    vs_loss = vs_loss[0]
    gr_loss = gr_loss[0]
    gr_loss_ideal = gr_loss_ideal[0] if ideal_logits is not None else None
    if th.is_grad_enabled() and training_variables is not None:
        with th.no_grad():
            grl = gr_loss.detach().clone().cpu()
            vsl = vs_loss.detach().clone().cpu()
            grli = gr_loss_ideal.detach().clone().cpu() if ideal_logits is not None else None
            training_variables.record_grounding_loss(gr_loss_detached=grl, vs_loss_detached=vsl, gr_loss_ideal_detached=grli)
    if use_metrics and training_variables is not None:
        assert "representativeness" in metrics.keys() and "coherences" in metrics.keys(), f"Expected metrics to contain 'representativeness' and 'coherences', but got {metrics.keys()}"
        training_variables.record_metrics(metrics)
        
    
    """Use this when only lagrange:
    total_loss = training_variables.forward(gr_loss, vs_loss)
    
    if th.is_grad_enabled():
        training_variables.record_grounding_loss(gr_loss.detach().clone(), vs_loss.detach().clone())
        
    return total_loss"""
    """if training_variables is not None:
        if th.is_grad_enabled():
            with th.no_grad():
                training_variables.record_grounding_loss(gr_loss.detach().clone(), vs_loss.detach().clone(), gr_loss_ideal.detach().clone() if ideal_logits is not None else None)
            training_variables.requires_grad_(False)
        loss_final = training_variables.forward(gr_loss, vs_loss, gr_loss_ideal if ideal_logits is not None else None)
        return loss_final"""
    if ideal_logits is not None:
        return th.cat([gr_loss, gr_loss_ideal, vs_loss.reshape(-1)])
    else:
        return th.cat([gr_loss, vs_loss.reshape(-1)])


    

def mo_compute_loss_func(outputs, labels, config=None, training_variables=None, accelerator=None, **kwargs):
    
    #print("OUTPUTS LOGITS SHAPE", outputs.logits.shape, "LABELS SHAPE", labels.shape)
    assert outputs.logits.device == labels.device, "Devices do not match"
    id_logits = getattr(outputs, "ideal_logits", None)
    return parse_loss_function(config)(outputs.logits, labels, outputs.logits, ideal_logits = id_logits, config=config, training_variables=training_variables, accelerator=accelerator, **kwargs)

@dataclass
class SequenceClassifierOutputWithPastAndIdeal(SequenceClassifierOutputWithPast):
    ideal_logits: th.Tensor = None
    


class MultiValueRewardHead(nn.Module):
    def __init__(self, value_heads: nn.ModuleList, normalization: nn.Module, optimized_head_indices: Optional[list[int]] = None):
        super().__init__()
        self.value_heads = value_heads
        self.normalization = normalization
        self.optimized_head_indices = optimized_head_indices if optimized_head_indices is not None else list(range(len(value_heads)))
        for head_i in range(len(value_heads)):
            if head_i in self.optimized_head_indices:
                self.value_heads[head_i].requires_grad_(True)
            else:
                self.value_heads[head_i].requires_grad_(False)
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        yield from self.normalization.parameters(recurse=recurse)

        if self.optimized_head_indices is None:
            yield from self.value_heads.parameters(recurse=recurse)
        else:
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

class MORMForSequenceClassification(PreTrainedModel, AutoModelForSequenceClassification):
    base_model_prefix = "full_model"
    supports_gradient_checkpointing = True
    
    predict_mode="rewards" # "rewards" or "logits", whether the model's forward should return reward values or raw logits (for interpretability, debugging, and ideal grounding model training)


    def parameters(self, recurse: bool = True) -> Iterator[th.nn.Parameter]:
        # Override parameters to only return reward head and value system parameters for optimization.
        if self.use_base_model_heads:
            return self.full_model.parameters(recurse=recurse)
        
        if self.reward_heads is not None and (self.config.loss_func_type in MOLossFunctionsCategories.REQUIRES_GRAD_ON_SOME_OR_ALL_GROUNDING):
            yield from self.reward_heads.parameters(recurse=recurse)
        if self.value_system_layer is not None and (self.config.loss_func_type in MOLossFunctionsCategories.REQUIRES_GRAD_ON_VALUE_SYSTEM_WEIGHTS):
            yield from self.value_system_layer.parameters(recurse=recurse)
        #yield from self.training_variables.parameters(recurse=recurse)
        if self.use_ideal_grounding_model and (self.config.loss_func_type in MOLossFunctionsCategories.REQUIRES_GRAD_ON_SOME_OR_ALL_GROUNDING):
            yield from self.reward_heads_ideal.parameters(recurse=recurse)

    def _select_reward_indices(self, rewards: th.Tensor) -> th.Tensor:
        indices = self.base_model_reward_head_indices
        if indices is None:
            return rewards
        if len(indices) == 0:
            raise ValueError("base_model_reward_head_indices cannot be an empty list when using base model outputs.")
        idx = th.tensor(indices, device=rewards.device, dtype=th.long)
        if rewards.ndim == 1:
            return rewards[idx]
        return th.index_select(rewards, dim=-1, index=idx)

    def _extract_logits_from_base_output(self, output: Any) -> th.Tensor:
        rewards_attr = self.base_model_rewards_attr_name
        score_attr = self.base_model_score_attr_name

        rewards = getattr(output, rewards_attr, None)
        if rewards is None:
            raise ValueError(f"Base model output does not have reward attribute '{rewards_attr}'.")

        score = getattr(output, score_attr, None)
        if score is None:
            raise ValueError(f"Base model output does not have score attribute '{score_attr}'.")

        rewards = self._select_reward_indices(rewards)

        

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(0)

        if score.ndim == 0:
            score = score.unsqueeze(0).unsqueeze(-1)
        elif score.ndim == 1:
            score = score.unsqueeze(-1)
        elif score.ndim > 2:
            score = score.reshape(score.shape[0], -1)
        
        

        assert score.shape[-1] == 1, f"Expected score to have last dimension 1 after processing, but got shape {score.shape}"
        """if not isinstance(rewards, th.Tensor):
            rewards = th.as_tensor(rewards)
        if not isinstance(score, th.Tensor):
            score = th.as_tensor(score)

        rewards = self._select_reward_indices(rewards)

        

        if score.shape[-1] != 1:
            score = score[..., :1]"""

        return th.cat([rewards, score], dim=-1)

    def _extract_reward_heads_by_index(self, base_reward_heads, indices: list):
        """
        Extract reward heads at specified indices from the base model.
        Handles both nn.ModuleList and other container types.
        """
        if isinstance(base_reward_heads, nn.ModuleList):
            extracted = nn.ModuleList([base_reward_heads[i] for i in indices])
        elif hasattr(base_reward_heads, '__getitem__'):
            extracted = nn.ModuleList([base_reward_heads[i] for i in indices])
        else:
            raise ValueError(f"Cannot extract indices from reward heads of type {type(base_reward_heads)}")
        return extracted

    def _infer_base_hidden_size(self, base_model: AutoModelForSequenceClassification) -> int:
        # SequenceClassification wrappers often expose the classifier head input width here.
        if hasattr(base_model, "score") and hasattr(base_model.score, "in_features"):
            return int(base_model.score.in_features)

        cfg = getattr(base_model, "config", None)
        for attr in ("hidden_size", "d_model", "n_embd", "dim"):
            value = getattr(cfg, attr, None)
            if value is not None:
                return int(value)

        # Last fallback for models with custom configs but standard embedding modules.
        input_emb = base_model.get_input_embeddings() if hasattr(base_model, "get_input_embeddings") else None
        if input_emb is not None and hasattr(input_emb, "embedding_dim"):
            return int(input_emb.embedding_dim)
        if input_emb is not None and hasattr(input_emb, "weight"):
            return int(input_emb.weight.shape[-1])

        raise ValueError(
            "Could not infer hidden size for reward heads. Expected one of: "
            "score.in_features, config.hidden_size/d_model/n_embd/dim, or input embedding width."
        )
    
    

    def construct_value_layer(self, config: MORMForSequenceClassificationConfig, base_model: BaseModelOutputWithPast = None):
        layers = []
        input_size = self._infer_base_hidden_size(base_model)
        for hidden_size in config.hidden_sizes: 
            layers.append(nn.Linear(input_size, hidden_size, dtype=config.dtype, device=base_model.device))
            if config.value_layer_intermediate_activation == "ReLU":
                layers.append(nn.ReLU())
            elif config.value_layer_intermediate_activation == "Tanh":
                layers.append(nn.Tanh())
            elif config.value_layer_intermediate_activation == "Softplus":
                layers.append(nn.Softplus())
            elif config.value_layer_intermediate_activation == "SiLU":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unsupported intermediate activation: {config.value_layer_intermediate_activation}")
            #layers.append(nn.Dropout(config.value_layer_dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1, dtype=config.dtype, device=base_model.device))
        
        if config.value_layer_final_activation == "ReLU":
            layers.append(nn.ReLU())
        elif config.value_layer_final_activation == "Tanh":
            layers.append(nn.Tanh())
        elif config.value_layer_final_activation == "Softplus": 
            layers.append(nn.Softplus())
        elif config.value_layer_final_activation == "SiLU":
            layers.append(nn.SiLU())
        elif config.value_layer_final_activation == "none":
            pass
        else:
            raise ValueError(f"Unsupported final activation: {config.value_layer_final_activation}")

        return nn.Sequential(*layers)

    def construct_value_normalization(self, config: MORMForSequenceClassificationConfig, base_model: BaseModelOutputWithPast = None):
        # Normalize across value dimensions to keep reward channels on a comparable scale.
        if config.layer_normalization == 'LayerNorm':
            return nn.LayerNorm(config.num_values, dtype=config.dtype, device=base_model.device)
        if config.layer_normalization == 'BatchNorm':
            return nn.BatchNorm1d(config.num_values, dtype=config.dtype, device=base_model.device)
        if config.layer_normalization == 'none':
            return nn.Identity()
        raise ValueError(f"Unsupported normalization: {config.layer_normalization}")

    def construct_reward_head(self, config: MORMForSequenceClassificationConfig, base_model: BaseModelOutputWithPast = None) -> MultiValueRewardHead:
        value_heads = nn.ModuleList([
            self.construct_value_layer(config, base_model)
            for _ in range(config.num_values)
        ])
        normalization = self.construct_value_normalization(config, base_model)
        return MultiValueRewardHead(value_heads=value_heads, normalization=normalization, optimized_head_indices=config.loss_func_type_kwargs.get('value_indices', None))
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.reward_heads is not None:
                self.reward_heads = self.reward_heads.to(*args, **kwargs)
        if self.use_ideal_grounding_model and self.reward_heads_ideal is not None:
                self.reward_heads_ideal = self.reward_heads_ideal.to(*args, **kwargs)
        if self.value_system_layer is not None:
                self.value_system_layer = self.value_system_layer.to(*args, **kwargs)
        self.training_variables = self.training_variables.to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True):
        self._freeze_base_model_keep_heads_trainable()
        self.forward_ideal_grounding = mode and self.use_ideal_grounding_model
        return super().train(mode)
    
    def __init__(self, config: MORMForSequenceClassificationConfig, base_model: AutoModelForSequenceClassification = None):
        super().__init__(config)
        
        self.full_model = base_model
        self.supports_gradient_checkpointing = hasattr(self.full_model, "gradient_checkpointing_enable")
        self.num_values = config.num_values
        self.use_ideal_grounding_model = config.use_ideal_grounding_model
        self.use_base_model_heads = config.use_base_model_heads
        self.base_model_reward_head_indices = config.base_model_reward_head_indices
        self.base_model_rewards_attr_name = config.base_model_reward_heads_module_name
        self.base_model_score_attr_name = config.base_model_value_system_module_name
        
        # In base-model mode, consume reward/score attributes from base model outputs.
        if self.use_base_model_heads:
            if not self.base_model_rewards_attr_name:
                raise ValueError("use_base_model_heads=True requires base_model_reward_heads_module_name to specify the output reward attribute name.")
            if not self.base_model_score_attr_name:
                raise ValueError("use_base_model_heads=True requires base_model_value_system_module_name to specify the output score attribute name.")
            self.reward_heads = None
            self.value_system_layer = None
            self.reward_heads_ideal = None
        else:
            self.reward_heads = self.construct_reward_head(config, base_model)
            if config.use_ideal_grounding_model:
                self.reward_heads_ideal = self.construct_reward_head(config, base_model)
            self.value_system_layer = ConvexAlignmentLayer(config.num_values, 1, device=self.full_model.device, dtype=self.full_model.dtype)
            self.reward_heads_ideal = None if not config.use_ideal_grounding_model else self.reward_heads_ideal

        self.training_variables = MORMTrainingVariables(n_values=config.num_values, initial_lambda=1.0,
            device=self.full_model.device, dtype=config.dtype, grounding_loss_tendency_update_ratio=config.grounding_loss_tendency_update_ratio, 
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            metric_buffer_size=config.metrics_accumulation_steps,
            use_metrics_or_losses=config.use_metrics_or_losses_for_lagrange_updates,
            grad_on_only_worst_value=config.grad_on_only_worst_value,
            zero_constraint=config.zero_constraint,
            lambda_decay=config.lambda_decay if hasattr(config, "lambda_decay") else 0.0
                                                )
        self._freeze_base_model_keep_heads_trainable()
        self.forward_ideal_grounding = self.use_ideal_grounding_model

        self.loss_function = partial(parse_loss_function(self.config), training_variables=self.training_variables, config=self.config)
        
        self.zero_grad(set_to_none=True)
		#self.score_weight_head: ConvexAlignmentLayer = ConvexAlignmentLayer(num_values, 1)
    
    def _freeze_base_model_keep_heads_trainable(self) -> None:
        # Freeze pretrained weights and train only the custom reward/value-system heads.
        for param in self.full_model.parameters():
            param.requires_grad = False

        if self.reward_heads is not None:
            for param in self.reward_heads.parameters():
                param.requires_grad = True

        if self.value_system_layer is not None:
            for param in self.value_system_layer.parameters():
                param.requires_grad = True
        self.training_variables.requires_grad_(False)

    """def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None):
        if not hasattr(self.full_model, "gradient_checkpointing_enable"):
            raise ValueError(f"{self.full_model.__class__.__name__} does not support gradient checkpointing.")
        return self.full_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        if hasattr(self.full_model, "gradient_checkpointing_disable"):
            return self.full_model.gradient_checkpointing_disable()
        return None"""

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.full_model, "is_gradient_checkpointing", False))

    """def get_input_embeddings(self):
        return self.full_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.full_model.set_input_embeddings(value)
	"""
    def set_value_system_layer(self, layer: nn.Module):
        self.value_system_layer = layer

    def grounding_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        if self.config.loss_func_type in MOLossFunctionsCategories.REQUIRES_GRAD_ON_SOME_OR_ALL_GROUNDING:
            if self.reward_heads is not None:
                return self.reward_heads.parameters(recurse=recurse)
            elif self.use_base_model_heads:
                # In base model mode, we assume all parameters require grad, but we only want to return the reward head parameters for optimization.
                return self.full_model.parameters(recurse=recurse)
        else:
            return iter([])
    
    def value_system_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        if self.config.loss_func_type in MOLossFunctionsCategories.REQUIRES_GRAD_ON_VALUE_SYSTEM_WEIGHTS:
            if self.value_system_layer is not None:
                return self.value_system_layer.parameters(recurse=recurse)
            elif self.use_base_model_heads:
                # In base model mode, we assume all parameters require grad, but we only want to return the reward head parameters for optimization.
                return self.full_model.parameters(recurse=recurse)
        else:
            return iter([])
    
    def score_ideal(self, hidden_state):
        # This is used inside the GenericForSequenceClassification forward method.
        #assert hidden_state.dtype == self.reward_heads_ideal.reference_weight().dtype, f"Expected hidden state dtype {self.reward_heads_ideal.reference_weight().dtype}, but got {hidden_state.dtype}"
        rewards = self.reward_heads_ideal(hidden_state)
        
        if self.value_system_layer is not None:
            vs_reward = self.value_system_layer.forward(rewards)
            all_rewards = th.cat([rewards, vs_reward], dim=-1)
        else:
            all_rewards = rewards
        return all_rewards
    
    def score_normal(self, hidden_state):
        if hasattr(self.reward_heads, 'reference_weight'):
            #assert hidden_state.dtype == self.reward_heads.reference_weight().dtype, f"Expected hidden state dtype {self.reward_heads.reference_weight().dtype}, but got {hidden_state.dtype}"
            pass
        rewards = self.reward_heads(hidden_state)
        
        if self.value_system_layer is not None:
            if self.config.loss_func_type in MOLossFunctionsCategories.NEEDS_NO_GRAD_ON_VALUE_SYSTEM_WEIGHTS:
                with th.no_grad():
                    vs_reward = self.value_system_layer.forward(rewards)
            else:
                vs_reward = self.value_system_layer.forward(rewards)
            all_rewards = th.cat([rewards, vs_reward], dim=-1)
        else:
            all_rewards = rewards
        return all_rewards

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none)
        if self.reward_heads is not None:
            self.reward_heads.zero_grad(set_to_none)
        if self.use_ideal_grounding_model:
            self.reward_heads_ideal.zero_grad(set_to_none)
        if self.value_system_layer is not None:
            self.value_system_layer.zero_grad(set_to_none)
        self.training_variables.zero_grad(set_to_none)

    def forward(self, *args, **kwargs):
        #print("FORWARD CALLED WITH ARGS", self.use_base_model_heads)
        #exit(0)
        if self.use_base_model_heads:
            kwargs.pop("labels", None)
            kwargs.pop("embedding", None)
            kwargs.pop("context_embedding", None)

            kwargs.pop("return_loss", None)

            # TODO this will not work with other models, do not know how to check this.
            kwargs.pop("num_items_in_batch")
            base_output = self.full_model(*args, **kwargs, return_dict=True)
            pooled_logits = self._extract_logits_from_base_output(base_output)
            del base_output
            #print("LABELS SHAPE", labels.shape)

            return SequenceClassifierOutputWithPast(
                logits=pooled_logits,
                #past_key_values=getattr(base_output, "past_key_values", None),
                #hidden_states=getattr(base_output, "hidden_states", None),
                #attentions=getattr(base_output, "attentions", None),
            )

        self.score = self.score_normal
        if 'embedding' in kwargs:
            # If embeddings are provided, bypass the base model and directly compute rewards from embeddings.
            embeddings = kwargs.pop('embedding')
            all_rewards = self.score(embeddings)
            
            if self.forward_ideal_grounding:
                grounding_ideal = self.reward_heads_ideal(embeddings)
                return SequenceClassifierOutputWithPastAndIdeal(logits=all_rewards, ideal_logits=grounding_ideal)
            else:
                return SequenceClassifierOutputWithPast(logits=all_rewards)
        else:
            raise NotImplementedError("Forward without embeddings is not implemented yet. This requires modifying the base model's forward method to call the reward heads on the appropriate hidden states. This is left as future work to keep the current implementation simpler and more focused on the training loop and loss function.")
            
            #sq = GenericForSequenceClassification.forward(self, *args, **kwargs)
            sq = self.generic_forward(*args, **kwargs)
            if self.forward_ideal_grounding:
                self.score = self.score_ideal
                ideal_sq = self.generic_forward(*args, **kwargs) # = GenericForSequenceClassification.forward(self, *args, **kwargs)
                self.score = self.score_normal
                return SequenceClassifierOutputWithPastAndIdeal(logits=sq.logits, ideal_logits=ideal_sq.logits)
        
    def __slowed_debug_forward(self, *args, **kwargs):
        
        if 'embeddings' in kwargs:
            # If embeddings are provided, bypass the base model and directly compute rewards from embeddings.
            embeddings = kwargs.pop('embeddings')
            # embeding tuple=?????
            all_rewards = self.score(embeddings)
            grounding_ideal = self.reward_heads_ideal(embeddings)
            ar = SequenceClassifierOutputWithPastAndIdeal(logits=all_rewards, ideal_logits=grounding_ideal)
        
        all_rewards_2 = GenericForSequenceClassification.forward(self, *args, **kwargs)
        all_rewards_4 = GenericForSequenceClassification.forward(self, *args, **kwargs)
        th.testing.assert_close(all_rewards_2.logits, all_rewards_4.logits, atol=1e-4, rtol=1e-4)

        lhs = self.full_model(*args, **kwargs).last_hidden_state
        input_ids = kwargs.get("input_ids", None)
        non_pad_mask = (input_ids != self.config.pad_token_id).to(lhs.device, th.int32)
        token_indices = th.arange(input_ids.shape[-1], device=lhs.device, dtype=th.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        embedding = lhs[th.arange(input_ids.size(0), device=lhs.device), last_non_pad_token]

        scores_all = self.score(lhs)
        selected_scores = scores_all[th.arange(input_ids.size(0), device=lhs.device), last_non_pad_token]
        score_embed = self.score(embedding)
        print("last_non_pad_token", last_non_pad_token)
        print("SCORES ALL SHAPE", scores_all.shape)
        print("SCORES Selected SHAPE", selected_scores.shape)
        print("SCORES EMBED SHAPE", score_embed.shape)
        print("SCORE 1", selected_scores[0])
        print("SCORE EMBED 1", score_embed[0])
        print("SCORE 2", selected_scores[1])
        print("SCORE EMBED 2", score_embed[1])
        
        all_rewards_5 = SequenceClassifierOutputWithPast(logits=selected_scores)
        all_rewards_3 = SequenceClassifierOutputWithPast(logits=score_embed)
        th.testing.assert_close(all_rewards_5.logits, all_rewards_3.logits, atol=1e-1, rtol=1e-1)

        print("SHAPE", all_rewards_2.logits.shape)
        assert all_rewards_2.logits.shape == all_rewards.shape, f"Expected logits shape {all_rewards_2.logits.shape} to match pooled_logits shape {all_rewards_2.pooled_logits.shape}"
        th.testing.assert_close(all_rewards_3.logits, ar.logits, atol=1e-4, rtol=1e-4)
        th.testing.assert_close(all_rewards_3.logits, all_rewards_2.logits, atol=1e-1, rtol=1e-1)
        #th.testing.assert_close(ar.logits, all_rewards_2.logits, atol=1e-4, rtol=1e-4)
        return all_rewards_2


    
    
    def generic_forward(
        self,
        input_ids: th.LongTensor | None = None,
        attention_mask: th.Tensor | None = None,
        position_ids: th.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        labels: th.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[Dict[str, Any]],
    ) -> SequenceClassifierOutputWithPast:
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
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, th.int32)
            token_indices = th.arange(input_ids.shape[-1], device=logits.device, dtype=th.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[th.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    
    
        