
from ast import Tuple
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass

from functools import partial
import re
from typing import Any, Literal

from datasets import config
import numpy as np
import torch as th
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from transformers.modeling_layers import GenericForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast

from vsllib.defines import NO_RATING_MASK
from vsllib.utils import MORMTrainingVariables
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
            return w_bounded.clone().detach().view(-1).cpu().tolist()
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


from transformers import PreTrainedConfig
from typing import List

class MORMForSequenceClassificationConfig(PreTrainedConfig):
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
        reward_diff_threshold: str = 50.0,
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

def accuracy_rewards_labels(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, epsilon=1.0e-2, threshold=50.0, assume_qualitative_labels=False, check_undefined_label=True, missing_mask=None, assume_torch=True) -> th.Tensor:
    logits = logits_BT(reward1, reward2, threshold=threshold, check_undefined_label=check_undefined_label, missing_mask=missing_mask, assume_torch=assume_torch)
    targets = scores_to_target_probs(scores1, scores2, reward_diff_threshold=threshold, assume_qualitative_labels=assume_qualitative_labels, check_undefined_label=check_undefined_label, missing_mask=missing_mask, assume_torch=assume_torch)
    return accuracy_logits(logits, targets, epsilon=epsilon, missing_mask=missing_mask, assume_torch=assume_torch)
def accuracy_logits(logits: th.Tensor, target_probs: th.Tensor, epsilon=1.0e-2, missing_mask=None, assume_torch=True) -> th.Tensor:
    with th.no_grad():

        rep_mask1 = (logits > 0) & (target_probs > 0.5)
        rep_mask2 = (logits < 0) & (target_probs < 0.5)
        rep_mask3 = (target_probs == 0.5) & ((logits <= epsilon) & (logits >= -epsilon))
        all_defined_cases = (target_probs != NO_RATING_MASK)  if missing_mask is None else ~missing_mask
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
            missing_mask = (x == NO_RATING_MASK) | (y == NO_RATING_MASK)
            
        

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
                mask = (scores1 == NO_RATING_MASK) | (scores2 == NO_RATING_MASK) if missing_mask is None else missing_mask
                if assume_torch:
                    target_probs.masked_fill_(mask, NO_RATING_MASK)
                else:
                    target_probs[mask] = NO_RATING_MASK
    return target_probs
def grounding_loss(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor=None, scores2: th.Tensor=None, reward_diff_threshold: float=50.0, return_metrics: bool=False, assume_qualitative_labels=False, check_undefined_label=True, rew_center_coefficient=0.0) -> th.Tensor:
    """Multi-objective Cross-entropy loss: target_probs(1,2)*log(exp(r1) / (exp(r1) + exp(r2)))- (1-target_probs(1,2))*log(exp(r2) / (exp(r1) + exp(r2)))"""
    # label = 1: reward1 should be higher.
    # label = 0: reward2 should be higher.
    # label = 0.5: no preference.
    missing_mask = (scores1 == NO_RATING_MASK) | (scores2 == NO_RATING_MASK) if check_undefined_label else None
    assert len(reward1.shape) == 2 and reward1.shape[-1] == scores1.shape[-1], f"Expected reward1 shape (batch_size, num_values) and scores1 shape (batch_size, num_values), but got {reward1.shape} and {scores1.shape}"
    
    
    logits_p = logits_BT(reward1, reward2, threshold=reward_diff_threshold, missing_mask=missing_mask, assume_torch=True)
    target_probs_p = scores_to_target_probs(scores1, scores2, reward_diff_threshold=reward_diff_threshold, assume_qualitative_labels=assume_qualitative_labels, check_undefined_label=check_undefined_label, missing_mask=missing_mask, assume_torch=True)
    
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

    loss = th.nn.functional.binary_cross_entropy_with_logits(
                # /sum(weights)
                logits, target_probs, reduction='none', reduce=False) 
    assert loss.shape == reward1.shape, f"Expected loss shape {(reward1.shape[0],)}, got {loss.shape}"
    mean = th.mean(loss, dim=-2)
    if rew_center_coefficient != 0:
        mean += rew_center_coefficient * th.mean((reward1 + reward2)**2, dim=-2)
    assert mean.shape == (reward1.shape[-1],), f"Expected loss shape {(reward1.shape[-1],)}, got {loss.shape}"
    if return_metrics:
        metrics = {}
        metrics['coherences'] = accuracy_logits(logits_p, target_probs_p, missing_mask=missing_mask)
        metrics['avg_coherence'] = metrics['coherences'].mean().item()
        return mean, metrics
    return mean


def value_system_loss(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold=50.0, assume_qualitative_labels=False, check_undefined_label=False, return_metrics=False, rew_center_coefficient=0.0) -> th.Tensor:
    missing_mask = (scores1 == NO_RATING_MASK) | (scores2 == NO_RATING_MASK) if check_undefined_label else None 
    logits_p = logits_BT(reward1, reward2, threshold=reward_diff_threshold, missing_mask=missing_mask, assume_torch=True)
    target_probs_p = scores_to_target_probs(scores1, scores2, reward_diff_threshold, assume_qualitative_labels, check_undefined_label, missing_mask=missing_mask, assume_torch=True)

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
        loss += rew_center_coefficient * th.mean((reward1 + reward2)**2)

    if return_metrics:
        metrics = {}
        metrics['representativeness'] = accuracy_logits(logits_p, target_probs_p, missing_mask=missing_mask)
        return loss, metrics
    return loss


def mo_loss_function(logits, labels, pooled_logits, ideal_logits=None, config: MORMForSequenceClassificationConfig =None, training_variables: MORMTrainingVariables =None, accelerator=None, **kwargs):
    
    bsz = pooled_logits.size(0)

    jidx = th.arange(0, bsz, 2, device=pooled_logits.device)
    kidx = jidx + 1

    if ideal_logits is not None:

        rewards_1_ideal = ideal_logits[jidx]
        rewards_2_ideal = ideal_logits[kidx]

    rewards_1 = pooled_logits[jidx]
    rewards_2 = pooled_logits[kidx]
    labels_1 = labels[jidx]
    labels_2 = labels[kidx]
    

    
    assert rewards_1.shape[-1] == config.num_values + 1
    #assert labels.shape == pooled_logits.shape, f"Labels shape {labels.shape} does not match pooled logits shape {pooled_logits.shape}"
    use_metrics = training_variables is not None and training_variables.use_metrics_or_losses == 'metrics'
    gr_loss = grounding_loss(rewards_1[...,0:-1], rewards_2[...,0:-1], scores1=labels_1[...,0:-1], scores2=labels_2[...,0:-1], reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label, return_metrics=use_metrics)
    if ideal_logits is not None:
        gr_loss_ideal = grounding_loss(rewards_1_ideal, rewards_2_ideal, scores1=labels_1[...,0:-1], scores2=labels_2[...,0:-1], reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
    vs_loss = value_system_loss(rewards_1[...,-1],rewards_2[...,-1], scores1=labels_1[..., -1], scores2=labels_2[..., -1] , reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient)
    
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
    if th.is_grad_enabled():
        with th.no_grad():
            grl = gr_loss.detach().clone()
            vsl = vs_loss.detach().clone()
            grli = gr_loss_ideal.detach().clone() if ideal_logits is not None else None
            training_variables.record_grounding_loss(gr_loss_detached=grl, vs_loss_detached=vsl, gr_loss_ideal_detached=grli)
    if use_metrics:
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
    return mo_loss_function(outputs.logits, labels, outputs.logits, ideal_logits = id_logits, config=config, training_variables=training_variables, accelerator=accelerator, **kwargs)

@dataclass
class SequenceClassifierOutputWithPastAndIdeal(SequenceClassifierOutputWithPast):
    ideal_logits: th.Tensor = None


class MultiValueRewardHead(nn.Module):
    def __init__(self, value_heads: nn.ModuleList, normalization: nn.Module):
        super().__init__()
        self.value_heads = value_heads
        self.normalization = normalization

    def forward(self, hidden_state: th.Tensor) -> th.Tensor:
        
        rewards = th.cat([head(hidden_state) for head in self.value_heads], dim=-1)
        return self.normalization(rewards)

    def reference_weight(self) -> th.Tensor:
        # Used for dtype/device consistency assertions.
        for layer in self.value_heads[0]:
            if isinstance(layer, nn.Linear):
                return layer.weight
        raise ValueError("Expected at least one Linear layer in value head")

class MORMForSequenceClassification(PreTrainedModel, GenericForSequenceClassification):
    base_model_prefix = "full_model"
    supports_gradient_checkpointing = True
    
    def parameters(self, recurse: bool = True) -> Iterator[th.nn.Parameter]:
        # Override parameters to only return reward head and value system parameters for optimization.
        yield from self.reward_heads.parameters(recurse=recurse)
        yield from self.value_system_layer.parameters(recurse=recurse)
        yield from self.training_variables.parameters(recurse=recurse)
        if self.use_ideal_grounding_model:
            yield from self.reward_heads_ideal.parameters(recurse=recurse)

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
        return MultiValueRewardHead(value_heads=value_heads, normalization=normalization)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.reward_heads = self.reward_heads.to(*args, **kwargs)
        if self.use_ideal_grounding_model:
            self.reward_heads_ideal = self.reward_heads_ideal.to(*args, **kwargs)
        self.value_system_layer = self.value_system_layer.to(*args, **kwargs)
        self.training_variables = self.training_variables.to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True):
        self._freeze_base_model_keep_heads_trainable()
        self.forward_ideal_grounding = mode and self.use_ideal_grounding_model
        return super().train(mode)
    
    def __init__(self, config: MORMForSequenceClassificationConfig, base_model: AutoModelForSequenceClassification = None):
        super().__init__(config)
        
        self.reward_heads = self.construct_reward_head(config, base_model)
        if config.use_ideal_grounding_model:
            self.reward_heads_ideal = self.construct_reward_head(config, base_model)
        self.full_model = base_model
        self.supports_gradient_checkpointing = hasattr(self.full_model, "gradient_checkpointing_enable")
        self.value_system_layer = ConvexAlignmentLayer(config.num_values, 1, device=self.full_model.device, dtype=self.full_model.dtype)
        self.num_values = config.num_values
        self.use_ideal_grounding_model = config.use_ideal_grounding_model

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

        self.loss_function = partial(mo_loss_function, training_variables=self.training_variables)
        
        
		#self.score_weight_head: ConvexAlignmentLayer = ConvexAlignmentLayer(num_values, 1)
    
    def _freeze_base_model_keep_heads_trainable(self) -> None:
        # Freeze pretrained weights and train only the custom reward/value-system heads.
        for param in self.full_model.parameters():
            param.requires_grad = False

        for param in self.reward_heads.parameters():
            param.requires_grad = True

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
    
    def score_ideal(self, hidden_state):
        # This is used inside the GenericForSequenceClassification forward method.
        assert hidden_state.dtype == self.reward_heads_ideal.reference_weight().dtype, f"Expected hidden state dtype {self.reward_heads_ideal.reference_weight().dtype}, but got {hidden_state.dtype}"
        rewards = self.reward_heads_ideal(hidden_state)
        vs_reward = self.value_system_layer.forward(rewards)

        
        all_rewards = th.cat([rewards, vs_reward], dim=-1)
        #assert all_rewards.shape[-1] == self.config.num_labels, f"Expected rewards shape to have last dimension {self.config.num_labels}, but got {rewards.shape}"
        
        return all_rewards
    
    def score_normal(self, hidden_state):
        # This is used inside the GenericForSequenceClassification forward method.
        assert hidden_state.dtype == self.reward_heads.reference_weight().dtype, f"Expected hidden state dtype {self.reward_heads.reference_weight().dtype}, but got {hidden_state.dtype}"
        rewards = self.reward_heads(hidden_state)
        vs_reward = self.value_system_layer.forward(rewards)

        
        all_rewards = th.cat([rewards, vs_reward], dim=-1)
        #assert all_rewards.shape[-1] == self.config.num_labels, f"Expected rewards shape to have last dimension {self.config.num_labels}, but got {rewards.shape}"
        
        return all_rewards

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none)
        self.reward_heads.zero_grad(set_to_none)
        if self.use_ideal_grounding_model:
            self.reward_heads_ideal.zero_grad(set_to_none)
        self.value_system_layer.zero_grad(set_to_none)
        self.training_variables.zero_grad(set_to_none)

    def forward(self, *args, **kwargs):
        self.score = self.score_normal
        if 'embeddings' in kwargs:
            
            # If embeddings are provided, bypass the base model and directly compute rewards from embeddings.
            embeddings = kwargs.pop('embeddings')
            
            all_rewards = self.score(embeddings)
            if self.forward_ideal_grounding:
                grounding_ideal = self.reward_heads_ideal(embeddings)
                return SequenceClassifierOutputWithPastAndIdeal(logits=all_rewards, ideal_logits=grounding_ideal)
            else:
                return SequenceClassifierOutputWithPast(logits=all_rewards)
        else:
            raise NotImplementedError("Forward without embeddings is not implemented yet. This requires modifying the base model's forward method to call the reward heads on the appropriate hidden states. This is left as future work to keep the current implementation simpler and more focused on the training loop and loss function.")
            
            sq = GenericForSequenceClassification.forward(self, *args, **kwargs)
            if self.forward_ideal_grounding:
                self.score = self.score_ideal
                ideal_sq = GenericForSequenceClassification.forward(self, *args, **kwargs)
                self.score = self.score_normal
                return SequenceClassifierOutputWithPastAndIdeal(logits=sq.logits, ideal_logits=ideal_sq.logits)
        
    def __slowed_debug_forward(self, *args, **kwargs):
        
        if 'embeddings' in kwargs:
            # If embeddings are provided, bypass the base model and directly compute rewards from embeddings.
            embeddings = kwargs.pop('embeddings')
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
