
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import torch as th
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from transformers.modeling_layers import GenericForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

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

        output = th.nn.functional.linear(input, w_bounded)
        

        return output

    def get_alignment_layer(self):
        w_bounded = self.weight
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))
        b_bounded = 0.0
        if self.linear_bias:
            b_bounded = self.bias
        return w_bounded, b_bounded

    def copy(self):
        with th.no_grad():
            new = self.__class__(in_features=self.in_features, out_features=self.out_features, bias=self.linear_bias, device=self.weight.device, dtype=self.weight.dtype)
            new.load_state_dict(deepcopy(self.state_dict()))
        return new


class ConvexAlignmentLayer(LinearAlignmentLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=th.float32, data=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype, data)

    def set_weights(self, weights: tuple):
        # Convert to tensor with same dtype and device as self.weight
        pure_w = th.tensor(weights, dtype=self.weight.dtype, device=self.weight.device)
        new_weights = th.log(pure_w+1e-8)
        # Reshape to match weight shape
        new_weights = new_weights.view_as(self.weight)
        # Ensure requires_grad matches previous setting
        new_weights.requires_grad = self.weight.requires_grad
        # Update state dict in place
        with th.no_grad():
            self.weight.copy_(new_weights)
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

    def __init__(
        self,
        base_model: AutoModelForSequenceClassification,
        num_values: int = 3,
        hidden_sizes: list[int] = [4096,],
        value_layer_dropout: float = 0.1,
        value_layer_intermediate_activation: str = "SiLU",
        value_layer_final_activation: str = "none",
        reward_diff_threshold: str = 50.0,
        assume_qualitative_labels: bool = False,
        check_undefined_label: bool = True,
        **kwargs,
    ):
        assert num_values > 0, "num_values must be greater than 0"
        assert len(hidden_sizes) > 0, "hidden_sizes must be a non-empty list"

        if value_layer_intermediate_activation not in ['ReLU', 'SiLU', 'Tanh', 'Softplus']:
             raise ValueError(f"value_layer_intermediate_activation must be one of 'ReLU', 'SiLU', 'Tanh', 'Softplus', but got {value_layer_intermediate_activation}")
        if value_layer_final_activation not in ['ReLU', 'SiLU', 'Tanh', 'Softplus', 'none']:
             raise ValueError(f"value_layer_final_activation must be one of 'ReLU', 'SiLU', 'Tanh', 'Softplus', 'none', but got {value_layer_final_activation}")

        self.num_values = num_values
        self.hidden_sizes = hidden_sizes
        self.value_layer_dropout = value_layer_dropout
        self.value_layer_intermediate_activation = value_layer_intermediate_activation
        self.value_layer_final_activation = value_layer_final_activation
        self.base_model = base_model
        self.reward_diff_threshold = reward_diff_threshold
        self.assume_qualitative_labels = assume_qualitative_labels
        self.check_undefined_label = check_undefined_label

        default_id2label = {
            index: f"VALUE_{index}" for index in range(num_values)
        }
        default_id2label[num_values] = "VALUE_SYSTEM"
        id2label = kwargs.pop("id2label", default_id2label)
        label2id = kwargs.pop("label2id", {label: index for index, label in id2label.items()})
        self.pad_token_id = base_model.config.pad_token_id

        super().__init__(num_labels=num_values + 1, id2label=id2label, label2id=label2id, **kwargs)


def logits_BT(x: th.Tensor, y: th.Tensor, threshold=50.0, check_undefined_label=False) -> th.Tensor:
    # print("DIFF", th.max(x - y))
    if check_undefined_label:
        missing_mask = (x == NO_RATING_MASK) | (y == NO_RATING_MASK)
        if th.any(missing_mask).item():
            
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
                y[y_missing_mask] = x[y_missing_mask]

        # IMPORTANT: Because we use a bradley terry model, the scale does not matter. The model will learn indifference, perhaps force it... This is not ideal...
            
    returns_diff = x - y
    returns_diff = th.clip(returns_diff, -threshold, threshold)
    assert th.max(returns_diff) <= threshold and th.min(returns_diff) >= - \
        threshold, f"Clipping failed: max {th.max(returns_diff)}, min {th.min(returns_diff)}, threshold {threshold}"
    

    if returns_diff.requires_grad:
        assert returns_diff.grad_fn is not None, "The returned tensor does not require gradients."
    return returns_diff
def scores_to_target_probs(scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold: int=50.0, assume_qualitative_labels=False, check_undefined_label=True):
    with th.no_grad():
        if assume_qualitative_labels:	
            if check_undefined_label: 
                # If either score is NO_RATING_MASK, set target_prob to 0.5 (indicating no preference)
                mask = (scores1 == NO_RATING_MASK) | (scores2 == NO_RATING_MASK)
                target_probs = th.where(mask, 0.5, scores1)
            assert th.max(scores1) == 1.0
            assert th.min(scores1) == 0.0
            target_probs: th.Tensor = scores1 # model probability of first one being preferred.
        else:
            target_probs = th.sigmoid(logits_BT(scores1, scores2, threshold=reward_diff_threshold, check_undefined_label=check_undefined_label))
    return target_probs
def grounding_loss(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor=None, scores2: th.Tensor=None, reward_diff_threshold: float=50.0, assume_qualitative_labels=False, check_undefined_label=True):
    """Multi-objective Cross-entropy loss: target_probs(1,2)*log(exp(r1) / (exp(r1) + exp(r2)))- (1-target_probs(1,2))*log(exp(r2) / (exp(r1) + exp(r2)))"""
    # label = 1: reward1 should be higher.
    # label = 0: reward2 should be higher.
    # label = 0.5: no preference.
    assert len(reward1.shape) == 2 and reward1.shape[-1] == scores1.shape[-1], f"Expected reward1 shape (batch_size, num_values) and scores1 shape (batch_size, num_values), but got {reward1.shape} and {scores1.shape}"
    print(reward1.shape, reward2.shape, scores1.shape if scores1 is not None else None, scores2.shape if scores2 is not None else None)
    #input("Check things work")
    logits = logits_BT(reward1, reward2, threshold=reward_diff_threshold)
    target_probs = scores_to_target_probs(scores1, scores2, reward_diff_threshold=reward_diff_threshold, assume_qualitative_labels=assume_qualitative_labels, check_undefined_label=check_undefined_label)
    print("LOGITS SHAPE", logits.shape)
    print("TARGET PROBS", target_probs.shape)
    assert target_probs.shape == logits.shape, f"Target probabilities shape {target_probs.shape} does not match logits shape {logits.shape}"	
    assert not th.any(logits.isnan()) and not th.any(logits.isinf()), f"Logits contain NaN or Inf values: {logits}"
    assert not th.any(target_probs.isnan()) and not th.any(target_probs.isinf()), f"Target probabilities contain NaN or Inf values: {target_probs}"	
    assert th.all(target_probs.detach() >= 0.0) and th.all(target_probs.detach() <= 1.0), f"Target probabilities should be in [0, 1], but got {target_probs}"

    loss = th.nn.functional.binary_cross_entropy_with_logits(
                # /sum(weights)
                logits, target_probs, reduction='none', reduce=False)
    assert loss.shape == reward1.shape, f"Expected loss shape {(reward1.shape[0],)}, got {loss.shape}"
    mean = th.mean(loss, dim=-2)
    assert mean.shape == (reward1.shape[-1],), f"Expected loss shape {(reward1.shape[-1],)}, got {loss.shape}"
    return mean

def value_system_loss(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold=50.0, assume_qualitative_labels=False, check_undefined_label=True):
    logits = logits_BT(reward1, reward2, threshold=reward_diff_threshold)
    target_probs = scores_to_target_probs(scores1, scores2, reward_diff_threshold, assume_qualitative_labels, check_undefined_label)

    loss = th.nn.functional.binary_cross_entropy_with_logits(
                # /sum(weights)
                logits, target_probs, reduction='none')
    assert loss.shape == reward1.shape, f"Expected loss shape {(reward1.shape[0],)}, got {loss.shape}"
    return th.mean(loss)


def mo_loss_function(logits, labels, pooled_logits, config: MORMForSequenceClassificationConfig =None, training_variables: MORMTrainingVariables =None, **kwargs):
    
    bsz = pooled_logits.size(0)

    jidx = th.arange(0, bsz, 2, device=pooled_logits.device)
    kidx = jidx + 1
    rewards_1 = pooled_logits[jidx]
    rewards_2 = pooled_logits[kidx]
    labels_1 = labels[jidx]
    labels_2 = labels[kidx]

    """print(labels.shape)
    print(logits.shape,pooled_logits.shape)
    input("SHAPES???")
    print(rewards_1, rewards_2)
    exit(0)"""
    
    assert pooled_logits.shape[-1] == config.num_values + 1
    assert labels.shape == pooled_logits.shape, f"Labels shape {labels.shape} does not match pooled logits shape {pooled_logits.shape}"

    gr_loss = grounding_loss(rewards_1[...,0:config.num_values], rewards_2[...,0:config.num_values], scores1=labels_1[...,0:config.num_values], scores2=labels_2[...,0:config.num_values], reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label)
    vs_loss = value_system_loss(rewards_1[...,-1],rewards_2[...,-1], scores1=labels_1[..., -1], scores2=labels_2[..., -1] , reward_diff_threshold=config.reward_diff_threshold, assume_qualitative_labels=config.assume_qualitative_labels, check_undefined_label=config.check_undefined_label)
    
    
    lag_gr_loss = th.dot(gr_loss, training_variables.lagrange_multipliers)
    with th.no_grad():
        weighting_factor = 1.0 / 1.0 + sum(training_variables.lagrange_multipliers)
    last_loss_original_unscaled = lag_gr_loss + vs_loss
    total_loss = weighting_factor * last_loss_original_unscaled
    
    training_variables.record_grounding_loss(gr_loss.detach().clone(), last_loss_original_unscaled.detach().clone())
        
    return total_loss


def mo_compute_loss_func(outputs, labels, training_variables: MORMTrainingVariables, config=None, **kwargs):

    return mo_loss_function(outputs.logits, labels.to(outputs.logits.device), outputs.logits, config=config, **kwargs)


class MORMForSequenceClassification(PreTrainedModel, GenericForSequenceClassification):
    base_model_prefix = "wrapped_model"
    supports_gradient_checkpointing = True
    
    
    def construct_value_layer(self, config: MORMForSequenceClassificationConfig):
        layers = []
        if hasattr(config.base_model, "score") and hasattr(config.base_model.score, "in_features"):
            input_size = config.base_model.score.in_features
        else:
            input_size = config.base_model.config.hidden_size
        for hidden_size in config.hidden_sizes: 
            layers.append(nn.Linear(input_size, hidden_size))
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
            layers.append(nn.Dropout(config.value_layer_dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, config.num_values))
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

        # Normalize across value dimensions to keep reward channels on a comparable scale.
        layers.append(nn.LayerNorm(config.num_values))
        return nn.Sequential(*layers)
    
    def __init__(self, config: MORMForSequenceClassificationConfig):
        super().__init__(config)
        
        self.reward_heads = self.construct_value_layer(config)
        self.full_model = config.base_model
        self.wrapped_model = config.base_model.base_model
        self.supports_gradient_checkpointing = hasattr(self.full_model, "gradient_checkpointing_enable")
        self.loss_function = mo_loss_function
        self.value_system_layer = ConvexAlignmentLayer(config.num_values, 1)
        self._freeze_base_model_keep_heads_trainable()
        
		#self.score_weight_head: ConvexAlignmentLayer = ConvexAlignmentLayer(num_values, 1)

    def _freeze_base_model_keep_heads_trainable(self) -> None:
        # Freeze pretrained weights and train only the custom reward/value-system heads.
        for param in self.full_model.parameters():
            param.requires_grad = False

        for param in self.reward_heads.parameters():
            param.requires_grad = True

        for param in self.value_system_layer.parameters():
            param.requires_grad = True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None):
        if not hasattr(self.full_model, "gradient_checkpointing_enable"):
            raise ValueError(f"{self.full_model.__class__.__name__} does not support gradient checkpointing.")
        return self.full_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        if hasattr(self.full_model, "gradient_checkpointing_disable"):
            return self.full_model.gradient_checkpointing_disable()
        return None

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.full_model, "is_gradient_checkpointing", False))

    def get_input_embeddings(self):
        return self.full_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.full_model.set_input_embeddings(value)
	
    def set_value_system_layer(self, layer: nn.Module):
        self.value_system_layer = layer
    
    def score(self, hidden_state):
        # This is used inside the GenericForSequenceClassification forward method.
        
        rewards = self.reward_heads(hidden_state) 
        vs_reward = self.value_system_layer.forward(rewards)
        all_rewards = th.cat([rewards, vs_reward], dim=-1)
        assert all_rewards.shape[-1] == self.config.num_labels, f"Expected rewards shape to have last dimension {self.config.num_labels}, but got {rewards.shape}"
        
        return all_rewards

    def forward(self, *args, **kwargs):
        outputs = GenericForSequenceClassification.forward(self, *args, **kwargs)
        print("FORWARD DONE")
        for param in self.full_model.parameters():
            assert not param.requires_grad 
        for param in self.wrapped_model.parameters():
            assert not param.requires_grad
        return outputs
    
