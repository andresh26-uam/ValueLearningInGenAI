
import dis

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from collections.abc import Iterator
from dataclasses import dataclass

from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Unpack

import numpy as np
import torch as th
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel
from transformers.utils import logging
from transformers.cache_utils import Cache

from vsllib.training_utils import MORMTrainingVariables
from vsllib.defines import MIN_EPSILON, NO_RATING_MASK, SCORE_DIFF_EPSILON, VALUE_LAYER_ACTIVATIONS, MOLossFunctions, MOLossFunctionsCategories

logger = logging.get_logger(__name__)

class LinearAlignmentLayer(th.nn.Linear):
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

    @th.compile
    def forward(self, input: th.Tensor) -> th.Tensor:
        # assert w_bounded.dtype == self.weight.dtype, f"Expected w_bounded dtype {self.weight.dtype}, but got {w_bounded.dtype}"
        # assert w_bounded.device == self.weight.device, f"Expected w_bounded device {self.weight.device}, but got {w_bounded.device}"
        return th.nn.functional.linear(input, self.get_alignment_layer())
        # assert input.shape[-1] == self.n_values, f"Expected output shape to have last dimension {self.n_values}, but got {output.shape}"
        # return output

    def get_alignment_layer(self):
        return self.weight
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))

    def get_weights(self):
        with th.no_grad():
            return self.get_alignment_layer().detach().clone().view(-1).cpu().tolist()


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


class MORMForSequenceClassificationConfig(PretrainedConfig):
    model_type = "morm_for_sequence_classification"
    has_no_defaults_at_init = True

    def __init__(
        self,
        # This will be set properly in the model init based on the tokenizer
        pad_token_id: int = "UNKNOWN",
        num_values: int = 3,
        hidden_sizes: list[int] = [1024, 1024, 1024],
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
        rew_center_coefficient: float = 0.0,
        gradient_accumulation_steps: int = 2,
        use_metrics_or_losses_for_lagrange_updates: str = "metrics",
        use_exponential_moving_average_or_optimum_targets: str = "optimum",
        grad_on_only_worst_value: bool = False,
        zero_constraint: bool = True,
        lambda_decay: float = 0.0,
        gather_train_metrics: bool = False,
        use_ideal_grounding_model: bool = False,
        dtype: str = "float16",
        base_model_name_or_path: Optional[str] = None,
        base_model_trust_remote_code: bool = True,
        base_model_num_labels: int = 1,
        use_base_model_heads: bool = False,
        base_model_reward_heads_module_name: str = None,
        base_model_value_system_module_name: str = None,
        base_model_reward_head_indices: list = None,
        loss_func_type: str = MOLossFunctions.DEFAULT,
        loss_func_kwargs: dict = None,
        lr_grounding: Optional[float] = None,
        lr_value_system: Optional[float] = None,
        lr_lambda: Optional[float] = None,
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

        if pad_token_id == "UNKNOWN":
            raise ValueError("pad_token_id must be set to a valid integer value corresponding to the tokenizer's pad token ID. It is currently set to 'UNKNOWN', which is not valid. Please set it to the correct value when initializing the config.")
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
        self.zero_constraint = zero_constraint
        self.rew_center_coefficient = rew_center_coefficient
        self.discordance_epsilon = discordance_epsilon
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

        loss_func_enum = MOLossFunctions(self.loss_func_type)

        if loss_func_enum not in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS:
            self.lr_grounding = 0.0
        else:
            assert lr_grounding is not None and lr_grounding > 0.0, f"Loss function type {loss_func_type} requires applying gradients on grounding parameters, but lr_grounding is set to {lr_grounding}. Please set lr_grounding to a positive value to enable optimization of grounding parameters."
            self.lr_grounding = lr_grounding

        if loss_func_enum not in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_VALUE_SYSTEM_WEIGHTS:
            self.lr_value_system = 0.0
        else:
            assert lr_value_system is not None and lr_value_system > 0.0, f"Loss function type {loss_func_type} requires applying gradients on value system parameters, but lr_value_system is set to {lr_value_system}. Please set lr_value_system to a positive value to enable optimization of value system parameters."
            self.lr_value_system = lr_value_system

        if loss_func_enum not in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_LAGRANGE_MULTIPLIERS:
            self.lr_lambda = 0.0
        elif loss_func_enum in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_PART_OF_GROUNDING_PARAMETERS and len(self.loss_func_type_kwargs.get('value_indices', [])) <= 1:
            self.lr_lambda = 0.0
        else:
            if lr_lambda is None:
                lr_lambda = lr_value_system
            assert lr_lambda is not None and lr_lambda > 0.0, f"Loss function type {loss_func_type} requires applying gradients on Lagrange multipliers, but lr_lambda is set to {lr_lambda}. Please set lr_lambda to a positive value to enable optimization of Lagrange multipliers."
            self.lr_lambda = lr_lambda

        self.lambda_decay = lambda_decay
        if loss_func_enum not in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_LAGRANGE_MULTIPLIERS:
            self.lambda_decay = 0.0
        self.base_model_reward_head_indices = base_model_reward_head_indices if base_model_reward_head_indices is not None else "use_base_model_value_system_module_name"

        
        super().__init__(num_labels=num_values + 1,
                         id2label=id2label, label2id=label2id, **kwargs)


LossFuncType = Callable[[th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor],
                         MORMForSequenceClassificationConfig, MORMTrainingVariables], th.Tensor]


def parse_loss_function(config: MORMForSequenceClassificationConfig) -> LossFuncType:
    return mo_loss_function


def accuracy_rewards_labels(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, threshold=50.0, assume_qualitative_labels=False, check_undefined_label=True, missing_mask=None, assume_torch=True, discordance_epsilon=MIN_EPSILON) -> th.Tensor:

    logits, targets, others = reward_pairs_and_scores_to_logits_and_targets(reward1, reward2, scores1, scores2, reward_diff_threshold=threshold,
                                                                            assume_qualitative_labels=assume_qualitative_labels, check_undefined_label=check_undefined_label, missing_mask=missing_mask, assume_torch=assume_torch)
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
        logits1 = (logits > discordance_epsilon) if hard_classification else (logits > 0)
        logits0 = (logits < -discordance_epsilon) if hard_classification else (logits < 0)
        target1 = (target_probs > 0.5 + score_diff_epsilon) if hard_classification else (target_probs > 0.5)
        target0 = (target_probs < 0.5 - score_diff_epsilon) if hard_classification else (target_probs < 0.5)

        mask1_1 = logits1 & target1
        mask1_2 = logits0 & target0
        
        equal_cases_logits = ~(logits1  | logits0) if not hard_classification else (logits <= discordance_epsilon) & (logits >= -discordance_epsilon)
        equal_cases_targets = ~(target1  |target0) if not hard_classification else (target_probs <= 0.5 + score_diff_epsilon) & (target_probs >= 0.5 - score_diff_epsilon)
        mask1_3 = equal_cases_logits & equal_cases_targets

        #print_logits_target_mismatches(logits, target_probs, equal_cases_logits, equal_cases_targets, all_defined_cases, assume_torch=assume_torch)
        

        mask05_1 = equal_cases_logits & ~equal_cases_targets
        mask05_2 = equal_cases_targets & ~equal_cases_logits

        mask05 = (mask05_1 | mask05_2) & all_defined_cases
        mask1 = (mask1_1 | mask1_2  | mask1_3) & all_defined_cases

        if assume_torch:
            mask = mask1.float()
            mask[mask05] = 0.5   
            factor = (all_defined_cases).float().sum(dim=0)
            positive_cases = mask.sum(dim=0)
        else:
            mask = mask1.astype(float)
            mask[mask05] = 0.5   
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

    logits_new, target_probs, others = reward_pairs_and_scores_to_logits_and_targets(
        rewards_1, rewards_2, labels_1, labels_2, 
        reward_diff_threshold=config.reward_diff_threshold, 
        assume_qualitative_labels=config.assume_qualitative_labels, 
        check_undefined_label=config.check_undefined_label, 
        assume_torch=assume_torch)
    return logits_new, target_probs, others

from vsllib.utils import print_tensor_and_grad_fn

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
                assert th.allclose(target_probs[mask_less], th.zeros_like(target_probs[mask_less]))
                assert th.allclose(target_probs[mask_greater], th.ones_like(target_probs[mask_greater]))
                assert th.allclose(target_probs[mask_equal], 0.5 * th.ones_like(target_probs[mask_equal]))
            else:
                target_probs[mask] = NO_RATING_MASK

    if assume_qualitative_labels:
        assert np.allclose(target_probs[mask_less & mask], np.zeros_like(target_probs[mask_less & mask]))
        assert np.allclose(target_probs[mask_greater & mask], np.ones_like(target_probs[mask_greater & mask]))
        assert np.allclose(target_probs[mask_equal & mask], 0.5 * np.ones_like(target_probs[mask_equal & mask]))

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
            
            discordance = th.full_like(logits, fill_value=discordance_epsilon)
            discordance = discordance.masked_fill(target_probs < 0.5, -discordance_epsilon)
            if missing_mask is not None:
                discordance = discordance.masked_fill(missing_mask, 0.0)
        logits_app = logits+discordance
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
            
            discordance = th.full_like(logits, fill_value=discordance_epsilon)
            discordance = discordance.masked_fill(target_probs < 0.5, -discordance_epsilon)
            if missing_mask is not None:
                discordance = discordance.masked_fill(missing_mask, 0.0)
        logits_app = logits+discordance
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


def reward_pairs_and_scores_to_logits_and_targets(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold=50.0, assume_qualitative_labels=False, check_undefined_label=True, assume_torch=True) -> tuple[th.Tensor, th.Tensor, Dict[str, Any]]:
    # assert check_undefined_label
    missing_mask = get_missing_rating_mask(
        scores1, scores2) if check_undefined_label else None
    logits_p = logits_BT(reward1, reward2, threshold=reward_diff_threshold, missing_mask=missing_mask,
                         assume_torch=assume_torch, check_undefined_label=missing_mask is not None)
    target_probs_p = scores_to_target_probs(scores1, scores2, reward_diff_threshold=reward_diff_threshold, assume_qualitative_labels=assume_qualitative_labels,
                                            check_undefined_label=missing_mask is not None, missing_mask=missing_mask, assume_torch=assume_torch)
    
    target_probs_p_ = scores_to_target_probs(scores1, scores2, reward_diff_threshold=reward_diff_threshold, assume_qualitative_labels=not assume_qualitative_labels,
                                            check_undefined_label=missing_mask is not None, missing_mask=missing_mask, assume_torch=assume_torch)

    rew_sum = reward1 + reward2
    rew_sum = rew_sum.masked_fill(
        missing_mask, 0.0) if missing_mask is not None and check_undefined_label else rew_sum

    others = {
        'missing_mask': missing_mask,
        'rew_sum': rew_sum,
    }
    others["target_probs_quantitative"] = target_probs_p_ if assume_qualitative_labels else target_probs_p
    others["target_probs_qualitative"] = target_probs_p if assume_qualitative_labels else target_probs_p_
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


def mo_loss_function(logits, labels, ideal_logits=None, config: MORMForSequenceClassificationConfig = None, training_variables: MORMTrainingVariables = None, **kwargs):

    logits, labels, others = rewards_and_labels_to_logits_and_targets(
        logits, labels, assume_torch=True, config=config)
    missing_mask = others.get('missing_mask', None)
    grounding_mask = missing_mask[..., 0:-1] if missing_mask is not None else None
    vs_mask = missing_mask[..., -1] if missing_mask is not None else None

    rew_sum = others.get('rew_sum', None)
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

    if MOLossFunctions(config.loss_func_type) in MOLossFunctionsCategories.REQUIRES_GRAD_FOR_ALL_GROUNDING_LOSSES:
        
        gr_loss = grounding_loss_logits(logits[..., 0:-1], labels[..., 0:-1], rew_sum=grounding_rew_sum, missing_mask=grounding_mask,
                                            check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, 
                                            rew_center_coefficient=config.rew_center_coefficient, 
                                            discordance_epsilon=config.discordance_epsilon, 
                                            activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)

    elif MOLossFunctions(config.loss_func_type) in MOLossFunctionsCategories.REQUIRES_GRAD_FOR_ONLY_SOME_GROUNDING_LOSSES:
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
        assert MOLossFunctions(config.loss_func_type) not in MOLossFunctionsCategories.REQUIRES_GRAD_FOR_SOME_OR_ALL_GROUNDING_LOSSES, f"Unexpected loss function type {config.loss_func_type} that does not fit into any grounding loss category"
        with th.no_grad():
            gr_loss = grounding_loss_logits(logits[..., 0:-1], labels[..., 0:-1], rew_sum=grounding_rew_sum, missing_mask=grounding_mask,
                                        check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient, discordance_epsilon=config.discordance_epsilon, 
                                        activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)

    if ideal_logits is not None:
        gr_loss_ideal = grounding_loss_logits(ideal_logits[..., 0:-1], labels[..., 0:-1], rew_sum=grounding_rew_sum, missing_mask=grounding_mask,
                                              check_undefined_label=config.check_undefined_label, return_metrics=use_metrics, rew_center_coefficient=config.rew_center_coefficient, discordance_epsilon=config.discordance_epsilon, 
                                              activate_disc_epsilon_for_loss=config.activate_discordance_epsilon_for_loss)

    if MOLossFunctions(config.loss_func_type) in MOLossFunctionsCategories.REQUIRES_GRAD_FOR_VALUE_SYSTEM_LOSS:
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
    if th.is_grad_enabled() and training_variables is not None:
        with th.no_grad():
            grl = gr_loss.detach()
            vsl = vs_loss.detach()
            grli = gr_loss_ideal.detach() if ideal_logits is not None else None
            training_variables.record_grounding_loss(
                gr_loss_detached=grl, vs_loss_detached=vsl, gr_loss_ideal_detached=grli)
    if use_metrics and training_variables is not None:
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
    return parse_loss_function(config)(outputs.logits, labels, ideal_logits=id_logits, config=config, training_variables=training_variables, **kwargs)


@dataclass
class SequenceClassifierOutputWithPastAndIdeal(SequenceClassifierOutputWithPast):
    ideal_logits: th.Tensor = None


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

class MORMForSequenceClassification(PreTrainedModel):
    config_class = MORMForSequenceClassificationConfig
    base_model_prefix = "full_model"
    supports_gradient_checkpointing = True

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

    def _build_base_model_from_config(self, config: MORMForSequenceClassificationConfig) -> AutoModelForSequenceClassification:
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
            getattr(config, "dtype", "float32"))
        trust_remote_code = bool(
            getattr(config, "base_model_trust_remote_code", True))
        base_cfg = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "config": base_cfg,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
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
        if self.use_base_model_heads and (MOLossFunctions(self.config.loss_func_type) in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_GROUNDING_OR_VALUE_SYSTEM_PARAMS):
            yield from self.full_model.parameters(recurse=recurse)

        if self.reward_heads is not None and (MOLossFunctions(self.config.loss_func_type) in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS):
            yield from self.reward_heads.parameters(recurse=recurse)
        if self.value_system_layer is not None and (MOLossFunctions(self.config.loss_func_type) in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_VALUE_SYSTEM_WEIGHTS):
            yield from self.value_system_layer.parameters(recurse=recurse)
        # yield from self.training_variables.parameters(recurse=recurse)
        if self.use_ideal_grounding_model and (MOLossFunctions(self.config.loss_func_type) in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS):
            yield from self.reward_heads_ideal.parameters(recurse=recurse)

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
        input_emb = base_model.get_input_embeddings() if hasattr(
            base_model, "get_input_embeddings") else None
        if input_emb is not None and hasattr(input_emb, "embedding_dim"):
            return int(input_emb.embedding_dim)
        if input_emb is not None and hasattr(input_emb, "weight"):
            return int(input_emb.weight.shape[-1])

        raise ValueError(
            "Could not infer hidden size for reward heads. Expected one of: "
            "score.in_features, config.hidden_size/d_model/n_embd/dim, or input embedding width."
        )

    def construct_value_layer(self, config: MORMForSequenceClassificationConfig, base_model: BaseModelOutputWithPast = None, n_outputs: int = 1, add_normalization: bool = False) -> nn.Sequential:
        layers = []
        model_device = self._module_device(base_model)
        model_dtype = self._resolve_torch_dtype(config.dtype)
        input_size = self._infer_base_hidden_size(base_model)

        try:
            intermediate_activation = VALUE_LAYER_ACTIVATIONS[
                config.value_layer_intermediate_activation]
        except KeyError:
            raise ValueError(
                f"Unsupported intermediate activation: {config.value_layer_intermediate_activation}")

        if intermediate_activation is None:
            raise ValueError(
                f"Unsupported intermediate activation: {config.value_layer_intermediate_activation}")
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size,
                          dtype=model_dtype, device=model_device))

            layers.append(intermediate_activation())
            if config.value_layer_dropout > 0.0:
                layers.append(nn.Dropout(config.value_layer_dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, n_outputs,
                      dtype=model_dtype, device=model_device))

        try:
            final_activation = VALUE_LAYER_ACTIVATIONS[config.value_layer_final_activation]
        except KeyError:
            raise ValueError(
                f"Unsupported final activation: {config.value_layer_final_activation}")
        if final_activation is not None:
            layers.append(final_activation())
        if add_normalization:
            layers.append(self.construct_value_normalization(config, base_model))
        return nn.Sequential(*layers)

    def construct_value_normalization(self, config: MORMForSequenceClassificationConfig, base_model: BaseModelOutputWithPast = None):
        # Normalize across value dimensions to keep reward channels on a comparable scale.
        model_device = self._module_device(base_model)
        model_dtype = self._resolve_torch_dtype(config.dtype)
        if config.layer_normalization == 'LayerNorm':
            return nn.LayerNorm(config.num_values, dtype=model_dtype, device=model_device)
        if config.layer_normalization == 'BatchNorm':
            return nn.BatchNorm1d(config.num_values, dtype=model_dtype, device=model_device)
        if config.layer_normalization == 'none':
            return nn.Identity()
        raise ValueError(
            f"Unsupported normalization: {config.layer_normalization}")

    def construct_reward_head(self, config: MORMForSequenceClassificationConfig, base_model: BaseModelOutputWithPast = None) -> MultiValueRewardHead:
        value_heads = nn.ModuleList([
            self.construct_value_layer(config, base_model)
            for _ in range(config.num_values)
        ])
        normalization = self.construct_value_normalization(config, base_model)
        return MultiValueRewardHead(value_heads=value_heads, normalization=normalization, optimized_head_indices=config.loss_func_type_kwargs.get('value_indices', None))


    def train(self, mode: bool = True):
        self._set_train_mode(train_mode=mode)
        self.forward_ideal_grounding = mode and self.use_ideal_grounding_model

        return super().train(mode)

    def __init__(self, config: MORMForSequenceClassificationConfig, base_model: AutoModelForSequenceClassification = None):
        super().__init__(config)
        if base_model is None:
            base_model = self._build_base_model_from_config(config)

        self.full_model = base_model
        self.supports_gradient_checkpointing = hasattr(
            self.full_model, "gradient_checkpointing_enable")
        model_device = self._module_device(self.full_model)
        model_dtype = self._module_dtype(self.full_model)
        self.num_values = config.num_values
        self.use_ideal_grounding_model = config.use_ideal_grounding_model
        self.use_base_model_heads = config.use_base_model_heads
        self.base_model_reward_head_indices = config.base_model_reward_head_indices
        self.base_model_rewards_attr_name = config.base_model_reward_heads_module_name
        self.base_model_score_attr_name = config.base_model_value_system_module_name

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
            if len(config.hidden_sizes) > 0:
                # This constructs one NN per value to avoid that changing one value loss parameter chagnes also affect others.
                constructor = self.construct_reward_head
            else:
                # In this case, the parameters of different value dimensions do not conflict.
                constructor = partial(
                    self.construct_value_layer, n_outputs=self.num_values, add_normalization=config.layer_normalization != 'none')
            
            self.reward_heads = constructor(config, base_model)
            if config.use_ideal_grounding_model:
                self.reward_heads_ideal = constructor(config, base_model)
            self.value_system_layer = ConvexAlignmentLayer(
                config.num_values, 1, device=model_device, dtype=model_dtype)
            self.reward_heads_ideal = None if not config.use_ideal_grounding_model else self.reward_heads_ideal

        # if config.training_variables_dtype == "float32" else th.float16 if config.training_variables_dtype == "float16" else self._resolve_torch_dtype(config.training_variables_dtype)
        training_variables_dtype = th.float32

        self.training_variables = MORMTrainingVariables(n_values=config.num_values, initial_lambda=1.0,
                                                        device=model_device, dtype=training_variables_dtype, grounding_loss_tendency_update_ratio=config.grounding_loss_tendency_update_ratio,
                                                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                                                        use_metrics_or_losses=config.use_metrics_or_losses_for_lagrange_updates,
                                                        use_exponential_moving_average_or_optimum_targets=config.use_exponential_moving_average_or_optimum_targets,
                                                        grad_on_only_worst_value=config.grad_on_only_worst_value,
                                                        zero_constraint=config.zero_constraint,
                                                        lambda_decay=config.lambda_decay
                                                        )
        self._set_train_mode(train_mode=True)
        self.forward_ideal_grounding = self.use_ideal_grounding_model

        self.loss_function = partial(parse_loss_function(
            self.config), training_variables=self.training_variables, config=self.config)

        self.post_init()

        # TODO: Apparetly this is much faster. See https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        self.zero_grad(set_to_none=True)
        # self.score_weight_head: ConvexAlignmentLayer = ConvexAlignmentLayer(num_values, 1)

    def _set_train_mode(self, train_mode: bool = True) -> None:
        # Freeze pretrained weights and train only the custom reward/value-system heads.
        possibly_change_train_mode_in_base_model = self.use_base_model_heads and (MOLossFunctions(self.config.loss_func_type) in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_GROUNDING_OR_VALUE_SYSTEM_PARAMS)
        train_mode_base_model = False
        if possibly_change_train_mode_in_base_model:
            train_mode_base_model = train_mode
        
        self.full_model.train(train_mode_base_model)

        for param in self.grounding_parameters():
            param.requires_grad = train_mode

        for param in self.value_system_parameters():
            param.requires_grad = train_mode
        # This is set to True inside training_variables in prepare_for_optimizer_step method.
        self.training_variables.requires_grad_(False)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.full_model, "is_gradient_checkpointing", False))

    def set_value_system_layer(self, layer: nn.Module):
        self.value_system_layer = layer

    def grounding_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        params = []
        if MOLossFunctions(self.config.loss_func_type) in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS:
            if self.reward_heads is not None:
                params.extend(self.reward_heads.parameters(recurse=recurse))
            elif self.use_base_model_heads:
                # In base model mode, we assume all parameters require grad, but we only want to return the reward head parameters for optimization.
                params.extend(self.full_model.parameters(recurse=recurse))
        return iter(params)

    def value_system_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        params = []
        if MOLossFunctions(self.config.loss_func_type) in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_VALUE_SYSTEM_WEIGHTS:
            if self.value_system_layer is not None:
                params.extend(
                    self.value_system_layer.parameters(recurse=recurse))
            elif self.use_base_model_heads:
                # In base model mode, we assume all parameters require grad, but we only want to return the reward head parameters for optimization.
                params.extend(self.full_model.parameters(recurse=recurse))
        return iter(params)

    def score(self, hidden_state, reward_heads='normal') -> th.Tensor:
        if reward_heads == 'ideal' and self.use_ideal_grounding_model:
            reward_heads = self.reward_heads_ideal
            raise ValueError(f"Unexpected loss function type {self.config.loss_func_type} that does not fit into any grounding loss category, cannot determine whether to apply grad on grounding parameters or not.")
        else:
            reward_heads = self.reward_heads

        if MOLossFunctions(self.config.loss_func_type) in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS:
            rewards = reward_heads(hidden_state)
        else:
            raise ValueError(f"Unexpected loss function type {self.config.loss_func_type} that does not fit into any grounding loss category, cannot determine whether to apply grad on grounding parameters or not.")
            with th.no_grad():
                rewards = reward_heads(hidden_state)

        if self.value_system_layer is not None:
            if MOLossFunctions(self.config.loss_func_type) in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_VALUE_SYSTEM_WEIGHTS:
                vs_reward = self.value_system_layer.forward(rewards)
            else:
                raise ValueError(f"Unexpected loss function type {self.config.loss_func_type} that does not fit into any grounding loss category, cannot determine whether to apply grad on grounding parameters or not.")
                with th.no_grad():
                    vs_reward = self.value_system_layer.forward(rewards)
                
            all_rewards = th.cat([rewards, vs_reward], dim=-1)
        else:
            all_rewards = rewards
        return all_rewards

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

    def forward(self, *args, **kwargs) -> SequenceClassifierOutputWithPast | SequenceClassifierOutputWithPastAndIdeal:
        """perfect_debug_forward = True #DEBUG ONLY.
        if perfect_debug_forward: 
            #print(kwargs.keys())
            embeddings = kwargs.pop('embedding')
            all_rewards = self.score(embeddings)
            logits = kwargs.pop("labels", None)
            missing = get_missing_rating_mask(logits)
            logits = logits  + th.tensor([0.0], requires_grad=True) + all_rewards*0.1 - all_rewards*0.1 # Just to have a tensor that requires grad for testing.  
            logits = logits.masked_fill(missing, float("-inf"))
            return SequenceClassifierOutputWithPast(logits=logits)"""

        if self.use_base_model_heads:
            kwargs.pop("labels", None)
            kwargs.pop("embedding", None)
            kwargs.pop("context_embedding", None)

            kwargs.pop("return_loss", None)
            # TODO this might not work with other models
            kwargs.pop("num_items_in_batch", None)
            base_output = self.full_model(*args, **kwargs, return_dict=True)
            pooled_logits = self._extract_logits_from_base_output(base_output)
            del base_output
            # print("LABELS SHAPE", labels.shape)

            return SequenceClassifierOutputWithPast(
                logits=pooled_logits,
                past_key_values=getattr(base_output, "past_key_values", None),
                hidden_states=getattr(base_output, "hidden_states", None),
                attentions=getattr(base_output, "attentions", None),
            )

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

            # sq = GenericForSequenceClassification.forward(self, *args, **kwargs)
            kwargs["score_mode"] = "normal"
            sq = self.generic_forward(*args, **kwargs)
            if self.forward_ideal_grounding:
                # = GenericForSequenceClassification.forward(self, *args, **kwargs)
                kwargs["score_mode"] = "ideal"
                ideal_sq = self.generic_forward(*args, **kwargs)
                kwargs["score_mode"] = "normal"
                return SequenceClassifierOutputWithPastAndIdeal(logits=sq.logits, ideal_logits=ideal_sq.logits)

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
        logits = self.score(hidden_states, score_mode=score_mode) # Change here.

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

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
