

from abc import abstractmethod
from copy import deepcopy
from typing import Tuple
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from ordered_set import OrderedSet


from transformers.utils import PaddingStrategy

import torch as th

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vsllib.defines import MOLossFunctionsCategories, MOLossFunctions

from vsllib.utils import to_float

from torch.optim.lr_scheduler import ReduceLROnPlateau

from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional
import torch as th
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer as Optimizer

from transformers.trainer import *


from ordered_set import OrderedSet
from vsllib.defines import MOLossFunctions, MOLossFunctionsCategories


from vsllib.utils import to_float

@th.compile
def normalizing_params(used_mults, vs_coeff, dtype=th.float32) -> Tuple[th.Tensor, th.Tensor]:
        mults = th.nn.functional.softmax(th.cat([used_mults, vs_coeff], dim=0), dim=0, dtype=dtype)
        return mults[:-1], mults[-1]

@th.compile
def norm_penalty( lags, vs_coeff, penalty_coeff) -> th.Tensor:
        return penalty_coeff*th.norm(th.cat([lags, vs_coeff],dim=0), p=2)
    




@dataclass
class MORewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    use_embeddings: bool = False
    pad_to_multiple_of: Optional[int] = 16
    return_tensors: str = "pt"
    dtype: Optional[th.dtype] = None

    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []


        for feature in features:
            pair_labels = feature["labels"]
            c = feature.get("context_embedding", None) if self.use_embeddings else None
            dic1 = {
                    "input_ids": feature["input_ids_1"],
                    "attention_mask": feature["attention_mask_1"],
                    "labels": pair_labels[0],
                }
            dic2 = {
                    "input_ids": feature["input_ids_2"],
                    "attention_mask": feature["attention_mask_2"],
                    "labels": pair_labels[1],
                }
            if self.use_embeddings:
                dic1["embedding"] = feature.get("embedding_1", None)
                dic1["context_embedding"] = c
                dic2["embedding"] = feature.get("embedding_2", None)
                dic2["context_embedding"] = c
            merged_features.append(
                dic1
            )
                
            merged_features.append(
                dic2
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            
        )
        
        batch["return_loss"] = True
        batch["embedding"] = batch["embedding"].to(dtype=self.dtype) if "embedding" in batch.keys() else None
        batch["context_embedding"] = batch["context_embedding"].to(dtype=self.dtype) if "context_embedding" in batch.keys() else None
        """if self.use_embeddings:
            assert batch["embedding"] is not None, "Expected 'embedding' key in the batch when use_embeddings is True."
            assert batch["context_embedding"] is not None, "Expected 'context_embedding' key in the batch when use_embeddings is True."
        """
        return batch




class MORMTrainingVariables(th.nn.Module):
    def _iter_cached_field_names(self) -> Iterable[str]:
        for name in vars(self):
            if name.startswith("_cache"):
                yield name


    
    def _collect_train_metrics_for_logging(self) -> Dict[str, float]:
        self: MORMTrainingVariables = self
        result: Dict[str, float] = {}


        if self.last_accumulated_coherences is not None:
            for i in range(len(self.last_accumulated_coherences)):
                if self.last_accumulated_coherences[i] is not None:
                    result[f"coherence_v{i}"] = to_float(self.last_accumulated_coherences[i])
        if self.last_accumulated_coherences_ideal is not None:
            for i in range(len(self.last_accumulated_coherences_ideal)):
                if self.last_accumulated_coherences_ideal[i] is not None:
                    result[f"coherence_v{i}_ideal"] = to_float(self.last_accumulated_coherences_ideal[i])
        if self.last_accumulated_representativeness is not None:
            result["representativeness"] = to_float(self.last_accumulated_representativeness)
        
        if self.last_accumulated_grounding_loss is not None:
            result["grounding_loss"] = to_float(self.last_accumulated_grounding_loss)
            for v in range(len(self.last_accumulated_grounding_loss)):
                result[f"grounding_loss_{v}"] = to_float(self.last_accumulated_grounding_loss[v])
            
        if self.last_accumulated_vs_loss is not None:
            result["value_system_loss"] = to_float(self.last_accumulated_vs_loss)
        multipliers = self.get_multipliers(used_only=False)[0]

        for i in range(len(multipliers)):
            if multipliers[i] is not None:
                result[f"lagrange_multiplier_{i}"] = to_float(multipliers[i])
        if self.maximum_coherences_tendency is not None:
            result["maximum_coherences_tendency"] = to_float(self.maximum_coherences_tendency)
        if self.minimum_grounding_loss_tendency is not None:
            result["minimum_grounding_loss_tendency"] = to_float(self.minimum_grounding_loss_tendency)
        
        if self.coherences_tendency is not None:
            result["avg_coherences_tendency"] = to_float(self.coherences_tendency)
        if self.grounding_loss_tendency is not None:
            result["avg_grounding_loss_tendency"] = to_float(self.grounding_loss_tendency)
        return result
    
    def forward(self, grounding_losses: th.Tensor, vs_losses: th.Tensor, target_gr_loss: th.Tensor = None, selected_indices: list = None) -> th.Tensor:
        
        used_mults, vs_coeff = self.normalize_coefficients(selected_indices=selected_indices)
        if __debug__:
            if selected_indices is not None:
                assert used_mults.shape == (len(selected_indices),), f"Expected used_mults shape to match grounding_losses shape, but got {used_mults.shape} and {grounding_losses.shape}"
            else:
                assert used_mults.shape == grounding_losses.shape, f"Expected used_mults shape to match grounding_losses shape, but got {used_mults.shape} and {grounding_losses.shape}"
        
        
        
        used_grounding_losses = grounding_losses[selected_indices] if selected_indices is not None else grounding_losses
        
        if target_gr_loss is None or self.zero_constraint:
            lag_gr_loss = th.dot(used_mults, used_grounding_losses)

        else:
            lag_gr_loss = th.dot(used_mults, th.maximum(used_grounding_losses - target_gr_loss[selected_indices], th.zeros_like(used_grounding_losses)))
        
        #last_loss_original_unscaled = lag_gr_loss + vs_loss
        total_loss = lag_gr_loss +  (vs_losses if vs_losses is not None else 0.0)*vs_coeff
        self._last_selected_indices = selected_indices
        return total_loss
    
    def _apply(self, fn, recurse=True) -> Any:
        def move_list(lst):
            if lst is None:
                return None
            return [fn(x) if isinstance(x, th.Tensor) else x for x in lst]

        for field_name in self._iter_cached_field_names():
            field_value = getattr(self, field_name, None)
            if isinstance(field_value, list) or field_value is None:
                setattr(self, field_name, move_list(field_value))

        return super()._apply(fn, recurse=recurse)

    
    
    def normalize_coefficients(self, selected_indices: list = None) -> th.Tensor:
        used_mults = self.lagrange_multipliers[selected_indices] if selected_indices is not None else self.lagrange_multipliers
        
        return normalizing_params(used_mults, self.vs_coeff, dtype=self.lagrange_multipliers.dtype)

    def get_multipliers(self, used_only=False) -> Tuple[th.Tensor, th.Tensor]:
        return self.normalize_coefficients(selected_indices=self._last_selected_indices if used_only else None)
        #union_mults_s = th.nn.functional.softmax(union_mults, dim=0)
            
    def __init__(self, n_values: int, initial_lambda: int =1.0 , 
                 device: th.DeviceObjType|str ='cpu', dtype: th.Type = th.float32, 
                 grounding_loss_tendency_update_ratio: float = 0.01, 
                 gradient_accumulation_steps=10, 
                 metric_buffer_size=3, use_metrics_or_losses='metrics', 
                 use_exponential_moving_average_or_optimum_targets="optimum",
                 grad_on_only_worst_value=False, zero_constraint: bool = True,
                 lambda_decay: float = 1e-9):
        
        super().__init__()
        self.lagrange_multipliers = th.nn.Parameter(th.tensor([initial_lambda]*n_values,  device=device, dtype=dtype), requires_grad=False)
        self.vs_coeff = th.nn.Parameter(th.tensor([initial_lambda], device=device, dtype=dtype), requires_grad=False)

        self._last_selected_indices = None

        self.zero_constraint = zero_constraint
        self.minimum_grounding_loss_tendency: th.Tensor | None = None
        self.maximum_coherences_tendency: th.Tensor | None = None

        self.minimum_vs_loss_tendency: th.Tensor | None = None
        self.maximum_representativeness_tendency: th.Tensor | None = None

        self.last_accumulated_grounding_loss: th.Tensor | None = None
        self.last_accumulated_vs_loss: th.Tensor | None = None

        self.last_accumulated_grounding_loss_ideal: th.Tensor | None = None
        self.last_accumulated_coherences_ideal: th.Tensor | None = None
        self.last_accumulated_coherences: th.Tensor | None = None
        self.last_accumulated_representativeness: th.Tensor | None = None

        self.metric_buffer_size = metric_buffer_size
        self.loss_metric_tendency_update_ratio = grounding_loss_tendency_update_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lambda_decay = lambda_decay
        self.initial_lambda = initial_lambda
        self._cached_groundings = []
        self._cached_groundings_ideal = []
        self._cached_vs_losses = [] 
        self._cached_representativeness = []
        self._cached_coherences = []
        self._cached_coherences_ideal = []
        self._cached_avg_coherence = [] 

        self.use_metrics_or_losses = use_metrics_or_losses 
        self.use_exponential_moving_average_or_optimum_targets = use_exponential_moving_average_or_optimum_targets
        self.grad_on_only_worst_value = grad_on_only_worst_value

        self.historic_eval_metrics = {}
            
    
    def prepare_for_optimizer_step(self) -> None:
        coeff: th.Tensor

        self.requires_grad_(True)
        with th.no_grad():
                #assert self.lagrange_multipliers[vi].grad is not None, f"Lagrange multiplier {vi} gradient is None before optimizer step."
            self.update_loss_tendencies()
            self.update_metrics_tendencies()

            if self.use_exponential_moving_average_or_optimum_targets == "optimum":
                coherences_tendency = self.maximum_coherences_tendency
                grounding_loss_tendency = self.minimum_grounding_loss_tendency
                representativeness_tendency = self.maximum_representativeness_tendency
                vs_loss_tendency = self.minimum_vs_loss_tendency
            elif self.use_exponential_moving_average_or_optimum_targets == "average":
                coherences_tendency = self.coherences_tendency
                grounding_loss_tendency = self.grounding_loss_tendency
                representativeness_tendency = self.representativeness_tendency
                vs_loss_tendency = self.vs_loss_tendency
            else:
                raise ValueError(f"Invalid value for use_exponential_moving_average_or_minimum_targets: {self.use_exponential_moving_average_or_optimum_targets}. Expected 'minimum' or 'average'.")
            if self.use_metrics_or_losses == 'metrics':
                # Use metrics for the optimizer step
                gr_ideal_diff = -(self.last_accumulated_coherences - coherences_tendency).detach()
                vs_ideal_diff = -(self.last_accumulated_representativeness - representativeness_tendency).detach() if self.last_accumulated_representativeness is not None else None
                #coeff = gr_ideal_diff #/ lag_sum
                
                
            elif self.use_metrics_or_losses == 'losses':
                gr_ideal_diff = (self.last_accumulated_grounding_loss - grounding_loss_tendency).detach()
                vs_ideal_diff = (self.last_accumulated_vs_loss - vs_loss_tendency).detach() if self.last_accumulated_vs_loss is not None else None
                # This is the derivative w.r.t. lambda of "1/(1+lambda) * (lambda(gr_loss - gr_ideal) + vs_loss)".
            else:
                raise ValueError(f"Invalid value for use_metrics_or_losses: {self.use_metrics_or_losses}. Expected 'metrics' or 'losses'.")    
                #should be...? 1) forward = self.forward(grounding_losses=gr_ideal_diff, vs_losses=self.last_accumulated_vs_loss)
                #should be...? 2) coeff = ((gr_ideal_diff*(lag_sum)) - 1*(forward))/th.pow(lag_sum, 2) 
                
                # Use losses for the optimizer step
            #coeff = ((gr_ideal_diff*(lag_sum)) - 1*(forward))/th.pow(lag_sum, 2) #should be...? 2)
        
        was_grad_none = self.lagrange_multipliers.grad is None
        
        if self._last_selected_indices is not None:
            norm_penalty_ =  norm_penalty(self.lagrange_multipliers[self._last_selected_indices], self.vs_coeff, self.lambda_decay)
        else:
            norm_penalty_ =  norm_penalty(self.lagrange_multipliers, self.vs_coeff, self.lambda_decay)
        with th.no_grad():
            if th.all(gr_ideal_diff < 0.0):
                # This means all groundings are below their ideal grounding losses (or above their ideal metrics, depending on the mode), so we don't need to apply gradients to push them down further, and can just focus on the value system loss if present.
                pass
            else:
                vs_ideal_diff = None

        forward = -self.forward(grounding_losses=gr_ideal_diff, vs_losses=vs_ideal_diff, selected_indices=self._last_selected_indices) + norm_penalty_
        forward.backward()
        coeff = self.lagrange_multipliers.grad.detach()
        #coeff = gr_ideal_diff  / lag_sum
        
        
        if self.grad_on_only_worst_value:
            worst_idx = th.argmax(gr_ideal_diff).detach(), 
            worst_val = coeff[worst_idx].detach().item()
            coeff.zero_()
            coeff[worst_idx] = worst_val
            #print("WORST VI", worst_idx, "WITH COEFF", worst_val)
            if was_grad_none:
                self.lagrange_multipliers.grad = coeff
            else:
                self.lagrange_multipliers.grad += coeff 
                #self.lagrange_multipliers.grad = coeff

        #grad = th.clamp(coeff, max=0.0, min=-1000.0)
        
        #assert th.all(grad <= 0.0), f"Expected all gradients to be non-positive, but got {grad}"
        
            
    def zero_grad(self, set_to_none: bool = True) -> None:
        set_to_none = True # TODO: Apparetly this is much faster. See https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        self.lagrange_multipliers.grad = None
        self.vs_coeff.grad = None
        #return super().zero_grad(set_to_none)
    def requires_grad_(self, requires_grad: bool = True):
        self.lagrange_multipliers.requires_grad_(requires_grad)
        self.vs_coeff.requires_grad_(requires_grad)
    def post_optimizer_step(self) -> None:
        """with th.no_grad():
            
            #self.normalize_coefficients()

            if self.lambda_decay > 0.0: # UNUSED.
                self.vs_coeff.data += self.lambda_decay.data * self.vs_coeff"""
                #self.lagrange_multipliers.data = th.clamp(update, min=self.initial_lambda, max=1000.0)
                #self.normalize_coefficients()
        self.zero_grad(set_to_none=True)
        
        

    def reset_lagrange_gradients(self) -> None:
        with th.no_grad():
            self.requires_grad_(False)
            self.zero_grad(set_to_none=True)
            self.last_accumulated_grounding_loss = None
            self.last_accumulated_grounding_loss_ideal = None
            self.last_accumulated_coherences_ideal = None
            self.last_accumulated_coherences = None
            self.last_accumulated_representativeness = None

            
            for field_name in self._iter_cached_field_names():
                field_value = getattr(self, field_name, None)
                if isinstance(field_value, list):
                    setattr(self, field_name, [])


    def update_metrics_tendencies(self) -> None:
        with th.no_grad():
            self.last_accumulated_coherences = th.stack(self._cached_coherences).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
            
            self.last_accumulated_representativeness = th.stack(self._cached_representativeness).mean().detach()

            coherence_target =  self.last_accumulated_coherences
            if len(self._cached_coherences_ideal) > 0:
                self.last_accumulated_coherences_ideal = th.stack(self._cached_coherences_ideal).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()

                coherence_target = th.maximum(self.last_accumulated_coherences, self.last_accumulated_coherences_ideal)
            if self.maximum_coherences_tendency is None:
                self.maximum_coherences_tendency = th.full_like(coherence_target, fill_value=0.5).detach()
                self.maximum_representativeness_tendency = th.full_like(self.last_accumulated_representativeness, fill_value=0.5).detach()
                self.coherences_tendency = th.full_like(coherence_target, fill_value=0.5).detach()
                self.representativeness_tendency = th.full_like(self.last_accumulated_representativeness, fill_value=0.5).detach()
            else:
                maximum = th.maximum(coherence_target, self.maximum_coherences_tendency)
                
                self.maximum_coherences_tendency = (th.multiply(maximum, self.loss_metric_tendency_update_ratio) + th.multiply(self.maximum_coherences_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.coherences_tendency = (th.multiply(coherence_target, self.loss_metric_tendency_update_ratio) + th.multiply(self.maximum_coherences_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                
                maximum_repr = th.maximum(self.last_accumulated_representativeness, self.maximum_representativeness_tendency)
                self.maximum_representativeness_tendency = (th.multiply(maximum_repr, self.loss_metric_tendency_update_ratio) + th.multiply(self.maximum_representativeness_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.representativeness_tendency = (th.multiply(self.last_accumulated_representativeness, self.loss_metric_tendency_update_ratio) + th.multiply(self.maximum_representativeness_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()

    def update_loss_tendencies(self) -> Optional[th.Tensor]:
        
        with th.no_grad():
            self.last_accumulated_grounding_loss = th.stack(self._cached_groundings).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
            if len(self._cached_groundings_ideal) > 0:
                self.last_accumulated_grounding_loss_ideal = th.stack(self._cached_groundings_ideal).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
            else:
                self.last_accumulated_grounding_loss_ideal = self.last_accumulated_grounding_loss.detach()
            self.last_accumulated_vs_loss = th.stack(self._cached_vs_losses).mean().detach()

            minimum_actual = th.minimum(self.last_accumulated_grounding_loss, self.last_accumulated_grounding_loss_ideal).detach()
            
            if self.minimum_grounding_loss_tendency is None:
                self.minimum_grounding_loss_tendency = th.full_like(minimum_actual, fill_value=th.max(minimum_actual).float()).detach()
                self.minimum_vs_loss_tendency = th.full_like(self.last_accumulated_vs_loss, fill_value=th.max(self.last_accumulated_vs_loss).float()).detach()
                self.grounding_loss_tendency = th.full_like(minimum_actual, fill_value=th.max(minimum_actual).float()).detach()
                self.vs_loss_tendency = th.full_like(self.last_accumulated_vs_loss, fill_value=th.max(self.last_accumulated_vs_loss).float()).detach()
            else:
                minimum = th.minimum(minimum_actual, self.minimum_grounding_loss_tendency)
                minimum_vs = th.minimum(self.last_accumulated_vs_loss, self.minimum_vs_loss_tendency)

                self.minimum_grounding_loss_tendency = (th.multiply(minimum, self.loss_metric_tendency_update_ratio) + th.multiply(self.minimum_grounding_loss_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.minimum_vs_loss_tendency = (th.multiply(minimum_vs, self.loss_metric_tendency_update_ratio) + th.multiply(self.minimum_vs_loss_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.grounding_loss_tendency = (th.multiply(minimum_actual, self.loss_metric_tendency_update_ratio) + th.multiply(self.minimum_grounding_loss_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.vs_loss_tendency = (th.multiply(self.last_accumulated_vs_loss, self.loss_metric_tendency_update_ratio) + th.multiply(self.minimum_vs_loss_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()

    def record_metrics(self, metrics: Dict[str, float|list], metric_type: Literal["train", "validation"] = "train") -> None:
        # This is called at the end of each evaluation phase, and records the grounding loss and vs loss for the last evaluation phase, which will be used to update the Lagrange multipliers before the next optimizer step.
        
        if metric_type == "validation":
            for key in metrics.keys():
                if key not in self.historic_eval_metrics:
                    self.historic_eval_metrics[key] = []
                else:
                    self.historic_eval_metrics[key].append(metrics[key])
        else:
            with th.no_grad():

                for field in self._iter_cached_field_names():
                    field_value = getattr(self, field, None)
                    if len(field_value) > self.gradient_accumulation_steps:
                        field_value.pop(0)
                
                coherences = metrics.get("coherences", None)
                if coherences is not None:
                    self._cached_coherences.append(coherences)
                    
                    avg_c = metrics.get("avg_coherence", None)
                    if avg_c is not None:
                        self._cached_avg_coherence.append(avg_c)
                represent = metrics.get("representativeness", None)     
                if represent is not None:
                    self._cached_representativeness.append(represent)
                coherences_ideal = metrics.get("coherences_ideal", None)
                if coherences_ideal is not None:
                    self._cached_coherences_ideal.append(coherences_ideal)

    def record_grounding_loss(self, gr_loss_detached: th.Tensor, vs_loss_detached: th.Tensor, gr_loss_ideal_detached=None) -> None:
        with th.no_grad():
            # This estimates the minimum obtainable loss for each value. The optimizer will take this into account
            for field in self._iter_cached_field_names():
                field_value = getattr(self, field, None)
                if len(field_value) > self.gradient_accumulation_steps:
                    print("Popping from", field)
                    field_value.pop(0)
            self._cached_groundings.append(gr_loss_detached)
            self._cached_vs_losses.append(vs_loss_detached)
            if gr_loss_ideal_detached is not None:
                self._cached_groundings_ideal.append(gr_loss_ideal_detached)



def _create_sub_optimizer(params: OrderedSet, lr: float, sub_optimizer_class: type[th.optim.Optimizer], optimizer_kwargs: dict) -> th.optim.Optimizer:
        copyargs= deepcopy(optimizer_kwargs)
        copyargs['lr'] = lr
        #copyargs['learning_rate'] = lr_grounding
        
        return sub_optimizer_class(
            params, **copyargs)


class VSLOptimizer(th.optim.Optimizer):
    def __init__(self, params_gr: th.ParameterDict, params_vs: th.ParameterDict, n_values: int, lr_grounding=None, lr_value_system=None, sub_optimizer_class=th.optim.Adam,  **optimizer_kwargs):
        
        self.lr_grounding = lr_grounding
        self.lr_value_system = lr_value_system
        defaults = dict(lr_grounding=lr_grounding,
                        lr_value_system=lr_value_system)

        self.optimizer_kwargs = optimizer_kwargs
        self.n_values = n_values

        params_gr = OrderedSet(params_gr)
        params_vs = OrderedSet(params_vs)

        self.params_gr = params_gr
        self.params_vs = params_vs

        print("SUBOPTIMIZER CLASS:", sub_optimizer_class)
        self.sub_optimizer_class = sub_optimizer_class

        if params_vs and len(params_gr) > 0:
            self.optimx = _create_sub_optimizer(params_gr, lr_grounding, self.sub_optimizer_class, self.optimizer_kwargs)
        else:
            self.optimx = None
        if params_vs and len(params_vs) > 0:
            self.optimy = _create_sub_optimizer(params_vs, lr_value_system, self.sub_optimizer_class, self.optimizer_kwargs)
        else:
            self.optimy = None
        # TODO: SCHEDULER COSINE...? ALSO HANDLE SUBOPTIMIZER self.optimx_scheduler.step()
        # self.optimy_scheduler.step()
        all_params = [*params_gr, *params_vs]
        if len(all_params) == 0:
            all_params = [th.nn.Parameter(th.empty(0), requires_grad=True)] # Dummy parameter for initialization.
            print("WARNING: No parameters provided to VSLOptimizer. Initializing with dummy parameter.")
        super(VSLOptimizer, self).__init__(all_params, defaults)

    

    def get_state(self, copy=False)-> Dict[str, Any]:
        return {}

    def set_state(self, state)-> None:
        return

    
    @abstractmethod
    def zero_grad(self, set_to_none=True)-> None:
        set_to_none = True # TODO: Apparetly this is much faster. See https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        super().zero_grad(set_to_none)
        if self.optimx is not None:
            self.optimx.zero_grad(set_to_none)
        if self.optimy is not None:
            self.optimy.zero_grad(set_to_none)
        return None

    @abstractmethod
    def step(self, closure=None)-> None:
        if self.optimx is not None:
            self.optimx.step()
        if self.optimy is not None:
            self.optimy.step()
        return None

class ConstrainedOptimizer(VSLOptimizer):
    def __init__(self, params, params_gr, params_vs, n_values, max_grad_norm, params_gr_ideal=None, lr_grounding=None,
                 lr_value_system=None, lr_lambda=None, initial_lambda=1.0, inner_optimization_iterations=3,
                 loss_func_type: MOLossFunctions=MOLossFunctions.DEFAULT, loss_func_type_kwargs: dict = {},
                 training_variables: MORMTrainingVariables = None,
                 sub_optimizer_class=th.optim.Adam, **optimizer_kwargs):
        
        if MOLossFunctions(loss_func_type) in MOLossFunctionsCategories.NEEDS_NO_GRAD_ON_VALUE_SYSTEM_WEIGHTS:
            lr_value_system = 0.0
        if MOLossFunctions(loss_func_type) in MOLossFunctionsCategories.NEEDS_NO_GRAD_ON_ALL_GROUNDINGS:
            lr_grounding = 0.0

        super(ConstrainedOptimizer, self).__init__(params_gr=params_gr, params_vs=params_vs, n_values=n_values,
                                                   lr_grounding=lr_grounding, lr_value_system=lr_value_system, sub_optimizer_class=sub_optimizer_class, **optimizer_kwargs)
        if params_gr_ideal is not None:
            assert len(params_gr) == len(params_gr_ideal), "Grounding parameters and ideal grounding parameters must have the same length."
            
            self.params_gr_ideal = params_gr_ideal
            self.optimx_ideal = _create_sub_optimizer(params_gr_ideal, lr_grounding, self.sub_optimizer_class, optimizer_kwargs)

        self.lr_lambda = lr_lambda if lr_lambda is not None else lr_value_system * 10.0
        if (MOLossFunctions(loss_func_type) in MOLossFunctionsCategories.NEEDS_NO_GRAD_ON_LAGRANGE_MULTIPLIERS) or (len(loss_func_type_kwargs.get('value_indices', [])) == 1 and MOLossFunctions(loss_func_type) == MOLossFunctions.ONLY_VALUES_IN_KWARGS):
            self.lr_lambda = 0.0

        self.loss_func_type = MOLossFunctions(loss_func_type)
        self.loss_func_kwargs=loss_func_type_kwargs

        self.initial_lambda = initial_lambda
        self.max_grad_norm = max_grad_norm
        self.training_variables: MORMTrainingVariables = training_variables

        self.current_iteration = 0
        self.inner_optimization_iterations = inner_optimization_iterations # Number of inner optimization steps for the value system per outer step.


        
        if self.lr_lambda > 0:
            self.optim_lambdas = th.optim.SGD(
                self.training_variables.parameters(), lr=self.lr_lambda, weight_decay=0.0)
        self.time = 0
        self.training_variables.reset_lagrange_gradients()
    def zero_grad(self, set_to_none=True)-> None:
        set_to_none=True
        super().zero_grad(set_to_none)
        if self.lr_lambda > 0:
            self.training_variables.requires_grad_(False)
            self.training_variables.zero_grad(set_to_none)
            self.optim_lambdas.zero_grad(set_to_none)
        if hasattr(self, 'optimx_ideal'):
            self.optimx_ideal.zero_grad(set_to_none)
        return None
    
    def custom_backward(self, loss_gr, loss_gr_ideal, loss_vs, **kwargs) -> th.Tensor:
        
        if __debug__:
            x = self.params_gr # Possibly need flatten into single tensor.
            
            w = self.params_vs
            #self.zero_grad()    
            # Just optimize the value system for a number of iterations before doing the full constrained optimization step.
            
            #print("BEFORE", x[0].data[0:10])
            if len(w) > 0:
                assert w[0] is self.optimy.param_groups[0]['params'][0], "Value system parameters do not match those in the optimizer"
                assert w[0].requires_grad, "Value system parameters must require gradients for stoic optimization."
            if len(x) > 0:
                assert x[0] is self.optimx.param_groups[0]['params'][0], "Grounding parameters do not match those in the optimizer"
            #assert x[0].requires_grad, "Grounding parameters must require gradients for stoic optimization."
        # PARAMS", self.optimx.param_groups[0]['params'][0].data[0:10])

        if loss_gr_ideal is not None:
            
            self.optimx_ideal.zero_grad(set_to_none=True)
            loss_sum = th.sum(loss_gr_ideal).detach()
            for i in range(len(loss_gr_ideal)):
                    (loss_gr_ideal[i]/loss_sum).backward(retain_graph=True)
                    self.optimx_ideal.step()
                    self.optimx_ideal.zero_grad(set_to_none=True)
        self.training_variables.requires_grad_(False)
        #if self.lr_lambda > 0:
            #assert self.optim_lambdas.param_groups[0]['params'][0:len(self.training_variables.lagrange_multipliers)] is self.training_variables.lagrange_multipliers, "Lagrange multipliers not found in optimizer parameters"
        
        target_gr_loss = loss_gr_ideal.detach() if loss_gr_ideal is not None else None
        selected_indices = None

        if self.training_variables.lagrange_multipliers.dtype != loss_gr.dtype:
            self.training_variables=self.training_variables.to(loss_gr.dtype) 
            
        if self.loss_func_type in MOLossFunctionsCategories.REQUIRES_GRAD_ON_EVERYTHING:
            loss = self.training_variables.forward(loss_gr, loss_vs, target_gr_loss=target_gr_loss)
        elif (self.loss_func_type in MOLossFunctionsCategories.REQUIRES_GRAD_ON_ALL_GROUNDING) and (self.loss_func_type not in MOLossFunctionsCategories.REQUIRES_GRAD_ON_VALUE_SYSTEM_WEIGHTS):
            loss_vs = loss_vs.detach() if loss_vs is not None else None
            loss = self.training_variables.forward(loss_gr, None, target_gr_loss=target_gr_loss)
        elif self.loss_func_type in MOLossFunctionsCategories.REQUIRES_GRAD_ON_VALUE_SYSTEM_WEIGHTS_ALONE:
            loss_gr = loss_gr.detach() if loss_gr is not None else None
            loss_gr.requires_grad_(False)
            self.training_variables.lagrange_multipliers = self.training_variables.lagrange_multipliers.detach()
            loss = loss_vs
        elif self.loss_func_type in MOLossFunctionsCategories.REQUIRES_GRAD_ON_SOME_GROUNDING:
            selected_indices = self.loss_func_kwargs['value_indices']
            loss = self.training_variables.forward(loss_gr, None, target_gr_loss=target_gr_loss, selected_indices=selected_indices)
            unselected_indices = [i for i in range(len(loss_gr)) if i not in selected_indices]
            loss_gr[unselected_indices] = loss_gr[unselected_indices].detach()
            if len(selected_indices) > 1:
                self.training_variables.lagrange_multipliers[unselected_indices] = self.training_variables.lagrange_multipliers[unselected_indices].detach()
            else: 
                self.training_variables.lagrange_multipliers = self.training_variables.lagrange_multipliers.detach()
            #print("YEs", loss_func_kwargs['value_indices'])
            #input("?")
        elif self.loss_func_type in MOLossFunctionsCategories.NEEDS_NO_GRAD_EVER:
            loss_gr = loss_gr.detach() if loss_gr is not None else None
            loss_vs = loss_vs.detach() if loss_vs is not None else None
            loss = self.training_variables.forward(loss_gr, loss_vs, target_gr_loss=target_gr_loss) + th.tensor(0.0, requires_grad=True) # Add dummy loss to allow backward to be called without error, even though no gradients will be computed.
            
        else:
            raise ValueError(f"Unsupported loss function {self.loss_func_type}")
        #print("GRADIENTS BEFORE BACKWARD - VALUE SYSTEM PARAMS:", [p.grad for p in w])
        #print("GRADIENTS BEFORE BACKWARD - GR PARAMS:", [p.grad for p in x][0:5][0:5])
        #loss_gr = loss_gr.detach()
        #loss = loss_vs

        loss.backward(**kwargs)
        
        
        #print("GRADIENTS AFTER BACKWARD - VALUE SYSTEM PARAMS:", [p.grad for p in w])
        #print("GRADIENTS AFTER BACKWARD - GR PARAMS:", [p.grad for p in x][0:5][0:5])
        
        return loss
    def step(self, closure=None)->None:
        #self.set_state({'time': 0, 'lambdas': training_variables.lagrange_multipliers})
        
        
        print("LAGRANGE MULTIPLIERS BEFORE STEP (VS right):", self.training_variables.get_multipliers())
        #print("TRAINING VARS", vars(self.training_variables))
        self.time += 1
        #th.nn.utils.clip_grad_norm_(self.params_gr, self.max_grad_norm)
        #th.nn.utils.clip_grad_norm_(self.params_vs, self.max_grad_norm)
        if __debug__:
            if self.lr_grounding > 0.0: 
                assert self.params_gr[0].grad is not None, "Grounding gradients have not been computed. Make sure to call the backward pass on the grounding loss before stepping the optimizer."
            
            if self.lr_value_system > 0.0:
                assert self.params_vs[0].grad is not None, "Value system gradients have not been computed. Make sure to call the backward pass on the value system loss before stepping the optimizer."
            #print("GRADIENTS BEFORE STEP - VALUE SYSTEM PARAMS:", [p.grad for p in self.params_vs])
        #print("GRADIENTS BEFORE STEP - GR PARAMS:", [p.grad for p in self.params_gr])
            
        if self.optimx is not None and self.lr_grounding > 0.0:
            self.optimx.step()
        if self.optimy is not None and self.lr_value_system > 0.0:
                self.optimy.step()
        

        self.training_variables.prepare_for_optimizer_step()
        if self.lr_lambda > 0:
            self.optim_lambdas.step()
        #print("LAGRANGE MULTIPLIERS AFTER STEP (BEFORE DECAY):", self.training_variables.lagrange_multipliers)
        self.training_variables.post_optimizer_step()

            
        #self.zero_grad()
        #print("GRADIENTS AFTER STEP - VALUE SYSTEM PARAMS:", [p.grad for p in self.params_vs])
        #print("GRADIENTS AFTER STEP - GR PARAMS:", [p.grad for p in self.params_gr])
        
        print("LAGRANGE MULTIPLIERS AFTER STEP (VS right):", self.training_variables.get_multipliers())
        
        #self.set_state({'time': 0, 'lambdas': training_variables.lagrange_multipliers})
        return None
    

class ConstrainedLRScheduler:
    """Composite scheduler that advances all internal schedulers together."""

    def __init__(self, optimizer: ConstrainedOptimizer, sched_x, sched_y=None, sched_lambda=None):
        self.optimizer = optimizer
        self.sched_x = sched_x
        self.sched_y = sched_y
        self.sched_lambda = sched_lambda

    @property
    def _all_schedulers(self):
        return [
            s for s in (self.sched_x, self.sched_y, self.sched_lambda) if s is not None
        ]

    def step(self, metric=None) -> None:
        for scheduler in self._all_schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metric)
            else:
                    scheduler.step()

    def state_dict(self):
        return {
            "sched_x": self.sched_x.state_dict() if self.sched_x is not None else None,
            "sched_y": self.sched_y.state_dict() if self.sched_y is not None else None,
            "sched_lambda": self.sched_lambda.state_dict() if self.sched_lambda is not None else None,
        }

    def load_state_dict(self, state_dict):
        if self.sched_x is not None and state_dict.get("sched_x") is not None:
            self.sched_x.load_state_dict(state_dict["sched_x"])
        if self.sched_y is not None and state_dict.get("sched_y") is not None:
            self.sched_y.load_state_dict(state_dict["sched_y"])
        if self.sched_lambda is not None and state_dict.get("sched_lambda") is not None:
            self.sched_lambda.load_state_dict(state_dict["sched_lambda"])

    def get_last_lr(self):
        lrs = []
        if self.sched_x is not None:
            lrs.extend(self.sched_x.get_last_lr())
        if self.sched_y is not None:
            lrs.extend(self.sched_y.get_last_lr())
        if self.sched_lambda is not None:
            lrs.extend(self.sched_lambda.get_last_lr())
        if len(lrs) == 0:
            return [0.0]
        return lrs

    """@override
    def set_parameters(self, params_gr, params_vs, optim_state={}):
        self.params_gr = params_gr
        self.params_vs = params_vs
        self.optimx = self.sub_optimizer_class(params_gr, lr=self.lr_grounding, **self.optimizer_kwargs)
        self.optimy = self.sub_optimizer_class(params_vs, lr=self.lr_value_system, **self.optimizer_kwargs)"""




