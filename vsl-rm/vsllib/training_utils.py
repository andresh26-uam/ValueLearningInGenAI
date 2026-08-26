

from abc import abstractmethod
from copy import deepcopy
from typing import Tuple
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import numpy as np
from numpy import add
from ordered_set import OrderedSet


from transformers.utils import PaddingStrategy

import torch as th

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vsllib.defines import MOLossFunctionsCategories, MOLossFunctions, MOLossManagement

from vsllib.utils import convert_to_tensors, to_float

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


def normalizing_params(used_mults, vs_coeff, dtype=th.float32) -> Tuple[th.Tensor, th.Tensor]:
        if vs_coeff is not None and used_mults is not None:
            mults_softmax = th.nn.functional.softmax(th.cat([used_mults, vs_coeff], dim=0), dim=0, dtype=dtype)
            n = float(len(mults_softmax))
            mults = th.full_like(mults_softmax, fill_value=1.0/(n**2))
            remaining = 1.0 - 1.0/(n**2) * n
            mults = mults + remaining * mults_softmax
            ret = mults[:-1]*n, mults[-1]*n
        elif used_mults is not None:
            mults_softmax = th.nn.functional.softmax(used_mults, dim=0, dtype=dtype)*(len(used_mults))
            n = float(len(mults_softmax))
            mults = th.full_like(mults_softmax, fill_value=1.0/(n**2))
            remaining = 1.0 - 1.0/(n**2) * n
            mults = mults + remaining * mults_softmax
            ret = mults*n, None
        else:
            ret = None, 1.0
        return ret

def normalizing_params_linear(used_mults, vs_coeff, dtype=th.float32) -> Tuple[th.Tensor, th.Tensor]:
        """Another implementation, unused, as softmax implementation above is more elegant and avoids the multipliers from getting too low."""
        if vs_coeff is not None and used_mults is not None:
            if len(vs_coeff.shape) == 1:
                vs_coeff = vs_coeff.squeeze(0)
            divv = (th.sum(used_mults) + vs_coeff)
            m = (len(used_mults)+1)
            #mults = th.nn.functional.softmax(th.cat([used_mults, vs_coeff], dim=0), dim=0, dtype=dtype)
            ret = used_mults / divv* m, vs_coeff / divv * m
        elif used_mults is not None:
            divv = (th.sum(used_mults))
            m = (len(used_mults))
            ret = used_mults / divv * m, None
        else:
            ret = None, 1.0
        return ret
@th.compile
def norm_penalty( lags, vs_coeff, penalty_coeff) -> th.Tensor:
    """L2 regularization penalty."""
    return penalty_coeff*(th.sum(th.pow(th.cat([lags, vs_coeff],dim=0), 2)))
    


@dataclass
class MORewardDataCollator:
    return_tensors: str = "pt"
    dtype: Optional[th.dtype] = None

    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This arranges features into a batch, so that:
        # batch["key"] = features[:]["key"]
        # Additionally, we manage cases where we have features[i]["key_1"], features[i]["key_2"] to produce:
        # batch["key"][i] = features[i]["key_1"] when i is 0,2,4,6...
        # batch["key"][i+1] = features[i]["key_2"] 
        # 
        merged_features = []

        for feature in features:
            pair_labels = feature["labels"]
            c = feature.pop("context_features", [])
            dic1 = {"labels": pair_labels[0],
                }
            dic2 = {"labels": pair_labels[1],
                }
            dic1["grounding_features"] = feature.pop("grounding_features_1", [])
            dic1["context_features"] = c
            dic2["grounding_features"] = feature.pop("grounding_features_2", [])
            dic2["context_features"] = c
            merged_features.append(
                dic1
            )
                
            merged_features.append(
                dic2
            )
        batch = {
            key: th.stack([th.tensor(feature[key],dtype=self.dtype) for feature in merged_features])
            for key in merged_features[0]
        }
        if __debug__:
            print("INPUT", batch["context"][0])
        
        
        batch["return_loss"] = True
        batch["grounding_features"] = batch["grounding_features"].to(dtype=self.dtype) 
        batch["context_features"] = batch["context_features"].to(dtype=self.dtype) 
        assert len(batch["grounding_features"]) == len(features)*2
        return batch

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
        # This arranges features into a batch, so that:
        # batch["key"] = features[:]["key"]
        # Additionally, we manage cases where we have features[i]["key_1"], features[i]["key_2"] to produce:
        # batch["key"][i] = features[i]["key_1"] when i is 0,2,4,6...
        # batch["key"][i+1] = features[i]["key_2"] 
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
    """This class is responsible for managing the Lagrange multipliers and value system coefficient during training, 
    as well as accumulating the tendency on grounding losses, value system losses, coherences and representativeness metrics across training steps, 
    used as signals for updating the multipliers and value system coefficient."""

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
        multipliers, vs_coeff = self.get_multipliers(used_only=False)

        for i in range(len(multipliers)):
            if multipliers[i] is not None:
                result[f"lagrange_multiplier_{i}"] = to_float(multipliers[i])
        if vs_coeff is not None:
            result["vs_coeff"] = to_float(vs_coeff)
        if self.maximum_coherences_tendency is not None:
            result["maximum_coherences_tendency"] = to_float(self.maximum_coherences_tendency)
        if self.minimum_grounding_loss_tendency is not None:
            result["minimum_grounding_loss_tendency"] = to_float(self.minimum_grounding_loss_tendency)
        
        if self.coherences_tendency is not None:
            result["avg_coherences_tendency"] = to_float(self.coherences_tendency)
        if self.grounding_loss_tendency is not None:
            result["avg_grounding_loss_tendency"] = to_float(self.grounding_loss_tendency)
        return result
    
    def forward(self, grounding_losses: th.Tensor, vs_losses: th.Tensor, target_gr_loss: th.Tensor = None, selected_indices: list = None, add_vs_loss: bool = True, add_gr_loss: bool = True) -> th.Tensor:
        """
        Computes the Lagrangian with certain configuration options.
        """
        used_mults, vs_coeff = self.normalize_coefficients(selected_indices=selected_indices, add_vs_loss=add_vs_loss, add_gr_loss=add_gr_loss)
        
        if __debug__ and add_gr_loss:
            if selected_indices is not None:
                assert used_mults.shape == (len(selected_indices),), f"Expected used_mults shape to match grounding_losses shape, but got {used_mults.shape} and {grounding_losses.shape}"
            else:
                assert used_mults.shape == grounding_losses.shape, f"Expected used_mults shape to match grounding_losses shape, but got {used_mults.shape} and {grounding_losses.shape}"
        used_grounding_losses = grounding_losses[selected_indices] if selected_indices is not None else grounding_losses
        
        if add_gr_loss:
            
            if target_gr_loss is None or self.zero_constraint:
                lag_gr_loss = th.dot(used_mults, used_grounding_losses)

            else:
                lag_gr_loss = th.dot(used_mults, th.maximum(used_grounding_losses - target_gr_loss[selected_indices], th.zeros_like(used_grounding_losses, requires_grad=False)))
        else:
            lag_gr_loss = th.tensor(0.0, device=grounding_losses.device, dtype=grounding_losses.dtype, requires_grad=False)
            
        #last_loss_original_unscaled = lag_gr_loss + vs_loss
        if __debug__:
            if add_vs_loss:
                assert vs_coeff is not None
                assert vs_losses is not None
        vs_loss_scaled =  (vs_losses if ((vs_losses is not None) and add_vs_loss) else 0.0)*(vs_coeff if vs_coeff is not None else 0.0)
        total_loss = lag_gr_loss + vs_loss_scaled
        self._last_selected_indices = selected_indices
        self._last_add_gr_loss = add_gr_loss
        self._last_add_vs_loss = add_vs_loss  
        """if __debug__:
            print("TOTAL LOSS: ", total_loss , "which is the sum of Lagrange grounding loss and value system loss:" )
            print(f" ADDED: {add_gr_loss} Lagrange grounding loss (lag_gr_loss): ", lag_gr_loss, "dot of" , used_grounding_losses)
            print(f"  ADDED: {add_vs_loss} Value system loss (vs_losses * vs_coeff): ", vs_loss_scaled, "which is vs_losses:", vs_losses, "times vs_coeff:", vs_coeff)
        """
        return total_loss
    
    def _apply(self, fn, recurse=True) -> Any:
        """This is a custom implementation of the _apply method to ensure that when the training variables are moved to a different device or dtype,
        the cached metrics and losses are also moved accordingly, as they are stored as lists of tensors."""
        def move_list(lst: list):
            if lst is None:
                return None
            return [fn(x) if isinstance(x, th.Tensor) else x for x in lst]

        for field_name in self._iter_cached_field_names():
            field_value = getattr(self, field_name, None)
            if isinstance(field_value, list) or field_value is None:
                setattr(self, field_name, move_list(field_value))

        return super()._apply(fn, recurse=recurse)

    
    
    
    def normalize_coefficients(self, selected_indices: list = None, add_vs_loss: bool = True, add_gr_loss: bool = True) -> th.Tensor:
        """Normalizes the coefficients for the Lagrangian multipliers and value system coefficient."""
        if add_gr_loss:
            used_mults = self.lagrange_multipliers[selected_indices] if selected_indices is not None else self.lagrange_multipliers
        else:
            used_mults = None
        if not add_vs_loss:
            vs_coeff = None
        else:
            vs_coeff = self.vs_coeff

        m1, m2 = normalizing_params(used_mults, vs_coeff, dtype=self.lagrange_multipliers.dtype) # This is to ensure the multipliers are 1 on average (to make all model training similar scale regardless of how many active multipliers/losses are used)
        
        return m1, m2

    def get_multipliers(self, used_only=False) -> Tuple[th.Tensor, th.Tensor]:
        return self.normalize_coefficients(selected_indices=self._last_selected_indices if used_only else None)
        #union_mults_s = th.nn.functional.softmax(union_mults, dim=0)
            
    def __init__(self, n_values: int, initial_lambda: int =1.0 , 
                 device: th.DeviceObjType|str ='cpu', dtype: th.Type = th.float32, 
                 grounding_loss_tendency_update_ratio: float = 0.01, 
                 gradient_accumulation_steps=10, 
                 update_tendencies_every_n_steps=1,
                 use_validation_for_tendencies=False,
                 use_metrics_or_losses='metrics', 
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
        self.grounding_loss_tendency: th.Tensor | None = None
        self.coherences_tendency: th.Tensor | None = None

        self.minimum_vs_loss_tendency: th.Tensor | None = None
        self.maximum_representativeness_tendency: th.Tensor | None = None
        self.vs_loss_tendency: th.Tensor | None = None
        self.representativeness_tendency: th.Tensor | None = None
        

        self.last_accumulated_grounding_loss: th.Tensor | None = None
        self.last_accumulated_vs_loss: th.Tensor | None = None

        self.last_accumulated_grounding_loss_ideal: th.Tensor | None = None
        self.last_accumulated_coherences_ideal: th.Tensor | None = None
        self.last_accumulated_coherences: th.Tensor | None = None
        self.last_accumulated_representativeness: th.Tensor | None = None

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

        self.use_validation_for_tendencies=use_validation_for_tendencies
        self.update_tendencies_every_n_steps=update_tendencies_every_n_steps

        self.training_steps_so_far = 0
        self.is_initialized = False
        self.historic_eval_metrics = {}
            
    
    def prepare_for_optimizer_step(self, need_backward: bool = True) -> None:
        coeff: th.Tensor
        self.zero_grad(set_to_none=True)
        

        self.update_losses()
        self.update_metrics()
        
        if self.training_steps_so_far % self.update_tendencies_every_n_steps == 0 and self.training_steps_so_far > 0:
            self.update_loss_tendencies()
            self.update_metrics_tendencies()
        if self.is_initialized:
            with th.no_grad():
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
                    # Use coherence difference with the tendency as improvement signal.
                    gr_ideal_diff_orig = -(self.last_accumulated_coherences - coherences_tendency).detach()
                    vs_ideal_diff = -(self.last_accumulated_representativeness - representativeness_tendency).detach() if self.last_accumulated_representativeness is not None else None
                    
                    
                elif self.use_metrics_or_losses == 'losses':
                    # Use loss difference with the tendency as improvement signal.
                    gr_ideal_diff_orig = (self.last_accumulated_grounding_loss - grounding_loss_tendency).detach()
                    vs_ideal_diff = (self.last_accumulated_vs_loss - vs_loss_tendency).detach() if self.last_accumulated_vs_loss is not None else None
                
                else:
                    raise ValueError(f"Invalid value for use_metrics_or_losses: {self.use_metrics_or_losses}. Expected 'metrics' or 'losses'.")    
                    #should be...? 1) forward = self.forward(grounding_losses=gr_ideal_diff, vs_losses=self.last_accumulated_vs_loss)
                    #should be...? 2) coeff = ((gr_ideal_diff*(lag_sum)) - 1*(forward))/th.pow(lag_sum, 2) 
            
            self.requires_grad_(need_backward)
            
            gr_ideal_diff = th.clamp(gr_ideal_diff_orig, min=0.0).detach() # We only want to increase the multipliers for the grounding losses that are above their ideal losses (or below their ideal metrics, depending on the mode), so we set the ideal differences to zero for the ones that are already in a good place.
            #gr_ideal_diff = gr_ideal_diff_orig.detach()
            
            
            add_vs_loss = self._last_add_vs_loss         
            if need_backward:
                before_vs_loss = self._last_add_vs_loss
                if self.lambda_decay > 0.0:
                    if self._last_selected_indices is not None and self._last_add_gr_loss:
                        norm_penalty_ =  norm_penalty(self.lagrange_multipliers[self._last_selected_indices], self.vs_coeff if self._last_add_vs_loss else th.zeros_like(self.vs_coeff,requires_grad=False), self.lambda_decay)
                    elif self._last_add_gr_loss:
                        norm_penalty_ =  norm_penalty(self.lagrange_multipliers, self.vs_coeff if self._last_add_vs_loss else th.zeros_like(self.vs_coeff,requires_grad=False), self.lambda_decay)
                        """elif self._last_add_vs_loss:
                            norm_penalty_ =  th.zeros(1, device=self.lagrange_multipliers.device, dtype=self.lagrange_multipliers.dtype, requires_grad=False) #th.norm(self.vs_coeff)#NOT USED.
                        """
                    else:
                        norm_penalty_ = th.zeros(1, device=self.lagrange_multipliers.device, dtype=self.lagrange_multipliers.dtype, requires_grad=False)
                else:
                    norm_penalty_ = th.zeros(1, device=self.lagrange_multipliers.device, dtype=self.lagrange_multipliers.dtype, requires_grad=False)
                if vs_ideal_diff is not None:
                    with th.no_grad():
                        vs_ideal_diff = 0.0
                        if th.any(gr_ideal_diff_orig > 0.0):
                            #add_vs_loss = False
                            vs_ideal_diff = 0.0
                        #else:
                            #vs_ideal_diff = th.clamp(vs_ideal_diff, min=0.0).detach()
                
                forward = -self.forward(grounding_losses=gr_ideal_diff, vs_losses=vs_ideal_diff, selected_indices=self._last_selected_indices, add_vs_loss=add_vs_loss, add_gr_loss=self._last_add_gr_loss) + norm_penalty_ # It is negated, as it is a maximization problem
                self._last_add_vs_loss = before_vs_loss
                forward.backward()
                
                if self.grad_on_only_worst_value:
                    coeff = self.lagrange_multipliers.grad.detach()
                    worst_idx = th.argmax(gr_ideal_diff).detach(), 
                    worst_val = coeff[worst_idx].detach().item()
                    coeff.zero_() # Set all gradients to zero
                    coeff[worst_idx] = worst_val # Except the worst one, which we keep as is.
                    self.lagrange_multipliers.grad = coeff
            
            coeff = self.lagrange_multipliers.grad
            if __debug__:
                print("GRAD DIFF", gr_ideal_diff, "ORIG", gr_ideal_diff_orig)
                
                print("LAG GRAD", coeff, "VS_COEFF GRAD", self.vs_coeff.grad if self._last_add_vs_loss else None)
                
                print("LAGRANGE MULTIPLIERS", self.lagrange_multipliers)
                print("VS COEFF", self.vs_coeff)
                #print("MULTS", self.get_multipliers(used_only=False))
                
                print("---------")
                #print("CHR TENDENCY", coherences_tendency)
                #print("REPR TENDENCY", representativeness_tendency)
                #print("GROUNDING LOSSES TENDENCY", grounding_loss_tendency)
                #print("VS LOSSES TENDENCY", vs_loss_tendency)
                #print("---------")
            
            """with th.no_grad():
                if add_vs_loss and need_backward:
                    assert self.vs_coeff.grad is not None, "VS Coefficient gradient is None before optimizer step, but it should not be when add_vs_loss is True."
                else:
                    if add_vs_loss:
                        assert self.vs_coeff.grad is None or th.allclose(self.vs_coeff.grad, th.zeros_like(self.vs_coeff.grad)), "VS Coefficient gradient is not zero before optimizer step, but it should be when add_vs_loss is False."
                if self._last_add_gr_loss  and need_backward:
                    if self._last_selected_indices is not None:
                        assert not th.allclose(coeff[self._last_selected_indices], th.zeros_like(coeff[self._last_selected_indices])), "Selected grounding multipliers have zero gradients, but they should not be zero."
                    else:
                        if th.any(gr_ideal_diff != 0.0):
                            assert not th.allclose(coeff, th.zeros_like(coeff)), "Grounding multipliers have zero gradients, but they should not be zero."
                else:
                    assert coeff is None or th.allclose(coeff, 0.0), "Grounding multipliers have non-zero gradients, but they should be zero when add_gr_loss is False."
        """
            
    def zero_grad(self, set_to_none: bool = True) -> None:
        set_to_none = True # TODO: Apparetly this is much faster. See https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        self.lagrange_multipliers.grad = None
        self.vs_coeff.grad = None
        #return super().zero_grad(set_to_none)
    def requires_grad_(self, requires_grad: bool = True):
        self.lagrange_multipliers.requires_grad_(requires_grad)
        self.vs_coeff.requires_grad_(requires_grad)
    def post_optimizer_step(self) -> None:
        
        self.zero_grad(set_to_none=True)
        
        
        self.training_steps_so_far +=1
        #self.lagrange_multipliers.data.clamp_(min=0.1)
        #self.vs_coeff.data.clamp_(min=0.1)
        
        

    def reset_state(self) -> None:
        self.training_steps_so_far = 0
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


    def update_metrics(self) -> None:
        with th.no_grad():
            self.last_accumulated_coherences = th.stack(self._cached_coherences).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
            
            self.last_accumulated_representativeness = th.stack(self._cached_representativeness).mean().detach()

            if len(self._cached_coherences_ideal) > 0:
                self.last_accumulated_coherences_ideal = th.stack(self._cached_coherences_ideal).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
            
    def update_losses(self) -> None:
        with th.no_grad():
            self.last_accumulated_grounding_loss = th.stack(self._cached_groundings).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
            if len(self._cached_groundings_ideal) > 0:
                self.last_accumulated_grounding_loss_ideal = th.stack(self._cached_groundings_ideal).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
            else:
                self.last_accumulated_grounding_loss_ideal = self.last_accumulated_grounding_loss.detach()
            self.last_accumulated_vs_loss = th.stack(self._cached_vs_losses).mean().detach()

    def update_metrics_tendencies(self) -> None:
        cached_coherences = None
        if self.use_validation_for_tendencies and len(self.historic_eval_metrics)>0:
            hgr = self.historic_eval_metrics.get("coherences", [])
            hvs = self.historic_eval_metrics.get("representativeness", [])
            hgrideal = self.historic_eval_metrics.get("coherences_ideal", [])
            
            if len(hgr) > 0:
                assert len(hvs) > 0, "Expected 'value_system_loss' key in historic_eval_metrics when grounding is used"
                cached_coherences = [th.as_tensor(h).requires_grad_(False) for h in hgr[-min(1, len(hgr)):]]
            if len(hgrideal) > 0:
                cached_coherences_ideal = [th.as_tensor(h).requires_grad_(False) for h in hgrideal[-min(1, len(hgrideal)):]]
            else:
                cached_coherences_ideal = []
            if len(hvs) > 0:
                cached_representativeness = [th.as_tensor(h).requires_grad_(False) for h in hvs[-min(1, len(hvs)):]]

            #cached_coherences = th.as_tensor(self.historic_eval_metrics.get("coherences", [[0.0]*len(self.lagrange_multipliers)]*self.gradient_accumulation_steps)[-self.gradient_accumulation_steps:]).requires_grad_(False)
            #cached_coherences_ideal = th.as_tensor(self.historic_eval_metrics.get("coherences_ideal", [[0.0]*len(self.lagrange_multipliers)]*self.gradient_accumulation_steps)[-self.gradient_accumulation_steps:]).requires_grad_(False)
            #cached_representativeness = th.as_tensor(self.historic_eval_metrics.get("representativeness", [0.0]*self.gradient_accumulation_steps)[-self.gradient_accumulation_steps:]).requires_grad_(False)
        elif not self.use_validation_for_tendencies:
            cached_coherences = self._cached_coherences
            cached_coherences_ideal = self._cached_coherences_ideal
            cached_representativeness = self._cached_representativeness
        if cached_coherences is None:
            return None
        with th.no_grad():
            
            if self.use_validation_for_tendencies:
                tendency_coherences = th.stack(cached_coherences).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
                
                tendency_representativeness = th.stack(cached_representativeness).mean().detach()

                if len(cached_coherences_ideal) > 0:
                    tendency_coherences_ideal = th.stack(cached_coherences_ideal).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
                else:
                    tendency_coherences_ideal = tendency_coherences
            else:
                tendency_coherences = self.last_accumulated_coherences
                tendency_representativeness = self.last_accumulated_representativeness
                tendency_coherences_ideal = self.last_accumulated_coherences_ideal if self.last_accumulated_coherences_ideal is not None else self.last_accumulated_coherences

            coherence_target = th.maximum(tendency_coherences, tendency_coherences_ideal).detach()
            if self.maximum_coherences_tendency is None:
                self.maximum_coherences_tendency = th.full_like(coherence_target, fill_value=0.5).detach()
                self.maximum_representativeness_tendency = th.full_like(tendency_representativeness, fill_value=0.5).detach()
                self.coherences_tendency = th.full_like(coherence_target, fill_value=0.5).detach()
                self.representativeness_tendency = th.full_like(tendency_representativeness, fill_value=0.5).detach()
            else:
                maximum = th.maximum(coherence_target, self.maximum_coherences_tendency)
                
                self.maximum_coherences_tendency = (th.multiply(maximum, self.loss_metric_tendency_update_ratio) + th.multiply(self.maximum_coherences_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.coherences_tendency = (th.multiply(coherence_target, self.loss_metric_tendency_update_ratio) + th.multiply(self.coherences_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                
                maximum_repr = th.maximum(tendency_representativeness, self.maximum_representativeness_tendency)
                self.maximum_representativeness_tendency = (th.multiply(maximum_repr, self.loss_metric_tendency_update_ratio) + th.multiply(self.maximum_representativeness_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.representativeness_tendency = (th.multiply(tendency_representativeness, self.loss_metric_tendency_update_ratio) + th.multiply(self.representativeness_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()

            self.is_initialized = True
    def update_loss_tendencies(self) -> Optional[th.Tensor]:
        cached_groundings = None
        if self.use_validation_for_tendencies and len(self.historic_eval_metrics)>0:
            hgr = self.historic_eval_metrics.get("grounding_loss", [])
            hvs = self.historic_eval_metrics.get("value_system_loss", [])
            hgrideal = self.historic_eval_metrics.get("grounding_loss_ideal", [])
            
            if len(hgr) > 0:
                assert len(hvs) > 0, "Expected 'value_system_loss' key in historic_eval_metrics when grounding is used"
                cached_groundings = [th.as_tensor(h).requires_grad_(False) for h in hgr[-min(1, len(hgr)):]]
            if len(hgrideal) > 0:
                cached_groundings_ideal = [th.as_tensor(h).requires_grad_(False) for h in hgrideal[-min(1, len(hgrideal)):]]
            else:
                cached_groundings_ideal = []
            if len(hvs) > 0:
                cached_vs_losses = [th.as_tensor(h).requires_grad_(False) for h in hvs[-min(1, len(hvs)):]]
        elif not self.use_validation_for_tendencies:
            cached_groundings = self._cached_groundings
            cached_groundings_ideal = self._cached_groundings_ideal
            cached_vs_losses = self._cached_vs_losses
        if cached_groundings is None:
            return None
        with th.no_grad():
            if self.use_validation_for_tendencies:
                tendency_gr_loss = th.stack(cached_groundings).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
                if len(cached_groundings_ideal) > 0:
                    tendency_gr_loss_ideal = th.stack(cached_groundings_ideal).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach()
                else:
                    tendency_gr_loss_ideal = tendency_gr_loss
                tendency_vs_loss = th.stack(cached_vs_losses).mean().detach()
            else:
                tendency_gr_loss = self.last_accumulated_grounding_loss.clone()
                tendency_gr_loss_ideal = self.last_accumulated_grounding_loss_ideal.clone() if self.last_accumulated_grounding_loss_ideal is not None else self.last_accumulated_grounding_loss.clone()
                tendency_vs_loss = self.last_accumulated_vs_loss.clone()

            grounding_loss = th.minimum(tendency_gr_loss, tendency_gr_loss_ideal).detach()
            
            if self.minimum_grounding_loss_tendency is None:
                self.minimum_grounding_loss_tendency = th.full_like(grounding_loss, fill_value=th.max(grounding_loss).float()).detach()
                self.minimum_vs_loss_tendency = th.full_like(tendency_vs_loss, fill_value=th.max(tendency_vs_loss).float()).detach()
                self.grounding_loss_tendency = th.full_like(grounding_loss, fill_value=th.max(grounding_loss).float()).detach()
                self.vs_loss_tendency = th.full_like(tendency_vs_loss, fill_value=th.max(tendency_vs_loss).float()).detach()
            else:
                minimum = th.minimum(grounding_loss, self.minimum_grounding_loss_tendency)
                minimum_vs = th.minimum(tendency_vs_loss, self.minimum_vs_loss_tendency)

                self.minimum_grounding_loss_tendency = (th.multiply(minimum, self.loss_metric_tendency_update_ratio) + th.multiply(self.minimum_grounding_loss_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.minimum_vs_loss_tendency = (th.multiply(minimum_vs, self.loss_metric_tendency_update_ratio) + th.multiply(self.minimum_vs_loss_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.grounding_loss_tendency = (th.multiply(grounding_loss, self.loss_metric_tendency_update_ratio) + th.multiply(self.grounding_loss_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
                self.vs_loss_tendency = (th.multiply(tendency_vs_loss, self.loss_metric_tendency_update_ratio) + th.multiply(self.vs_loss_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach()
            self.is_initialized = True
    def record_metrics(self, metrics: Dict[str, float|list], metric_type: Literal["train", "validation"] = "train") -> None:
        # This is called at the end of each evaluation phase, and records the grounding loss and vs loss for the last evaluation phase, which will be used to update the Lagrange multipliers before the next optimizer step.
        
        if metric_type == "validation":
            for key in metrics.keys():
                if key not in self.historic_eval_metrics:
                    self.historic_eval_metrics[key] = []
                m = metrics[key]
                if isinstance(m, np.ndarray):
                    m = m.tolist()
                self.historic_eval_metrics[key].append(m)
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

    def record_grounding_loss(self, gr_loss_detached: th.Tensor, vs_loss_detached: th.Tensor, gr_loss_ideal_detached=None, loss_type: Literal["train", "validation"] = "train") -> None:
        if loss_type == "validation":
            for key in ["grounding_loss", "value_system_loss", "grounding_loss_ideal"]:
                if key not in self.historic_eval_metrics:
                    self.historic_eval_metrics[key] = []
                if key == "grounding_loss_ideal" and gr_loss_ideal_detached is None:
                    continue
                else:
                    self.historic_eval_metrics[key].append(gr_loss_detached if key == "grounding_loss" else vs_loss_detached)
        else:
            with th.no_grad():
                # This estimates the minimum obtainable loss for each value. The optimizer will take this into account
                for field in self._iter_cached_field_names():
                    field_value = getattr(self, field, None)
                    if len(field_value) > self.gradient_accumulation_steps:
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
    def __init__(self, params_gr: th.ParameterDict, params_vs: th.ParameterDict, params_ctx: th.ParameterDict,  n_values: int, lr_grounding=None, lr_value_system=None, lr_context=None, sub_optimizer_class=th.optim.Adam,  **optimizer_kwargs):
        
        self.lr_grounding = lr_grounding
        self.lr_value_system = lr_value_system
        self.lr_context = lr_context
        defaults = dict(lr_grounding=lr_grounding,
                        lr_value_system=lr_value_system,lr_context=lr_context)

        self.optimizer_kwargs = optimizer_kwargs
        self.n_values = n_values

        params_gr = OrderedSet(params_gr)
        params_vs = OrderedSet(params_vs)
        params_ctx = OrderedSet(params_ctx)

        self.params_gr = params_gr
        self.params_vs = params_vs
        self.params_ctx = params_ctx

        print("SUBOPTIMIZER CLASS:", sub_optimizer_class)
        self.sub_optimizer_class = sub_optimizer_class

        if params_gr and len(params_gr) > 0:
            self.optimx = _create_sub_optimizer(params_gr, lr_grounding, self.sub_optimizer_class, self.optimizer_kwargs)
        else:
            self.optimx = None
        if params_vs and len(params_vs) > 0:
            self.optimy = _create_sub_optimizer(params_vs, lr_value_system, self.sub_optimizer_class, self.optimizer_kwargs)
        else:
            self.optimy = None
        
        if params_ctx is not None and len(params_ctx) > 0:
            self.optimz = _create_sub_optimizer(params_ctx, lr_context, self.sub_optimizer_class, self.optimizer_kwargs)
        else:
            self.optimz = None
            
        all_params = [*params_gr, *params_vs, *params_ctx]
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
        if self.optimz is not None:
            self.optimz.zero_grad(set_to_none)
        return None

    @abstractmethod
    def step(self, closure=None)-> None:
        if self.optimx is not None:
            self.optimx.step()
        if self.optimy is not None:
            self.optimy.step()
        if self.optimz is not None:
            self.optimz.step()
        return None

class ConstrainedOptimizer(VSLOptimizer):

    @property
    def loss_management(self) -> MOLossManagement:
        return MOLossManagement(self.loss_func_type, self.loss_func_kwargs)
    
    def __init__(self, params, params_gr, params_vs, params_ctx, n_values, params_gr_ideal=None, lr_grounding=None,
                 lr_value_system=None, lr_lambda=None, lr_context=None,
                 loss_func_type: MOLossFunctions=MOLossFunctions.DEFAULT, loss_func_type_kwargs: dict = {},
                 training_variables: MORMTrainingVariables = None,
                 sub_optimizer_class=th.optim.Adam, **optimizer_kwargs):
        # Params must be provided for compatibility with transformers library.
        super(ConstrainedOptimizer, self).__init__(params_gr=params_gr, params_vs=params_vs, params_ctx=params_ctx, n_values=n_values,
                                                   lr_grounding=lr_grounding, lr_value_system=lr_value_system, lr_context=lr_context, sub_optimizer_class=sub_optimizer_class, **optimizer_kwargs)
        self.loss_func_type = MOLossFunctions(loss_func_type)
        self.loss_func_kwargs=loss_func_type_kwargs
        
        if params_gr_ideal is not None:
            assert len(params_gr) == len(params_gr_ideal), "Grounding parameters and ideal grounding parameters must have the same length."
            
            self.params_gr_ideal = params_gr_ideal
            self.optimx_ideal = _create_sub_optimizer(params_gr_ideal, lr_grounding, self.sub_optimizer_class, optimizer_kwargs)

        self.lr_lambda = lr_lambda if lr_lambda is not None else lr_value_system
        if self.loss_management.should_apply_grad_on_lagrange_multipliers() and self.lr_lambda == 0.0:
            raise ValueError(f"Loss function type {loss_func_type} requires applying gradients on Lagrange multipliers, but lr_lambda is set to 0.0. Please set lr_lambda to a positive value to enable optimization of Lagrange multipliers.")
        

        self.training_variables: MORMTrainingVariables = training_variables
        
        if self.lr_lambda > 0:
            self.optim_lambdas = th.optim.SGD(
                self.training_variables.parameters(), lr=self.lr_lambda, weight_decay=0.0) # lambda_decay is managed manually.
        
        self.training_variables.reset_state()
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
    
    def custom_backward(self, loss_gr, loss_gr_ideal, loss_vs, epoch: int, **kwargs) -> th.Tensor:
        
        if __debug__:
            x = self.params_gr 
            w = self.params_vs
            z = self.params_ctx
            if len(w) > 0:
                assert w[0] is self.optimy.param_groups[0]['params'][0], "Value system parameters do not match those in the optimizer"
                assert w[0].requires_grad, "Value system parameters must require gradients for stoic optimization."
            if len(z) > 0:
                assert z[0] is self.optimz.param_groups[0]['params'][0], "Value system parameters do not match those in the optimizer"
                assert z[0].requires_grad, "Value system parameters must require gradients for stoic optimization."
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

        if self.training_variables.lagrange_multipliers.dtype != loss_gr.dtype:
            self.training_variables=self.training_variables.to(dtype=loss_gr.dtype) 
        

        add_vs_loss = True
        add_gr_loss = True
        selected_indices = None
        
        if self.loss_management.requires_grad_on_everything():
            loss = self.training_variables.forward(loss_gr, loss_vs, target_gr_loss=target_gr_loss, selected_indices=None, add_vs_loss=True, add_gr_loss=True)
        elif self.loss_management.needs_no_grad_ever():
                with th.no_grad():
                    loss = self.training_variables.forward(loss_gr, loss_vs, target_gr_loss=target_gr_loss, selected_indices=None, add_vs_loss=True, add_gr_loss=True)
                loss += 0.5*th.tensor(1.0, requires_grad=True, device=loss_vs.device, dtype=loss_vs.dtype) # Just to have a loss to call backward on, since the real loss is not used for optimization in this mode.
        else:
            if not self.loss_management.requires_grad_for_value_system_loss(epoch=epoch):
                loss_vs = loss_vs.detach() if loss_vs is not None else None
                loss_vs.requires_grad_(False)
                add_vs_loss = False

            if not self.loss_management.requires_grad_for_some_or_all_grounding_losses(epoch=epoch):#(self.loss_func_type not in MOLossFunctionsCategories.REQUIRES_GRAD_FOR_SOME_OR_ALL_GROUNDING_LOSSES):
                loss_gr = loss_gr.detach() if loss_gr is not None else None
                loss_gr.requires_grad_(False)
                add_gr_loss = False

            if not self.loss_management.should_apply_grad_on_lagrange_multipliers(epoch=epoch):
                self.training_variables.requires_grad_(False)
            
            if self.loss_func_type in MOLossFunctionsCategories.EPOCH_DEPENDENT_GRAD_REQUIREMENTS:
                grgrad = self.loss_management.should_apply_grad_on_grounding_parameters(epoch=epoch)
                
                for p in self.params_gr:
                    p.requires_grad_(grgrad)
                vsgrad = self.loss_management.should_apply_grad_on_value_system_weights(epoch=epoch)
                
                for p in self.params_vs:
                    p.requires_grad_(vsgrad)

                ctxgrad = self.loss_management.should_apply_grad_on_context_parameters(epoch=epoch)
                
                for p in self.params_ctx:
                    p.requires_grad_(ctxgrad)

            if self.loss_management.requires_grad_for_only_some_grounding_losses(epoch=epoch):
                selected_indices = self.loss_func_kwargs['value_indices']
                unselected_indices = [i for i in range(len(loss_gr)) if i not in selected_indices]
                loss_gr[unselected_indices] = loss_gr[unselected_indices].detach()
                if len(selected_indices) > 1:
                    self.training_variables.lagrange_multipliers[unselected_indices].requires_grad_(False)
                else: 
                    self.training_variables.lagrange_multipliers.requires_grad_(False)

            loss = self.training_variables.forward(loss_gr, loss_vs, target_gr_loss=target_gr_loss, selected_indices=selected_indices, add_vs_loss=add_vs_loss, add_gr_loss=add_gr_loss)   
        loss.backward(**kwargs)
        return loss
    def step(self, closure=None)->None:
        
        if __debug__:
            if self.lr_grounding > 0.0: 
                assert self.params_gr[0].grad is not None, "Grounding gradients have not been computed. Make sure to call the backward pass on the grounding loss before stepping the optimizer."
            
            if self.lr_value_system > 0.0:
                assert self.params_vs[0].grad is not None, "Value system gradients have not been computed. Make sure to call the backward pass on the value system loss before stepping the optimizer."
            if self.lr_context > 0.0:
                if len(self.params_ctx) > 0:
                    assert self.params_ctx[0].grad is not None, "Value system gradients have not been computed. Make sure to call the backward pass on the value system loss before stepping the optimizer."
            
        if self.optimx is not None and self.lr_grounding > 0.0:
            self.optimx.step()
        if self.optimy is not None and self.lr_value_system > 0.0:
                self.optimy.step()
        if self.optimz is not None and self.lr_context > 0.0:
                self.optimz.step()
        
        
        self.training_variables.prepare_for_optimizer_step(need_backward=self.lr_lambda > 0)
        if self.lr_lambda > 0:
                self.optim_lambdas.step()
        self.training_variables.post_optimizer_step()
        
class ConstrainedLRScheduler(th.optim.lr_scheduler.LRScheduler):
    """Composite scheduler that advances all internal schedulers together."""

    def __init__(self, optimizer: ConstrainedOptimizer, sched_x, sched_y=None, sched_z=None, sched_lambda=None):
        self.optimizer = optimizer
        self.sched_x = sched_x
        self.sched_y = sched_y
        self.sched_z = sched_z
        self.sched_lambda = sched_lambda

    def step(self, metric=None) -> None:
        for scheduler in (self.sched_x, self.sched_y, self.sched_z, self.sched_lambda):
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(metric)
                else:
                        scheduler.step()

    

    def get_last_lr(self) -> list[float]:
        lrs = []
        if self.sched_x is not None:
            lrs.extend(self.sched_x.get_last_lr())
        if self.sched_y is not None:
            lrs.extend(self.sched_y.get_last_lr())
        if self.sched_z is not None:
            lrs.extend(self.sched_z.get_last_lr())
        if self.sched_lambda is not None:
            lrs.extend(self.sched_lambda.get_last_lr())
        if len(lrs) == 0:
            return [0.0]
        return lrs



