

from copy import deepcopy
from dataclasses import dataclass
from heapq import merge
import json
import os
from typing import Any, Dict, List, Literal, Optional, Union

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DefaultDataCollator, PreTrainedModel, Trainer


from transformers.utils import PaddingStrategy

import torch as th
import numpy as np

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def print_tensor_and_grad_fn(grad_fn, level=0):
    indent = "  " * level
    if grad_fn is None:
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

def fuse_parameters(model):
    """Move model parameters to a contiguous tensor, and return that tensor."""
    n = sum(p.numel() for p in model.parameters())
    params = th.zeros(n, requires_grad=True, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    params.grad = th.zeros(n, device=params.device, dtype=params.dtype)
    i = 0
    for p in model.parameters():
        params_slice = params[i:i + p.numel()]
        with th.no_grad(): params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        p.grad = params.grad[i:i + p.numel()].view(p.shape)
        i += p.numel()
    return params

def infer_base_hidden_size(self, base_model: AutoModelForSequenceClassification|PreTrainedModel) -> int:
        # Prefer the task head width if present.
        if hasattr(base_model, "score") and hasattr(base_model.score, "in_features"):
            return int(base_model.score.in_features)

        # Common transformer config names used across model families.
        cfg = getattr(base_model, "config", None)
        for attr in ("hidden_size", "d_model", "n_embd", "dim"):
            value = getattr(cfg, attr, None)
            if value is not None:
                return int(value)

        # Final fallback: infer from token embedding width.
        input_emb = None
        if hasattr(base_model, "get_input_embeddings"):
            input_emb = base_model.get_input_embeddings()
        if input_emb is not None and hasattr(input_emb, "embedding_dim"):
            return int(input_emb.embedding_dim)
        if input_emb is not None and hasattr(input_emb, "weight"):
            return int(input_emb.weight.shape[-1])

        raise ValueError(
            "Could not infer base model hidden size. Expected one of: score.in_features, "
            "config.hidden_size/d_model/n_embd/dim, or get_input_embeddings().embedding_dim."
        )

def save_checkpoint_with_seed(trainer: Trainer, tokenizer: AutoTokenizer, checkpoint_dir: str, seed: int):
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    seed_info = {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_initial_seed": int(th.initial_seed()),
    }
    with open(os.path.join(checkpoint_dir, "seed_info.json"), "w", encoding="utf-8") as fp:
        json.dump(seed_info, fp, indent=2, sort_keys=True)
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
        """if __debug__:
            embed_before = th.tensor(merged_features[0].get("embedding", None))"""

        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            
        )
        """if __debug__:
            embed_after = batch.get("embedding", None)[0]
            th.testing.assert_close(embed_before, embed_after, atol=1e-6, rtol=1e-6)
        """
        #print(batch.keys())
        #assert "embedding" in batch, "Expected 'embedding' key in the batch after padding."
        #assert "context_embedding" in batch, "Expected 'context_embedding' key in the batch after padding."

        
        assert batch["input_ids"].shape[:-1] == batch["labels"].shape[:-1], f"Input IDs shape: {batch['input_ids'].shape}, Labels shape: {batch['labels'].shape}"
        """batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
            "return_loss": True, 
        }"""
        batch["return_loss"] = True
        batch["embedding"] = batch["embedding"].to(dtype=self.dtype) if "embedding" in batch.keys() else None
        batch["context_embedding"] = batch["context_embedding"].to(dtype=self.dtype) if "context_embedding" in batch.keys() else None
        if self.use_embeddings:
            assert batch["embedding"] is not None, "Expected 'embedding' key in the batch when use_embeddings is True."
            assert batch["context_embedding"] is not None, "Expected 'context_embedding' key in the batch when use_embeddings is True."
        
        return batch

def to_float(value: Any) -> float:
            if isinstance(value, th.Tensor):
                if value.numel() == 1:
                    return float(value.detach().cpu().item())
                return float(value.detach().cpu().mean().item())
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.item())
                return float(value.mean())
            return float(value)
class MORMTrainingVariables(th.nn.Module):

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
        if self.last_accumulated_vs_loss is not None:
            result["value_system_loss"] = to_float(self.last_accumulated_vs_loss)
        for i in range(len(self.lagrange_multipliers)):
            if self.lagrange_multipliers[i] is not None:
                result[f"lagrange_multiplier_{i}"] = to_float(self.lagrange_multipliers[i])
        if self.maximum_coherences_tendency is not None:
            result["maximum_coherences_tendency"] = to_float(self.maximum_coherences_tendency)
        if self.minimum_grounding_loss_tendency is not None:
            result["minimum_grounding_loss_tendency"] = to_float(self.minimum_grounding_loss_tendency)
        return result
    
    def forward(self, grounding_losses: th.Tensor, vs_losses: th.Tensor, target_gr_loss: th.Tensor = None) -> th.Tensor:
        
        
        if target_gr_loss is None or self.zero_constraint:
                lag_gr_loss = th.dot(self.lagrange_multipliers, grounding_losses)
        else:
                lag_gr_loss = th.dot(self.lagrange_multipliers, th.maximum(grounding_losses - target_gr_loss, th.zeros_like(grounding_losses)))
        
        with th.no_grad():
            weighting_factor = (1.0 / (1.0 + th.sum(self.lagrange_multipliers))).detach()
        #last_loss_original_unscaled = lag_gr_loss + vs_loss
        total_loss = weighting_factor * (lag_gr_loss +  vs_losses)

        return total_loss
    def to(self, *args, **kwargs) -> th.nn.Module:
        kwargs['dtype'] =  th.float32  
        self.lagrange_multipliers = self.lagrange_multipliers.to(*args, **kwargs)
         
        if self.minimum_grounding_loss_tendency is not None:
            self.minimum_grounding_loss_tendency = self.minimum_grounding_loss_tendency.to(*args, **kwargs) 
        if self.last_accumulated_grounding_loss is not None:
            self.last_accumulated_grounding_loss = self.last_accumulated_grounding_loss.to(*args, **kwargs)
        if self.last_accumulated_grounding_loss_ideal is not None:
            self.last_accumulated_grounding_loss_ideal = self.last_accumulated_grounding_loss_ideal.to(*args, **kwargs)
        for i in range(len(self._cached_groundings)):
            self._cached_groundings[i] = self._cached_groundings[i].to(*args, **kwargs) 
        for i in range(len(self._cached_groundings_ideal)):
            self._cached_groundings_ideal[i] = self._cached_groundings_ideal[i].to(*args, **kwargs) 
        for i in range(len(self._cached_vs_losses)):
            self._cached_vs_losses[i] = self._cached_vs_losses[i].to(*args, **kwargs)
        #print("MOVED Lagrange multipliers:", self.lagrange_multipliers, self.lagrange_multipliers.dtype)
        #exit(0)
        return super().to(*args, **kwargs)
    def parameters(self, recurse: bool = True) -> Any:
        return self.lagrange_multipliers
    
    
    def __init__(self, n_values: int, initial_lambda: int =1.0 , 
                 device: th.DeviceObjType|str ='cpu', dtype: th.Type = th.float32, 
                 grounding_loss_tendency_update_ratio: float = 0.01, 
                 gradient_accumulation_steps=10, 
                 metric_buffer_size=3, use_metrics_or_losses='metrics', 
                 grad_on_only_worst_value=False, zero_constraint: bool = True,
                 lambda_decay: float = 1e-9):
        super().__init__()
        self.lagrange_multipliers = th.tensor([initial_lambda]*n_values, requires_grad=False, device=device, dtype=dtype)
        self.zero_constraint = zero_constraint
        self.minimum_grounding_loss_tendency: th.Tensor | None = None
        self.maximum_coherences_tendency: th.Tensor | None = None

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
        self.grad_on_only_worst_value = grad_on_only_worst_value

        self.historic_eval_metrics = {}
    
    def prepare_for_optimizer_step(self) -> None:

        with th.no_grad():
            self.requires_grad_(True)
                #assert self.lagrange_multipliers[vi].grad is not None, f"Lagrange multiplier {vi} gradient is None before optimizer step."
            self.update_loss_tendencies()
            self.update_metrics_tendencies()

            lag_sum = 1.0+sum(self.lagrange_multipliers).detach()
            if self.use_metrics_or_losses == 'metrics':
                # Use metrics for the optimizer step
                gr_ideal_diff = -(self.last_accumulated_coherences - self.maximum_coherences_tendency).detach()
                
                coeff = gr_ideal_diff.to(device=lag_sum.device) / lag_sum
                
                
            elif self.use_metrics_or_losses == 'losses':
                gr_ideal_diff = (self.last_accumulated_grounding_loss - self.minimum_grounding_loss_tendency).detach()
                
                # This is the derivative w.r.t. lambda of "1/(1+lambda) * (lambda(gr_loss - gr_ideal) + vs_loss)".
                #should be...? 1) forward = self.forward(grounding_losses=gr_ideal_diff, vs_losses=self.last_accumulated_vs_loss)
                #should be...? 2) coeff = ((gr_ideal_diff*(lag_sum)) - 1*(forward))/th.pow(lag_sum, 2) 
                coeff = gr_ideal_diff.to(device=lag_sum.device)  / lag_sum
                # Use losses for the optimizer step
                
            
            
            print("COEFF OF CHANGE, ", coeff)
            print("DIFF", gr_ideal_diff)
            print("LOSS GR", self.last_accumulated_grounding_loss )
            print("LOSS TENDENCY", self.minimum_grounding_loss_tendency)
            print("CHR GR", self.last_accumulated_coherences )
            print("COHERENCE TENDENCY", self.maximum_coherences_tendency)

            if self.grad_on_only_worst_value:
                worst_vi = th.argmin(-coeff).item()
                coeff = coeff * th.nn.functional.one_hot(th.tensor(worst_vi, device=coeff.device), num_classes=coeff.shape[0])*self.lagrange_multipliers.shape[0]
                print("WORST VI", worst_vi)

            grad = th.clamp(-coeff, max=0.0, min=-1000.0)
            
            assert th.all(grad <= 0.0), f"Expected all gradients to be non-positive, but got {grad}"
        if self.lagrange_multipliers.grad is None:
            self.lagrange_multipliers.grad = grad
        else:
            self.lagrange_multipliers.grad += grad 
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.lagrange_multipliers.grad = None
        return super().zero_grad(set_to_none)
    def requires_grad_(self, requires_grad: bool = True):
        with th.no_grad():
            self.lagrange_multipliers.requires_grad_(requires_grad)
    def post_optimizer_step(self) -> None:
        with th.no_grad():
            if self.lambda_decay > 0.0:
                update = self.lagrange_multipliers.data * (1.0 - self.lambda_decay)
                self.lagrange_multipliers.data = th.clamp(update, min=self.initial_lambda, max=1000.0)
        #self.zero_grad()
        

    def reset_lagrange_gradients(self) -> None:
        with th.no_grad():
            self.requires_grad_(False)
            self.zero_grad()
            self.last_accumulated_grounding_loss = None
            self.last_accumulated_grounding_loss_ideal = None
            self.last_accumulated_coherences_ideal = None
            self.last_accumulated_coherences = None
            self.last_accumulated_representativeness = None
            self._cached_groundings_ideal = []
            self._cached_groundings = []
            self._cached_vs_losses = []
            self._cached_coherences = []
            self._cached_avg_coherence = []
            self._cached_representativeness = []
            self._cached_coherences_ideal = []


    def update_metrics_tendencies(self) -> None:
        with th.no_grad():
            self.last_accumulated_coherences = th.stack(self._cached_coherences).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach().clone().cpu()
            
            self.last_accumulated_representativeness = th.stack(self._cached_representativeness).mean().detach().clone().cpu().item()

            coherence_target =  self.last_accumulated_coherences
            if len(self._cached_coherences_ideal) > 0:
                self.last_accumulated_coherences_ideal = th.stack(self._cached_coherences_ideal).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach().clone().cpu()

                coherence_target = th.maximum(self.last_accumulated_coherences, self.last_accumulated_coherences_ideal)
            if self.maximum_coherences_tendency is None:
                self.maximum_coherences_tendency = th.full_like(coherence_target, fill_value=0.5).detach()
            else:
                maximum = th.maximum(coherence_target, self.maximum_coherences_tendency)
                self.maximum_coherences_tendency = (th.multiply(maximum, self.loss_metric_tendency_update_ratio) + th.multiply(self.maximum_coherences_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach().clone().cpu()
    def update_loss_tendencies(self) -> Optional[th.Tensor]:
        
        with th.no_grad():
            self.last_accumulated_grounding_loss = th.stack(self._cached_groundings).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach().clone().cpu()
            if len(self._cached_groundings_ideal) > 0:
                self.last_accumulated_grounding_loss_ideal = th.stack(self._cached_groundings_ideal).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach().clone().cpu()
            else:
                self.last_accumulated_grounding_loss_ideal = self.last_accumulated_grounding_loss.detach().clone().cpu()
            self.last_accumulated_vs_loss = th.stack(self._cached_vs_losses).mean().detach().clone().cpu().item()

            minimum_actual = th.minimum(self.last_accumulated_grounding_loss, self.last_accumulated_grounding_loss_ideal).detach().clone()
            assert self.last_accumulated_grounding_loss.shape == self.lagrange_multipliers.shape, f"Last accumulated grounding loss shape: {self.last_accumulated_grounding_loss.shape}, Lagrange multipliers shape: {self.lagrange_multipliers.shape}"
            if self.minimum_grounding_loss_tendency is None:
                self.minimum_grounding_loss_tendency = th.full_like(minimum_actual, fill_value=th.max(minimum_actual).float()).detach()
            else:
                minimum = th.minimum(minimum_actual, self.minimum_grounding_loss_tendency)

                self.minimum_grounding_loss_tendency = (th.multiply(minimum, self.loss_metric_tendency_update_ratio) + th.multiply(self.minimum_grounding_loss_tendency, (1.0 - self.loss_metric_tendency_update_ratio))).detach().clone().cpu()
                assert self.minimum_grounding_loss_tendency.shape == self.last_accumulated_grounding_loss.shape, f"Grounding loss tendency shape: {self.minimum_grounding_loss_tendency.shape}, Last grounding loss shape: {self.last_accumulated_grounding_loss.shape}"
            
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
                if len(self._cached_avg_coherence) > self.gradient_accumulation_steps:
                    self._cached_avg_coherence.pop(0)
                    self._cached_coherences.pop(0)
                    if len(self._cached_coherences_ideal) > 0:
                        self._cached_coherences_ideal.pop(0)
                    self._cached_representativeness.pop(0)
                
                coherences = metrics.get("coherences", None)
                if coherences is not None:
                    self._cached_coherences.append(coherences)
                    
                    avg_c = metrics.get("avg_coherence", None)
                    assert avg_c is not None, "Expected 'avg_coherence' in metrics when 'coherences' is present."
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
            
            if len(self._cached_groundings) > self.gradient_accumulation_steps:
                self._cached_groundings.pop(0)
                if len(self._cached_groundings_ideal) > 0:
                    self._cached_groundings_ideal.pop(0)
                self._cached_vs_losses.pop(0)
            self._cached_groundings.append(gr_loss_detached)
            self._cached_vs_losses.append(vs_loss_detached)
            if gr_loss_ideal_detached is not None:
                self._cached_groundings_ideal.append(gr_loss_ideal_detached)



