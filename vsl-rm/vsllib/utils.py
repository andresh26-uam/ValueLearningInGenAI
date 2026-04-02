

from copy import deepcopy
from dataclasses import dataclass
from heapq import merge
from typing import Any, Dict, List, Optional, Self, Union

from torch.nn.parameter import Parameter
from transformers import AutoTokenizer, DefaultDataCollator, loss


from transformers.utils import PaddingStrategy

import torch as th
import numpy as np
from triton.language import dtype

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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



@dataclass
class MORewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 16
    return_tensors: str = "pt"
    dtype: Optional[th.dtype] = None

    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []

        for feature in features:
            pair_labels = feature["labels"]
            c = feature.get("context_embedding", None)
            merged_features.append(
                {
                    "input_ids": feature["input_ids_1"],
                    "attention_mask": feature["attention_mask_1"],
                    "embedding": feature.get("embedding_1", None),
                    "context_embedding": c,
                    "labels": pair_labels[0],
                }
            )
            
            merged_features.append(
                {
                    "input_ids": feature["input_ids_2"],
                    "attention_mask": feature["attention_mask_2"],
                    "embedding": feature.get("embedding_2", None),
                    "context_embedding": c,
                    "labels": pair_labels[1],
                }
            )
        if __debug__:
            embed_before = th.tensor(merged_features[0].get("embedding", None))

        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            
        )
        if __debug__:
            embed_after = batch.get("embedding", None)[0]
            th.testing.assert_close(embed_before, embed_after, atol=1e-6, rtol=1e-6)
        
        #print(batch.keys())
        assert "embedding" in batch, "Expected 'embedding' key in the batch after padding."
        assert "context_embedding" in batch, "Expected 'context_embedding' key in the batch after padding."

        
        assert batch["input_ids"].shape[:-1] == batch["labels"].shape[:-1], f"Input IDs shape: {batch['input_ids'].shape}, Labels shape: {labels.shape}"
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
            "embeddings": batch["embedding"].to(dtype=self.dtype) if "embedding" in batch else None,
            "context_embedding": batch["context_embedding"].to(dtype=self.dtype) if "context_embedding" in batch else None,
            #"labels": th.cat([th.as_tensor(np.array(f['labels'], dtype=np.float16), dtype=th.float16) for f in features], dim=0).to(batch["input_ids"].device),
			#"score": th.tensor([f.get("score", 0.0) for f in merged_features], dtype=th.float32),
			#"value_ratings": [f.get("value_ratings", {}) for f in merged_features],
            "return_loss": True,
        }

        """if 'embedding_1' in features[0] and 'embedding_2' in features[0]:
            batch["embedding_1"] = th.tensor([f["embedding_1"] for f in features]).to(batch["input_ids"].device, dtype=th.float16)
            batch["embedding_2"] = th.tensor([f["embedding_2"] for f in features]).to(batch["input_ids"].device, dtype=th.float16)
        if 'context_embedding' in features[0]:
            batch["context_embedding"] = th.tensor([f["context_embedding"] for f in features]).to(batch["input_ids"].device)
    """
        return batch


class MORMTrainingVariables(th.nn.Module):

    def forward(self, grounding_losses: th.Tensor, vs_losses: th.Tensor, target_gr_loss: th.Tensor = None) -> th.Tensor:
        if target_gr_loss is None:
            lag_gr_loss = th.dot(self.lagrange_multipliers, grounding_losses)
        else:
            lag_gr_loss = th.dot(self.lagrange_multipliers, th.maximum(grounding_losses - target_gr_loss, th.zeros_like(grounding_losses)))
        with th.no_grad():
            weighting_factor = 1.0 / (1.0 + th.sum(self.lagrange_multipliers))
        #last_loss_original_unscaled = lag_gr_loss + vs_loss
        total_loss = weighting_factor * (lag_gr_loss +  vs_losses)

        return total_loss
    def to(self, *args, **kwargs) -> Self:
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
    
    
    def __init__(self, n_values: int, initial_lambda: int =1.0 , device: th.DeviceObjType|str ='cpu', dtype: th.Type = th.float32, grounding_loss_tendency_update_ratio: float = 0.01, gradient_accumulation_steps=10, lambda_decay: float = 1e-9):
        super().__init__()
        self.lagrange_multipliers = th.tensor([initial_lambda]*n_values, requires_grad=False, device=device, dtype=dtype)
        
        self.minimum_grounding_loss_tendency: th.Tensor | None = None
        self.last_accumulated_grounding_loss: th.Tensor | None = None

        self.last_accumulated_grounding_loss_ideal: th.Tensor | None = None
        self.grounding_loss_tendency_update_ratio = grounding_loss_tendency_update_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lambda_decay = lambda_decay
        self.initial_lambda = initial_lambda
        self._cached_groundings = []
        self._cached_groundings_ideal = []
        self._cached_vs_losses = [] 
    
    def prepare_for_optimizer_step(self) -> None:
        with th.no_grad():
            self.zero_grad()
            self.requires_grad_(True)
                #assert self.lagrange_multipliers[vi].grad is not None, f"Lagrange multiplier {vi} gradient is None before optimizer step."
            self.update_loss_tendencies()
            gr_ideal_diff = (self.last_accumulated_grounding_loss - self.minimum_grounding_loss_tendency)
            lag_sum = 1.0+sum(self.lagrange_multipliers).detach()
            # This is the derivative w.r.t. lambda of "1/(1+lambda) * (lambda(gr_loss - gr_ideal) + vs_loss)".
            forward = self.forward(grounding_losses=gr_ideal_diff, vs_losses=self.last_accumulated_vs_loss)
            #should be...? coeff = ((gr_ideal_diff*(lag_sum)) - 1*(forward))/th.pow(lag_sum, 2) 
            coeff = gr_ideal_diff / lag_sum
            
            print("COEFF OF CHANGE, ", coeff)
            print("DIFF", gr_ideal_diff)
            print("LAST GR", self.last_accumulated_grounding_loss )
            print("TENDENCY", self.minimum_grounding_loss_tendency)
            
            grad = th.clamp(-coeff, max=0.0, min=-1000.0)
        if self.lagrange_multipliers.grad is None:
            self.lagrange_multipliers.grad = grad
        else:
            self.lagrange_multipliers.grad += grad
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.lagrange_multipliers.grad = None
        return super().zero_grad(set_to_none)
    def requires_grad_(self, requires_grad: bool = True) -> Self:
        with th.no_grad():
            for vi in range(len(self.lagrange_multipliers)):
                
                self.lagrange_multipliers[vi].requires_grad_(requires_grad)
    def post_optimizer_step(self) -> None:
        

        self.lagrange_multipliers.grad = None

    def reset_lagrange_gradients(self) -> None:
        with th.no_grad():
            for vi in range(len(self.lagrange_multipliers)):
                self.lagrange_multipliers[vi].requires_grad_(False)
                self.lagrange_multipliers[vi].grad = None
            self.last_accumulated_grounding_loss = None
            self.last_accumulated_grounding_loss_ideal = None
            self._cached_groundings_ideal = []
            self._cached_groundings = []
            self._cached_vs_losses = []

    def update_loss_tendencies(self) -> Optional[th.Tensor]:
        with th.no_grad():
            self.last_accumulated_grounding_loss = th.stack(self._cached_groundings).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach().clone()
            self.last_accumulated_grounding_loss_ideal = th.stack(self._cached_groundings_ideal).mean(dim=0, dtype=self.lagrange_multipliers.dtype).detach().clone()

            self.last_accumulated_vs_loss = th.stack(self._cached_vs_losses).mean().detach().clone().item()

            minimum_actual = th.minimum(self.last_accumulated_grounding_loss, self.last_accumulated_grounding_loss_ideal).detach().clone()
            assert self.last_accumulated_grounding_loss.shape == self.lagrange_multipliers.shape, f"Last accumulated grounding loss shape: {self.last_accumulated_grounding_loss.shape}, Lagrange multipliers shape: {self.lagrange_multipliers.shape}"
            if self.minimum_grounding_loss_tendency is None:
                self.minimum_grounding_loss_tendency = minimum_actual
            else:
                minimum = th.minimum(minimum_actual, self.minimum_grounding_loss_tendency)

                self.minimum_grounding_loss_tendency = (th.multiply(minimum, self.grounding_loss_tendency_update_ratio) + th.multiply(self.minimum_grounding_loss_tendency, (1.0 - self.grounding_loss_tendency_update_ratio))).detach().clone()
                assert self.minimum_grounding_loss_tendency.shape == self.last_accumulated_grounding_loss.shape, f"Grounding loss tendency shape: {self.minimum_grounding_loss_tendency.shape}, Last grounding loss shape: {self.last_accumulated_grounding_loss.shape}"
            

    def record_grounding_loss(self, gr_loss_detached: th.Tensor, vs_loss_detached: th.Tensor, loss_gr_ideal=None) -> None:
        with th.no_grad():
            for vi in range(len(self.lagrange_multipliers)):
                self.lagrange_multipliers[vi].requires_grad_(False)
            
            # This estimates the minimum obtainable loss for each value. The optimizer will take this into account
            
            if len(self._cached_groundings) >= self.gradient_accumulation_steps:
                self._cached_groundings.pop(0)
                self._cached_groundings_ideal.pop(0)
                self._cached_vs_losses.pop(0)
            self._cached_groundings.append(gr_loss_detached)
            self._cached_vs_losses.append(vs_loss_detached)
            self._cached_groundings_ideal.append(loss_gr_ideal)

            