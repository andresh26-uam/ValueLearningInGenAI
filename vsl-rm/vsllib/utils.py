
# We need to define a special data collator that batches the data in our j vs k format.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers import AutoTokenizer


from transformers.utils import PaddingStrategy

import torch as th
import numpy as np
@dataclass
class MORewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        merged_labels = []
        for feature in features:
            pair_labels = feature["labels"]
            if hasattr(pair_labels, "tolist"):
                pair_labels = pair_labels.tolist()

            merged_features.append(
                {
                    "input_ids": feature["input_ids_1"],
                    "attention_mask": feature["attention_mask_1"],
                }
            )
            merged_labels.append(pair_labels[0])
            merged_labels.append(pair_labels[1])
            merged_features.append(
                {
                    "input_ids": feature["input_ids_2"],
                    "attention_mask": feature["attention_mask_2"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        labels_np = np.asarray(merged_labels, dtype=np.float32)
        if self.return_tensors == "pt":
            labels = th.as_tensor(labels_np, dtype=th.float32)
        elif self.return_tensors == "np":
            labels = labels_np
        elif self.return_tensors == "tf":
            labels = labels_np
        else:
            labels = merged_labels
        
        assert batch["input_ids"].shape[:-1] == labels.shape[:-1], f"Input IDs shape: {batch['input_ids'].shape}, Labels shape: {labels.shape}"
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
            #"labels": th.cat([th.as_tensor(np.array(f['labels'], dtype=np.float16), dtype=th.float16) for f in features], dim=0).to(batch["input_ids"].device),
			#"score": th.tensor([f.get("score", 0.0) for f in merged_features], dtype=th.float32),
			#"value_ratings": [f.get("value_ratings", {}) for f in merged_features],
            "return_loss": True,
        }
        return batch


class MORMTrainingVariables:
    lagrange_multipliers: th.Tensor

    def __init__(self, n_values: int, initial_lambda: int =1.0 , device: th.DeviceObjType|str ='cpu', grounding_loss_tendency_update_ratio: float = 0.01, gradient_accumulation_steps=10, lambda_decay: float = 1e-9):
        self.lagrange_multipliers = th.tensor([initial_lambda]*n_values, requires_grad=False, device=device)
        self.minimum_grounding_loss_tendency: th.Tensor | None = None
        self.last_accumulated_grounding_loss: th.Tensor | None = None
        self.grounding_loss_tendency_update_ratio = grounding_loss_tendency_update_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lambda_decay = lambda_decay
        self.initial_lambda = initial_lambda
        self._cached_groundings = []
        self._cached_vs_losses = [] 
    

    def prepare_for_optimizer_step(self) -> None:
        with th.no_grad():
            for vi in range(len(self.lagrange_multipliers)):
                self.lagrange_multipliers[vi].requires_grad_(True)
                assert self.lagrange_multipliers[vi].grad is not None, f"Lagrange multiplier {vi} gradient is None before optimizer step."
            self.update_loss_tendencies()
            gr_ideal_diff = th.clamp(self.last_accumulated_grounding_loss - self.minimum_grounding_loss_tendency, min=0.0)
            lag_sum = 1+sum(self.lagrange_multipliers)
            # This is the derivative w.r.t. lambda of "1/(1+lambda) * (gr_loss + vs_loss)". 
            self.lagrange_multipliers.grad += -th.clamp(((gr_ideal_diff*(lag_sum)) - (self.last_accumulated_vs_loss - th.dot(gr_ideal_diff,self.lagrange_multipliers)))/th.pow(lag_sum, 2) , min=0.0)
    
    def post_optimizer_step(self) -> None:
        with th.no_grad():
            for vi in range(len(self.lagrange_multipliers)):
                self.lagrange_multipliers[vi].requires_grad_(False)

        if self.lambda_decay > 0:
            with th.no_grad():
                for vi in range(len(self.lagrange_multipliers)):
                    if self.lagrange_multipliers[vi] > self.initial_lambda:
                        decay = (self.lagrange_multipliers[vi].detach()*self.lambda_decay)
                        self.lagrange_multipliers[vi].data = th.clamp(
                            self.lagrange_multipliers[vi].data - decay, min=self.initial_lambda)


    def reset_lagrange_gradients(self) -> None:
        with th.no_grad():
            for vi in range(len(self.lagrange_multipliers)):
                self.lagrange_multipliers[vi].requires_grad_(False)
                self.lagrange_multipliers[vi].grad = th.zeros_like(self.lagrange_multipliers[vi])
            self.last_accumulated_grounding_loss = None
            self._cached_groundings = []
            self._cached_vs_losses = []

    def update_loss_tendencies(self) -> Optional[th.Tensor]:
        self.last_accumulated_grounding_loss = th.stack(self._cached_groundings).mean(dim=0)[0]
        self.last_accumulated_vs_loss = th.stack(self._cached_vs_losses).mean(dim=0)[0]
        if self.minimum_grounding_loss_tendency is None:
            self.minimum_grounding_loss_tendency = self.last_accumulated_grounding_loss
        else:
            self.minimum_grounding_loss_tendency = self.grounding_loss_tendency_update_ratio * th.minimum(self.last_accumulated_grounding_loss, self.minimum_grounding_loss_tendency) + (1-self.grounding_loss_tendency_update_ratio)*self.minimum_grounding_loss_tendency
            assert self.minimum_grounding_loss_tendency.shape == self.last_accumulated_grounding_loss.shape, f"Grounding loss tendency shape: {self.minimum_grounding_loss_tendency.shape}, Last grounding loss shape: {self.last_accumulated_grounding_loss.shape}"
            

    def record_grounding_loss(self, gr_loss_detached: th.Tensor, vs_loss_detached: th.Tensor ) -> None:
        with th.no_grad():
            for vi in range(len(self.lagrange_multipliers)):
                self.lagrange_multipliers[vi].requires_grad_(False)
            
            # This estimates the minimum obtainable loss for each value. The optimizer will take this into account
            

            if len(self._cached_groundings) < self.gradient_accumulation_steps:
                self._cached_groundings.append(gr_loss_detached)
                self._cached_vs_losses.append(vs_loss_detached)

            else:
                self._cached_groundings.pop(0)
                self._cached_vs_losses.pop(0)
                self._cached_groundings.append(gr_loss_detached)
                self._cached_vs_losses.append(vs_loss_detached)
            
            