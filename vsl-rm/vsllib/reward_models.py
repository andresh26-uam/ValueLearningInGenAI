
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import torch as th
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel

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
        value_layer_intermediate_activation: str = "ReLU",
        value_layer_final_activation: str = "Tanh",
        **kwargs,
    ):
        assert num_values > 0, "num_values must be greater than 0"
        assert len(hidden_sizes) > 0, "hidden_sizes must be a non-empty list"

        if value_layer_intermediate_activation not in ['ReLU', 'Tanh', 'Softplus']:
             raise ValueError(f"value_layer_intermediate_activation must be one of 'ReLU', 'Tanh', 'Softplus', but got {value_layer_intermediate_activation}")
        if value_layer_final_activation not in ['ReLU', 'Tanh', 'Softplus', 'none']:
             raise ValueError(f"value_layer_final_activation must be one of 'ReLU', 'Tanh', 'Softplus', 'none', but got {value_layer_final_activation}")

        self.num_values = num_values
        self.value_layer_dropout = value_layer_dropout
        self.value_layer_intermediate_activation = value_layer_intermediate_activation
        self.value_layer_final_activation = value_layer_final_activation
        self.base_model = base_model
        super().__init__(**kwargs)

class MultiObjectiveRewardModel(PreTrainedModel):
    
    def construct_value_layer(self, config: MORMForSequenceClassificationConfig):
        layers = []
        input_size = config.hidden_size
        for hidden_size in config.hidden_sizes: 
            layers.append(nn.Linear(input_size, hidden_size))
            if config.value_layer_intermediate_activation == "ReLU":
                layers.append(nn.ReLU())
            elif config.value_layer_intermediate_activation == "Tanh":
                layers.append(nn.Tanh())
            elif config.value_layer_intermediate_activation == "Softplus":
                layers.append(nn.Softplus())
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
        elif config.value_layer_final_activation == "none":
            pass
        else:
            raise ValueError(f"Unsupported final activation: {config.value_layer_final_activation}")
        return nn.Sequential(*layers)
    
    def __init__(self, config: MORMForSequenceClassificationConfig):
        super().__init__(config)
        
        self.reward_heads = nn.ModuleList([
            self.construct_value_layer(config) for _ in range(config.num_values)
        ])
        self.base_model = config.base_model
        
        
		#self.score_weight_head: ConvexAlignmentLayer = ConvexAlignmentLayer(num_values, 1)
	
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return self.reward_heads.parameters(recurse)
	
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        pooled = last_hidden[:, -1, :]
        
        rewards: list[th.Tensor] = [head(pooled) for head in self.reward_heads]
        return th.cat(rewards, dim=1)
