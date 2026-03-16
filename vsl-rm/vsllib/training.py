from abc import abstractmethod
from random import sample
from typing import Any, Dict, List, Optional, Union
from datasets.arrow_dataset import Dataset
import numpy as np
import torch as th
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer
from datasets import DatasetDict, load_dataset, load_from_disk
from dataclasses import dataclass
from transformers.utils.generic import PaddingStrategy
from ordered_set import OrderedSet
from vsllib.defines import NO_RATING_MASK
from vsllib.reward_models import MORMForSequenceClassification
from vsllib.utils import MORMTrainingVariables



def tokenize_sample(sample: dict, tokenizer: Any, value_keys: list, delete_other_keys: bool = True, extra_keep_keys: list = None) -> dict:
    keep_keys = ["option1", "option2", "input_ids_1", "attention_mask_1", "input_ids_2", "attention_mask_2", "labels"]
    if extra_keep_keys:
        keep_keys.extend(extra_keep_keys)
    sample['option1'] = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response1']}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    sample['option2'] = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response2']}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    tokenized_pos = tokenizer(sample['option1'], truncation=True)
    tokenized_neg = tokenizer(sample['option2'], truncation=True)
    sample["input_ids_1"] = tokenized_pos["input_ids"]
    sample["attention_mask_1"] = tokenized_pos["attention_mask"]
    sample["input_ids_2"] = tokenized_neg["input_ids"]
    sample["attention_mask_2"] = tokenized_neg["attention_mask"]
    value_ratings1 = []
    value_ratings2 = []
    for key in value_keys:
        if sample.get(f"{key}_1", 'N/A') == 'N/A' or sample.get(f"{key}_2", 'N/A') == 'N/A':
            value_ratings1.append(NO_RATING_MASK)
            value_ratings2.append(NO_RATING_MASK)
        else:
            value_ratings1.append(float(sample.get(f"{key}_1", NO_RATING_MASK)))
            value_ratings2.append(float(sample.get(f"{key}_2", NO_RATING_MASK)))
            
    sample["labels"] = th.tensor(np.array([[*value_ratings1, sample.get("score1", NO_RATING_MASK)] , [*value_ratings2, sample.get("score2", NO_RATING_MASK)]]), dtype=th.float16)
    
    if delete_other_keys:	
        keys_to_delete = [key for key in sample.keys() if key not in keep_keys and not key.startswith("value_")]
    for key in keys_to_delete:
        del sample[key]	
    return sample

class PairwisePreferenceDataset(Dataset):
    
    def __init__(self, path: str, tokenizer, from_disk: bool = True, extra_keep_keys: list = None, retokenize: bool = False, split_seed: int = 42):
        if from_disk:
            self.data = load_from_disk(path).shuffle(seed=split_seed) 
        else:
            self.data = load_dataset(path, split="train").shuffle(seed=split_seed) 
        
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        # Extract value keys from the first data item
        if self.data:
            self.value_keys = [key for key in self.data[0].keys() if key.startswith("value_") and key.endswith("_1")]
            self.value_keys = [key.replace("_1", "") for key in self.value_keys]
        else:
            self.value_keys = []
        
        self.data: DatasetDict = self.data.map(lambda x: tokenize_sample(x, self.tokenizer, value_keys=self.value_keys, delete_other_keys=True, extra_keep_keys=extra_keep_keys), num_proc=16, load_from_cache_file=not retokenize)
        assert self.data[0].get("labels") is not None, "Labels are required in the dataset for training."
        self.data: DatasetDict = self.data.train_test_split(test_size=0.1, seed=split_seed) # pyright: ignore[reportAttributeAccessIssue]
        self.train_dataset, self.test_dataset = self.data['train'], self.data['test']	
        self.train_dataset = self.train_dataset.train_test_split(test_size=0.1, seed=split_seed)
        self.train_dataset, self.eval_dataset = self.train_dataset['train'], self.train_dataset['test']


	
    def __len__(self):
        return len(self.data)
    






class VSLOptimizer(th.optim.Optimizer):
    def __init__(self, params_gr: th.ParameterDict, params_vs: th.ParameterDict, n_values: int, lr=0.001, lr_grounding=None, lr_value_system=None, sub_optimizer_class=th.optim.Adam, scheduler="cosine", **optimizer_kwargs):
        lr_grounding = lr if lr_grounding is None else lr_grounding
        lr_value_system = lr if lr_value_system is None else lr_value_system
        self.lr_grounding = lr_grounding
        self.lr_value_system = lr_value_system
        defaults = dict(lr_grounding=lr_grounding,
                        lr_value_system=lr_value_system, lr_vt=lr_value_system)

        self.optimizer_kwargs = optimizer_kwargs
        self.n_values = n_values

        params_gr = OrderedSet(params_gr)
        params_vs = OrderedSet(params_vs)

        self.params_gr = params_gr
        self.params_vs = params_vs
        self.sub_optimizer_class = sub_optimizer_class
        self.optimx = sub_optimizer_class(
            params_gr, lr=lr_grounding, **self.optimizer_kwargs)
        if params_vs and len(params_vs) > 0:
            self.optimy = sub_optimizer_class(
                params_vs, lr=lr_value_system, **self.optimizer_kwargs)
        else:
            self.optimy = None
        # TODO: SCHEDULER COSINE... ALSO HANDLE SUBOPTIMIZER self.optimx_scheduler.step()
        # self.optimy_scheduler.step()
        super(VSLOptimizer, self).__init__([*params_gr, *params_vs], defaults)

    def get_state(self, copy=False)-> Dict[str, Any]:
        return {}

    def set_state(self, state)-> None:
        return

    @abstractmethod
    def zero_grad(self, set_to_none=True)-> None:
        super().zero_grad(set_to_none)
        self.optimx.zero_grad(set_to_none)
        if self.optimy is not None:
            self.optimy.zero_grad(set_to_none)
        return None

    @abstractmethod
    def step(self, closure=None)-> None:
        self.optimx.step()
        if self.optimy is not None:
            self.optimy.step()
        return None

class ConstrainedOptimizer(VSLOptimizer):
    def __init__(self, params, params_gr, params_vs, n_values, max_grad_norm, lr_grounding=None,
                 lr_value_system=None, lr_lambda=None, initial_lambda=1.0, lambda_decay=1e-9,
                 training_variables: MORMTrainingVariables = None,
                 sub_optimizer_class=th.optim.Adam, **optimizer_kwargs):
        super(ConstrainedOptimizer, self).__init__(params_gr=params_gr, params_vs=params_vs, n_values=n_values, lr=0.001,
                                                   lr_grounding=lr_grounding, lr_value_system=lr_value_system, sub_optimizer_class=sub_optimizer_class, **optimizer_kwargs)
        self.lr_lambda = lr_lambda if lr_lambda is not None else lr_value_system / 10.0
        self.initial_lambda = initial_lambda
        self.lambda_decay = lambda_decay
        self.max_grad_norm = max_grad_norm
        self.training_variables: MORMTrainingVariables = training_variables

        if self.lr_lambda > 0:
            self.optim_lambdas = th.optim.Adam(
                self.training_variables.lagrange_multipliers, lr=self.lr_lambda, betas=(0.5, 0.9))

    """@override
    def set_parameters(self, params_gr, params_vs, optim_state={}):
        self.params_gr = params_gr
        self.params_vs = params_vs
        self.optimx = self.sub_optimizer_class(params_gr, lr=self.lr_grounding, **self.optimizer_kwargs)
        self.optimy = self.sub_optimizer_class(params_vs, lr=self.lr_value_system, **self.optimizer_kwargs)"""

    def zero_grad(self, set_to_none=True)-> None:
        super().zero_grad(set_to_none)
        if self.lr_lambda > 0:
            self.training_variables.reset_lagrange_gradients()
        return None
    
    def step(self, closure=None)->None:
        #self.set_state({'time': 0, 'lambdas': training_variables.lagrange_multipliers})
        
        print("LAGRANGE MULTIPLIERS BEFORE STEP:", self.training_variables.lagrange_multipliers)
        self.time += 1
        th.nn.utils.clip_grad_norm_(self.params_gr, self.max_grad_norm)
        th.nn.utils.clip_grad_norm_(self.params_vs, self.max_grad_norm)
        self.optimx.step()
        if self.optimy is not None:
            self.optimy.step()
        

        if self.lr_lambda > 0:
            self.training_variables.prepare_for_optimizer_step()
            self.optim_lambdas.step()
            self.training_variables.post_optimizer_step()

        
            # print("LAMBDA a", self.training_variables.lagrange_multipliers)
        print("LAGRANGE MULTIPLIERS AFTER STEP:", self.training_variables.lagrange_multipliers)
        #self.set_state({'time': 0, 'lambdas': training_variables.lagrange_multipliers})
        return None
    
class MORewardTrainer(Trainer):
    
    def compute_metrics(eval_pred):
        result = {}
        pos_predictions_scores = eval_pred.predictions[0]
        neg_predictions_scores = eval_pred.predictions[1]

        print(eval_pred)
        print(eval_pred.predictions)
        input("Eval??")
        # We assume that the first sample is preferred by default in groundtruth
        result['representativeness'] = np.sum(
            pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
        
        result['representativeness'] = np.sum(
            pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
        
        return result

    # This assumes that the data is collated using RewardDataCollatorWithPadding, and that the model returns multiple rewards for each input.


