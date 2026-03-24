from abc import abstractmethod
from copy import deepcopy
from random import sample
from typing import Any, Dict, List, Optional, Union
from datasets.arrow_dataset import Dataset
from matplotlib.pylab import dtype
import numpy as np
import torch as th
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer
from transformers.optimization import get_scheduler
from datasets import DatasetDict, load_dataset, load_from_disk
from dataclasses import dataclass
from transformers.utils.generic import PaddingStrategy
from transformers.trainer_utils import SchedulerType
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
        self.train_dataset = self.train_dataset.train_test_split(test_size=0.0025, seed=split_seed)
        self.train_dataset, self.eval_dataset = self.train_dataset['train'], self.train_dataset['test']


	
    def __len__(self):
        return len(self.data)
    





class VSLOptimizer(th.optim.Optimizer):
    def __init__(self, params_gr: th.ParameterDict, params_vs: th.ParameterDict, n_values: int, lr_grounding=None, lr_value_system=None, sub_optimizer_class=th.optim.Adam, scheduler="cosine", **optimizer_kwargs):
        
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

        print("SUBOPTIMIZER CLASS:", sub_optimizer_class)
        

        copyargs= deepcopy(self.optimizer_kwargs)
        copyargs['lr'] = lr_grounding
        self.sub_optimizer_class = sub_optimizer_class
        self.optimx = sub_optimizer_class(
            params_gr, **self.optimizer_kwargs)
        if params_vs and len(params_vs) > 0:
            copyargs= deepcopy(self.optimizer_kwargs)
            copyargs['lr'] = lr_value_system
            self.optimy = sub_optimizer_class(
                params_vs, **self.optimizer_kwargs)
        else:
            self.optimy = None
        # TODO: SCHEDULER COSINE...? ALSO HANDLE SUBOPTIMIZER self.optimx_scheduler.step()
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
        super(ConstrainedOptimizer, self).__init__(params_gr=params_gr, params_vs=params_vs, n_values=n_values,
                                                   lr_grounding=lr_grounding, lr_value_system=lr_value_system, sub_optimizer_class=sub_optimizer_class, **optimizer_kwargs)
        self.lr_lambda = lr_lambda if lr_lambda is not None else lr_value_system * 10.0
        self.initial_lambda = initial_lambda
        self.lambda_decay = lambda_decay
        self.max_grad_norm = max_grad_norm
        self.training_variables: MORMTrainingVariables = training_variables

        if self.lr_lambda > 0:
            self.optim_lambdas = th.optim.Adam(
                (self.training_variables.lagrange_multipliers,), lr=self.lr_lambda, betas=(0.5, 0.9))
        self.time = 0
    def zero_grad(self, set_to_none=True)-> None:
        super().zero_grad(set_to_none)
        if self.lr_lambda > 0:
            self.training_variables.reset_lagrange_gradients()
        return None
    
    def step(self, closure=None)->None:
        #self.set_state({'time': 0, 'lambdas': training_variables.lagrange_multipliers})
        
        print("LAGRANGE MULTIPLIERS BEFORE STEP:", self.training_variables.lagrange_multipliers)
        #print("TRAINING VARS", vars(self.training_variables))
        self.time += 1
        #th.nn.utils.clip_grad_norm_(self.params_gr, self.max_grad_norm)
        #th.nn.utils.clip_grad_norm_(self.params_vs, self.max_grad_norm)
        self.optimx.step()
        if self.optimy is not None:
            self.optimy.step()
        

        if self.lr_lambda > 0:
            self.training_variables.prepare_for_optimizer_step()
            assert self.optim_lambdas.param_groups[0]['params'][0] is self.training_variables.lagrange_multipliers, "Lagrange multipliers not found in optimizer parameters"
            print("OPTIM LAMBDAS", self.optim_lambdas.param_groups[0]['params'][0]  )
            print("TRAINING_VARS", self.training_variables.lagrange_multipliers )
            
            self.optim_lambdas.step()
            self.training_variables.post_optimizer_step()

        
            # print("LAMBDA a", self.training_variables.lagrange_multipliers)
        print("LAGRANGE MULTIPLIERS AFTER STEP:", self.training_variables.lagrange_multipliers)
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

    def step(self, metric=None):
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
        return lrs

    """@override
    def set_parameters(self, params_gr, params_vs, optim_state={}):
        self.params_gr = params_gr
        self.params_vs = params_vs
        self.optimx = self.sub_optimizer_class(params_gr, lr=self.lr_grounding, **self.optimizer_kwargs)
        self.optimy = self.sub_optimizer_class(params_vs, lr=self.lr_value_system, **self.optimizer_kwargs)"""

from accelerate.optimizer import AcceleratedOptimizer
    
class MORewardTrainer(Trainer):

    def create_optimizer(self, model: MORMForSequenceClassification =None) -> th.optim.Optimizer:
        self.optimizer = super().create_optimizer(model)

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[th.optim.Optimizer] = None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = optimizer if optimizer is not None else self.optimizer
        print("Creating scheduler with optimizer: ", optimizer, "???")
        print(optimizer.__class__.__name__)
        #print(vars(optimizer))
        if (isinstance(optimizer, AcceleratedOptimizer) and isinstance(optimizer.optimizer, ConstrainedOptimizer)):
            constrained_optim = optimizer.optimizer
        elif isinstance(optimizer, ConstrainedOptimizer):
            constrained_optim = optimizer
        else:
            raise ValueError("Optimizer must be an instance of ConstrainedOptimizer or AcceleratedOptimizer wrapping a ConstrainedOptimizer. Unregistered optimizer type: {}".format(type(optimizer)))
            return super().create_scheduler(num_training_steps, optimizer)
        scheduler_name = SchedulerType(self.args.lr_scheduler_type)
        warmup_steps = self.args.get_warmup_steps(num_training_steps)

        sched_x = get_scheduler(
                name=scheduler_name,
                optimizer=constrained_optim.optimx,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        sched_y = None
        if constrained_optim.optimy is not None:
            sched_y = get_scheduler(
                name=scheduler_name,
                optimizer=constrained_optim.optimy,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        sched_lambda = None
        if getattr(constrained_optim, "optim_lambdas", None) is not None:
            sched_lambda = get_scheduler(
                name=scheduler_name,
                optimizer=constrained_optim.optim_lambdas,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        self.lr_scheduler = ConstrainedLRScheduler(
            optimizer=constrained_optim,
            sched_x=sched_x,
            sched_y=sched_y,
            sched_lambda=sched_lambda,
        )
        print("Optimizer and schedulers created successfully.")
        
        return self.lr_scheduler

        
    
    def compute_metrics(eval_pred):
        result = {}
        bsz = eval_pred.predictions.shape[0]

        jidx = th.arange(0, bsz, 2, device=eval_pred.predictions.device)
        kidx = jidx + 1
        rewards_1 = eval_pred.predictions[jidx]
        rewards_2 = eval_pred.predictions[kidx]
        print(vars(eval_pred))
        labels_1 = np.asarray(eval_pred.label_ids[jidx], dtype=rewards_1.dtype)
        labels_2 = np.asarray(eval_pred.label_ids[kidx], dtype=rewards_2.dtype)

        print(eval_pred)
        print(eval_pred.predictions.shape)
        print("LABELS", eval_pred.label_ids)
        print(rewards_1[...,-1].shape)
        
        # We assume that the first sample is preferred by default in groundtruth
        rep_mask1 = (rewards_1[..., -1] > rewards_2[..., -1]) & (labels_1[..., -1] >= labels_2[..., -1])
        rep_mask2 = (rewards_1[..., -1] < rewards_2[..., -1]) & (labels_1[..., -1] <= labels_2[..., -1])
        rep_mask3 = rep_mask = (rewards_1[..., -1] == rewards_2[..., -1]) & (labels_1[..., -1] == labels_2[..., -1])
        rep_mask = rep_mask1 | rep_mask2 | rep_mask3
        result['representativeness'] = np.sum(rep_mask) / len(rewards_1)
        
        coh_mask1 = (rewards_1[..., 0:-1] >= rewards_2[..., 0:-1]) & (labels_1[..., 0:-1] >= labels_2[..., 0:-1])
        coh_mask2 = (rewards_1[..., 0:-1] <= rewards_2[..., 0:-1]) & (labels_1[..., 0:-1] <= labels_2[..., 0:-1])
        coh_mask3 = (rewards_1[..., 0:-1] == rewards_2[..., 0:-1]) & (labels_1[..., 0:-1] == labels_2[..., 0:-1])
        coh_mask = coh_mask1 | coh_mask2 | coh_mask3
        result['coherence'] = np.sum(coh_mask, axis=0) / len(rewards_1)
        assert result['coherence'].shape == (rewards_1.shape[-1]-1,), f"Coherence shape: {result['coherence'].shape}, Expected shape: {(rewards_1.shape[-1]-1,)}"

        return result

    # This assumes that the data is collated using RewardDataCollatorWithPadding, and that the model returns multiple rewards for each input.


