from abc import abstractmethod
from calendar import c
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import inspect
from pathlib import Path
from random import sample
import shutil
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from click import File
from datasets.arrow_dataset import Dataset
from matplotlib.pylab import dtype
from networkx import constraint
import numpy as np
from pyarrow import dataset
from rich import constrain
from sympy import re
import torch as th
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, DefaultDataCollator, Trainer, loss

from transformers.trainer import *

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.optimization import get_scheduler
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from dataclasses import dataclass
from transformers.utils.generic import PaddingStrategy
from transformers.trainer_utils import SchedulerType
from ordered_set import OrderedSet
from vsllib.defines import NO_RATING_MASK
from vsllib.reward_models import MORMForSequenceClassification, accuracy_logits, accuracy_rewards_labels
from vsllib.utils import MORMTrainingVariables, MORewardDataCollatorWithPadding



def tokenize_sample(sample: dict, tokenizer: Any, value_keys: list, delete_other_keys: bool = True, extra_keep_keys: list = None, use_context: bool =True) -> dict:
    keep_keys = ["option1", "option2", "input_ids_1", "attention_mask_1", "input_ids_2", "attention_mask_2", "labels"]
    if extra_keep_keys:
        keep_keys.extend(extra_keep_keys)
    sample['option1'] = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response1']}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    sample['option2'] = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response2']}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    if use_context:
        if sample.get("context", None) is not None:
            ctx = sample['context']
        else:
            ctx = sample['prompt']
        ctemplate = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': ctx}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        tok_context = tokenizer(ctemplate, truncation=True)
        sample['context_input_ids'] = tok_context["input_ids"]
        sample['context_attention_mask'] = tok_context["attention_mask"]
        keep_keys.extend(["context_input_ids", "context_attention_mask", "context"])
    
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

def embed_sample(sample: dict, model: BaseModelOutputWithPast, tokenizer: AutoTokenizer, collator: MORewardDataCollatorWithPadding, use_context: bool =True) -> dict:
    # THIS ASSUMES BATCHED MAPPING FUNCTION.
    model_device = next(model.parameters()).device
    for ic, case in enumerate([("input_ids_1", "attention_mask_1", "embedding_1"), ("input_ids_2", "attention_mask_2", "embedding_2"), ("context_input_ids", "context_attention_mask", "context_embedding")]):
        if ic == 2 and not use_context:
            continue
        merged_features = {
            "input_ids": sample[case[0]],
            "attention_mask": sample[case[1]],
        }
        
        batch = tokenizer.pad(
            merged_features,
            padding=collator.padding,
            max_length=collator.max_length,
            pad_to_multiple_of=collator.pad_to_multiple_of,
            return_tensors=collator.return_tensors,
        )
        inputs = batch["input_ids"].to(model_device)
        atm = batch["attention_mask"].to(model_device)
        output = model(
            input_ids=inputs,
            attention_mask=atm,
            return_dict=True,
        ).last_hidden_state
        # Pick the last non-padding token embedding for each sequence.
        last_token_idx = atm.sum(dim=1) - 1

        sample[case[2]] = output[np.arange(output.size(0)), last_token_idx].detach().cpu()
        del output
        
    return sample

class PairwisePreferenceDataset(Dataset):
    
    def __init__(self, path: str, tokenizer, from_disk: bool = True, extra_keep_keys: list = None, retokenize: bool = False, recalculate_embeddings: bool = False, model_for_embeddings: AutoModelForCausalLM = None, collator: MORewardDataCollatorWithPadding = None, use_context: bool = True, split_seed: int = 42, embedded_dataset_output_path: Optional[str] = None, cleanup_cache_files: bool = True):
        should_rewrite_embedded_dataset = bool(retokenize or recalculate_embeddings)

        if model_for_embeddings is not None and embedded_dataset_output_path is None:
            embedded_dataset_output_path = f"{path.rstrip('/')}_embed_{model_for_embeddings.config._name_or_path.replace('/', '_')}"

        if recalculate_embeddings:
            shutil.rmtree(embedded_dataset_output_path, ignore_errors=True)

        if from_disk:

            if model_for_embeddings is not None and not should_rewrite_embedded_dataset:
                try:
                    self.data = load_from_disk(embedded_dataset_output_path).shuffle(seed=split_seed)
                except FileNotFoundError:
                    print(f"Embedded dataset not found at {embedded_dataset_output_path}. Loading (tentatively tokenized) dataset from {path}.")
                    self.data = load_from_disk(path).shuffle(seed=split_seed)
            else:
                
                self.data = load_from_disk(path).shuffle(seed=split_seed)
        else:
            self.data = load_dataset(path, split="train").shuffle(seed=split_seed) 
        
        self.max_length = tokenizer.model_max_length
        # Extract value keys from the first data item
        if self.data:
            self.value_keys = [key for key in self.data[0].keys() if key.startswith("value_") and key.endswith("_1")]
            self.value_keys = [key.replace("_1", "") for key in self.value_keys]
        else:
            self.value_keys = []
        
        if self.data[0].get("labels") is None:
            self.data: DatasetDict = self.data.map(lambda x: tokenize_sample(x, tokenizer, value_keys=self.value_keys, delete_other_keys=True, extra_keep_keys=extra_keep_keys, use_context=use_context), num_proc=16, load_from_cache_file=not retokenize)
        
        if model_for_embeddings is not None and recalculate_embeddings:
            batch_size = 16
            #self.data = self.data.select(range(min(1000, len(self.data))))
            with th.no_grad():
                def _embed_shard(dataset_shard, device):
                    local_model = deepcopy(model_for_embeddings).to(device)
                    local_model.eval()
                    return dataset_shard.map(
                        lambda x: embed_sample(x, local_model, tokenizer, collator, use_context=use_context),
                        load_from_cache_file=False,
                        batched=True,
                        batch_size=batch_size,
                    )

                if th.cuda.is_available() and th.cuda.device_count() > 1:
                    n_gpus = th.cuda.device_count()
                    print(f"Embedding map sharded across {n_gpus} GPUs")

                    shards = [
                        self.data.shard(num_shards=n_gpus, index=i, contiguous=True)
                        for i in range(n_gpus)
                    ]

                    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                        futures = [
                            executor.submit(_embed_shard, shard, th.device(f"cuda:{i}"))
                            for i, shard in enumerate(shards)
                        ]
                        mapped_shards = [f.result() for f in futures]

                    self.data = concatenate_datasets(mapped_shards)
                else:
                    device = th.device("cuda" if th.cuda.is_available() else "cpu")
                    model_for_embeddings = model_for_embeddings.to(device)
                    model_for_embeddings.eval()
                    self.data = self.data.map(
                        lambda x: embed_sample(x, model_for_embeddings, tokenizer, collator, use_context=use_context),
                        load_from_cache_file=False,
                        batched=True,
                        batch_size=batch_size,
                    )

        if model_for_embeddings is not None and should_rewrite_embedded_dataset:
            output_path = Path(embedded_dataset_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_output_path = output_path.parent / f".{output_path.name}.tmp-{uuid4().hex}"
            if temp_output_path.exists():
                shutil.rmtree(temp_output_path)
            self.data.save_to_disk(str(temp_output_path))
            if output_path.exists():
                shutil.rmtree(output_path)
            temp_output_path.replace(output_path)
            print(f"Saved embedded dataset to {output_path}")

        if cleanup_cache_files:
            removed_cache_files = self.data.cleanup_cache_files()
            print(f"Removed {removed_cache_files} dataset cache files")

        assert self.data[0].get("labels") is not None, "Labels are required in the dataset for training."
        self.data: DatasetDict = self.data.train_test_split(test_size=0.1, seed=split_seed) # pyright: ignore[reportAttributeAccessIssue]
        self.train_dataset, self.test_dataset = self.data['train'], self.data['test']	
        self.train_dataset = self.train_dataset.train_test_split(test_size=0.02, seed=split_seed)
        self.train_dataset, self.eval_dataset = self.train_dataset['train'], self.train_dataset['test']

    def __len__(self):
        return len(self.data)
    



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
                        lr_value_system=lr_value_system, lr_vt=lr_value_system)

        self.optimizer_kwargs = optimizer_kwargs
        self.n_values = n_values

        params_gr = OrderedSet(params_gr)
        params_vs = OrderedSet(params_vs)

        self.params_gr = params_gr
        self.params_vs = params_vs

        print("SUBOPTIMIZER CLASS:", sub_optimizer_class)
        self.sub_optimizer_class = sub_optimizer_class

        self.optimx = _create_sub_optimizer(params_gr, lr_grounding, self.sub_optimizer_class, self.optimizer_kwargs)
        if params_vs and len(params_vs) > 0:
            self.optimy = _create_sub_optimizer(params_vs, lr_value_system, self.sub_optimizer_class, self.optimizer_kwargs)
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
    def __init__(self, params, params_gr, params_vs, n_values, max_grad_norm, params_gr_ideal=None, lr_grounding=None,
                 lr_value_system=None, lr_lambda=None, initial_lambda=1.0, inner_optimization_iterations=3,
                 training_variables: MORMTrainingVariables = None,
                 sub_optimizer_class=th.optim.Adam, **optimizer_kwargs):
        super(ConstrainedOptimizer, self).__init__(params_gr=params_gr, params_vs=params_vs, n_values=n_values,
                                                   lr_grounding=lr_grounding, lr_value_system=lr_value_system, sub_optimizer_class=sub_optimizer_class, **optimizer_kwargs)
        if params_gr_ideal is not None:
            assert len(params_gr) == len(params_gr_ideal), "Grounding parameters and ideal grounding parameters must have the same length."
            
            self.params_gr_ideal = params_gr_ideal
            self.optimx_ideal = _create_sub_optimizer(params_gr_ideal, lr_grounding, self.sub_optimizer_class, optimizer_kwargs)

        self.lr_lambda = lr_lambda if lr_lambda is not None else lr_value_system * 10.0
        self.initial_lambda = initial_lambda
        self.max_grad_norm = max_grad_norm
        self.training_variables: MORMTrainingVariables = training_variables

        self.current_iteration = 0
        self.inner_optimization_iterations = inner_optimization_iterations # Number of inner optimization steps for the value system per outer step.

        if self.lr_lambda > 0:
            self.optim_lambdas = th.optim.SGD(
                (self.training_variables.lagrange_multipliers,), lr=self.lr_lambda, weight_decay=0.0)
        self.time = 0
        self.training_variables.reset_lagrange_gradients()
    def zero_grad(self, set_to_none=True)-> None:
        super().zero_grad(set_to_none)
        if self.lr_lambda > 0:
            self.training_variables.requires_grad_(False)
            self.training_variables.zero_grad(set_to_none)
            self.optim_lambdas.zero_grad(set_to_none)
        if hasattr(self, 'optimx_ideal'):
            self.optimx_ideal.zero_grad(set_to_none)
        return None
    
    def custom_backward(self, loss_gr, loss_gr_ideal, loss_vs, **kwargs) -> th.Tensor:
        
        x = self.params_gr # Possibly need flatten into single tensor.
        
        w = self.params_vs
        #self.zero_grad()    

        
        # Just optimize the value system for a number of iterations before doing the full constrained optimization step.
        
        #print("BEFORE", x[0].data[0:10])
        assert x[0] is self.optimx.param_groups[0]['params'][0], "Grounding parameters do not match those in the optimizer"
        assert w[0] is self.optimy.param_groups[0]['params'][0], "Value system parameters do not match those in the optimizer"
        assert x[0].requires_grad, "Grounding parameters must require gradients for stoic optimization."
        # PARAMS", self.optimx.param_groups[0]['params'][0].data[0:10])

        if loss_gr_ideal is not None:
            self.optimx_ideal.zero_grad()
            loss_sum = th.sum(loss_gr_ideal).detach()
            for i in range(len(loss_gr_ideal)):
                    (loss_gr_ideal[i]/loss_sum).backward(retain_graph=True)
                    self.optimx_ideal.step()
                    self.optimx_ideal.zero_grad()
        
        self.training_variables.requires_grad_(False)
        assert self.optim_lambdas.param_groups[0]['params'][0] is self.training_variables.lagrange_multipliers, "Lagrange multipliers not found in optimizer parameters"    
        
        loss = self.training_variables.forward(loss_gr, loss_vs, target_gr_loss=loss_gr_ideal.detach() if loss_gr_ideal is not None else None)
        loss.backward(**kwargs)
        return loss
    def step(self, closure=None)->None:
        #self.set_state({'time': 0, 'lambdas': training_variables.lagrange_multipliers})
        
        print("LAGRANGE MULTIPLIERS BEFORE STEP:", self.training_variables.lagrange_multipliers)
        #print("TRAINING VARS", vars(self.training_variables))
        self.time += 1
        #th.nn.utils.clip_grad_norm_(self.params_gr, self.max_grad_norm)
        #th.nn.utils.clip_grad_norm_(self.params_vs, self.max_grad_norm)

        assert self.params_vs[0].grad is not None, "Value system gradients have not been computed. Make sure to call the backward pass on the value system loss before stepping the optimizer."
       
        self.optimx.step()
        self.optimy.step()
        

        if self.lr_lambda > 0:
            self.training_variables.prepare_for_optimizer_step()
            assert self.optim_lambdas.param_groups[0]['params'][0] is self.training_variables.lagrange_multipliers, "Lagrange multipliers not found in optimizer parameters"
            
            self.optim_lambdas.step()
            print("LAGRANGE MULTIPLIERS AFTER STEP (BEFORE DECAY):", self.training_variables.lagrange_multipliers)
            self.training_variables.post_optimizer_step()

            
        #self.zero_grad()

        print("LAGRANGE MULTIPLIERS AFTER STEP:", self.training_variables.lagrange_multipliers)
        #input("...")
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
        print("Creating scheduler with optimizer: ", optimizer)
        print(optimizer.__class__.__name__)
        
        if (isinstance(optimizer, AcceleratedOptimizer) and isinstance(optimizer.optimizer, ConstrainedOptimizer)):
            constrained_optim = optimizer.optimizer
        elif isinstance(optimizer, ConstrainedOptimizer):
            constrained_optim = optimizer
        else:
            raise ValueError("Optimizer must be an instance of ConstrainedOptimizer or AcceleratedOptimizer wrapping a ConstrainedOptimizer. Unregistered optimizer type: {}".format(type(optimizer)))
            return super().create_scheduler(num_training_steps, optimizer)
        scheduler_name = SchedulerType(self.args.lr_scheduler_type)
        warmup_steps = self.args.get_warmup_steps(num_training_steps)
        #warmup_steps = 0 # TODO TODO TODO !!!!!!!!!

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

        
    
    def compute_metrics(eval_pred, training_variables: MORMTrainingVariables) -> Dict[str, float]:
        with th.no_grad():
            result = {}
            
            if eval_pred.predictions.shape[0] % 2 == 1:
                eval_pred.predictions = eval_pred.predictions[:-1]
            bsz = eval_pred.predictions.shape[0]

            jidx = th.arange(0, bsz, 2, device=eval_pred.predictions.device)
            kidx = jidx + 1
            rewards_1 = eval_pred.predictions[jidx]
            rewards_2 = eval_pred.predictions[kidx]
            
            labels_1 = eval_pred.label_ids[jidx]
            labels_2 = eval_pred.label_ids[kidx]

            # We assume that the first sample is preferred by default in groundtruth
            represent = accuracy_rewards_labels(rewards_1[..., -1], rewards_2[..., -1], labels_1[..., -1], labels_2[..., -1], threshold=50.0, epsilon=1.0e-1, assume_torch=False)
            result['representativeness'] = represent

            chr = accuracy_rewards_labels(rewards_1[..., 0:-1], rewards_2[..., 0:-1], labels_1[..., 0:-1], labels_2[..., 0:-1],  threshold=50.0, epsilon=1.0e-1 , assume_torch=False)
            result['coherences'] = chr.tolist()
            for i, ch in enumerate(result['coherences']):
                result[f'coherence_{i}'] = float(ch)
            result['avg_coherence'] = np.mean(chr)
            assert chr.shape == (rewards_1.shape[-1]-1,), f"Coherence shape: {result['coherence'].shape}, Expected shape: {(rewards_1.shape[-1]-1,)}"
            print("EVAL METRICS:", result)

            training_variables.record_metrics(result, metric_type='validation')
            #input("...")
            return result

    #overriden
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        
        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                raise NotImplementedError("Sagemaker model parallelism is not currently supported for MORewardTrainer.")
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean(dim=0) 
            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            
            loss_single = self._gradients(loss=loss, **kwargs)

            return loss_single.detach()
        
    def _gradients(self, loss: th.Tensor, **kwargs):
        # Compute gradients for grounding and value system losses separately

        learning_rate = kwargs.get("learning_rate")
        
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss /= self.accelerator.gradient_accumulation_steps
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            raise NotImplementedError("DeepSpeed is not currently supported for MORewardTrainer.")
            self.deepspeed_engine_wrapped.backward(loss, sync_gradients=self.sync_gradients, **kwargs)
        elif self.accelerator.distributed_type == DistributedType.MEGATRON_LM:
            raise NotImplementedError("Megatron-LM is not currently supported for MORewardTrainer.")
            return
        elif self.accelerator.scaler is not None:

            loss = self.accelerator.scaler.scale(loss)
        elif learning_rate is not None and self.has_lomo_optimizer:
            raise NotImplementedError("LOMO optimizers are not currently supported for MORewardTrainer.")
            self.accelerator.lomo_backward(loss, learning_rate)
        
        l=loss.shape[0]
        assert l % 2 == 1, f"Expected odd number of rewards in loss tensor (num_values, num_values, 1), got {l}. Make sure your data collator is correctly collating pairs of samples with their labels."
        if l == self.model.num_values*2 +1: 
            loss_gr = loss[0:l//2]
            loss_gr_ideal = loss[l//2:l-1]
            assert len(loss_gr) == len(loss_gr_ideal), f"Grounding loss and ideal grounding loss must have the same number of samples. Got {len(loss_gr)} and {len(loss_gr_ideal)}."
        
        else:
            assert l == self.model.num_values + 1, f"Expected loss tensor to have shape (num_values + 1,), got {loss.shape}. Make sure your model is returning a loss tensor of shape (num_values + 1,) where the first num_values entries correspond to the grounding loss and the last entry corresponds to the value system loss."
            loss_gr = loss[0:self.model.num_values]
            loss_gr_ideal = None
        
        loss_vs = loss[-1]

        optimizer = self.optimizer
        if (isinstance(optimizer, AcceleratedOptimizer) and isinstance(optimizer.optimizer, ConstrainedOptimizer)):
            constrained_optim = optimizer.optimizer
        elif isinstance(optimizer, ConstrainedOptimizer):
            constrained_optim = optimizer
        else:
            raise ValueError("Optimizer must be an instance of ConstrainedOptimizer or AcceleratedOptimizer wrapping a ConstrainedOptimizer. Unregistered optimizer type: {}".format(type(optimizer)))
        

        loss_combined = constrained_optim.custom_backward(loss_gr, loss_gr_ideal, loss_vs)
        
        return loss_combined
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss")
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raise NotImplementedError("Sagemaker model parallelism is not currently supported for MORewardTrainer.")
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                        )
                    if len(loss.shape) > 1:
                        loss = loss.detach().mean(dim=0) #CHANGED FOR MULTILABEL LOSS!
                    else:
                        loss = loss.detach()
                    assert loss.shape == (5,),  f"Expected loss to be a scalar tensor, got {loss.shape}. Make sure your model is returning a scalar loss value for evaluation."

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return (loss, logits, labels)
