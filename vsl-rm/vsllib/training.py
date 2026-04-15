from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
from uuid import uuid4
from datasets.arrow_dataset import Dataset
from matplotlib.pylab import dtype
import numpy as np
import torch as th
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer as Optimizer

from transformers.trainer import *

from transformers.optimization import get_scheduler

from transformers.trainer_utils import SchedulerType
from ordered_set import OrderedSet
from vsllib.defines import NO_RATING_MASK
from vsllib.reward_models import MOLossFunctions, MOLossFunctionsCategories, MORMForSequenceClassification, MORMForSequenceClassificationConfig, accuracy_logits, accuracy_rewards_labels, get_missing_rating_mask, logits_BT, parse_loss_function, rewards_and_labels_to_logits_and_targets, scores_to_target_probs
from vsllib.utils import MORMTrainingVariables, MORewardDataCollatorWithPadding, print_tensor_and_grad_fn, to_float

from accelerate.utils import recursively_apply

    



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
                 config: MORMForSequenceClassificationConfig = None,
                 sub_optimizer_class=th.optim.Adam, **optimizer_kwargs):
        
        if config.loss_func_type in MOLossFunctionsCategories.NEEDS_NO_GRAD_ON_VALUE_SYSTEM_WEIGHTS:
            lr_value_system = 0.0
        if config.loss_func_type in MOLossFunctionsCategories.NEEDS_NO_GRAD_ON_ALL_GROUNDINGS:
            lr_grounding = 0.0

        super(ConstrainedOptimizer, self).__init__(params_gr=params_gr, params_vs=params_vs, n_values=n_values,
                                                   lr_grounding=lr_grounding, lr_value_system=lr_value_system, sub_optimizer_class=sub_optimizer_class, **optimizer_kwargs)
        if params_gr_ideal is not None:
            assert len(params_gr) == len(params_gr_ideal), "Grounding parameters and ideal grounding parameters must have the same length."
            
            self.params_gr_ideal = params_gr_ideal
            self.optimx_ideal = _create_sub_optimizer(params_gr_ideal, lr_grounding, self.sub_optimizer_class, optimizer_kwargs)

        self.lr_lambda = lr_lambda if lr_lambda is not None else lr_value_system * 10.0
        if config.loss_func_type in MOLossFunctionsCategories.NEEDS_NO_GRAD_ON_LAGRANGE_MULTIPLIERS or (len(config.loss_func_type_kwargs.get('value_indices')) == 1 and config.loss_func_type == MOLossFunctions.ONLY_VALUES_IN_KWARGS):
            self.lr_lambda = 0.0

        self.loss_func_type = config.loss_func_type
        self.loss_func_kwargs=config.loss_func_type_kwargs

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
        if len(w) > 0:
            assert w[0] is self.optimy.param_groups[0]['params'][0], "Value system parameters do not match those in the optimizer"
            assert w[0].requires_grad, "Value system parameters must require gradients for stoic optimization."
        if len(x) > 0:
            assert x[0] is self.optimx.param_groups[0]['params'][0], "Grounding parameters do not match those in the optimizer"
            #assert x[0].requires_grad, "Grounding parameters must require gradients for stoic optimization."
        # PARAMS", self.optimx.param_groups[0]['params'][0].data[0:10])

        if loss_gr_ideal is not None:
            self.optimx_ideal.zero_grad()
            loss_sum = th.sum(loss_gr_ideal).detach()
            for i in range(len(loss_gr_ideal)):
                    (loss_gr_ideal[i]/loss_sum).backward(retain_graph=True)
                    self.optimx_ideal.step()
                    self.optimx_ideal.zero_grad()
        self.training_variables.requires_grad_(False)
        if self.lr_lambda > 0:
            assert self.optim_lambdas.param_groups[0]['params'][0] is self.training_variables.lagrange_multipliers, "Lagrange multipliers not found in optimizer parameters"    
        
        target_gr_loss = loss_gr_ideal.detach() if loss_gr_ideal is not None else None

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
            loss = self.training_variables.forward(loss_gr, None, target_gr_loss=target_gr_loss, selected_indices=self.loss_func_kwargs['value_indices'])
            unselected_indices = [i for i in range(len(loss_gr)) if i not in self.loss_func_kwargs['value_indices']]
            loss_gr[unselected_indices] = loss_gr[unselected_indices].detach()
            if len(self.loss_func_kwargs['value_indices']) > 1:
                self.training_variables.lagrange_multipliers[unselected_indices] = self.training_variables.lagrange_multipliers[unselected_indices].detach()
            else: 
                self.training_variables.lagrange_multipliers = self.training_variables.lagrange_multipliers.detach()
            #print("YEs", loss_func_kwargs['value_indices'])
            #input("?")

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
        
        print("LAGRANGE MULTIPLIERS BEFORE STEP:", self.training_variables.lagrange_multipliers)
        #print("TRAINING VARS", vars(self.training_variables))
        self.time += 1
        #th.nn.utils.clip_grad_norm_(self.params_gr, self.max_grad_norm)
        #th.nn.utils.clip_grad_norm_(self.params_vs, self.max_grad_norm)
        if self.lr_grounding > 0.0: 
            assert self.params_gr[0].grad is not None, "Grounding gradients have not been computed. Make sure to call the backward pass on the grounding loss before stepping the optimizer."
        
        if self.lr_value_system > 0.0:
            assert self.params_vs[0].grad is not None, "Value system gradients have not been computed. Make sure to call the backward pass on the value system loss before stepping the optimizer."
            #print("GRADIENTS BEFORE STEP - VALUE SYSTEM PARAMS:", [p.grad for p in self.params_vs])
        #print("GRADIENTS BEFORE STEP - GR PARAMS:", [p.grad for p in self.params_gr])
            
        if self.lr_grounding > 0.0:
            self.optimx.step()
        if self.optimy is not None and self.lr_value_system > 0.0:
                self.optimy.step()
        

        self.training_variables.prepare_for_optimizer_step()
        if self.lr_lambda > 0:
            self.optim_lambdas.step()
        print("LAGRANGE MULTIPLIERS AFTER STEP (BEFORE DECAY):", self.training_variables.lagrange_multipliers)
        self.training_variables.post_optimizer_step()

            
        #self.zero_grad()
        #print("GRADIENTS AFTER STEP - VALUE SYSTEM PARAMS:", [p.grad for p in self.params_vs])
        #print("GRADIENTS AFTER STEP - GR PARAMS:", [p.grad for p in self.params_gr])
        
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
from accelerate import Accelerator

class MORewardTrainer(Trainer):

    
    training_variables: MORMTrainingVariables
    model: MORMForSequenceClassification
    accelerator: Accelerator

    def __init__(self, **kwargs):
        args=kwargs.get("args")
        if "loss" not in args.include_for_metrics:
            args.include_for_metrics.append("loss")
        kwargs["args"] = args
        super().__init__(**kwargs)
        self.compute_loss_func = partial(self.compute_loss_func, accelerator=self.accelerator, training_variables=self.model.training_variables)
        self.model.loss_function = self.compute_loss_func

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        is_eval_log = any(k.startswith("eval_") for k in logs.keys())
        if self.model.training and not is_eval_log:
            train_metrics = self.model.training_variables._collect_train_metrics_for_logging()
            w = self.model.value_system_layer.get_weights()
            
            for i in range(self.model.num_values):
                train_metrics[f"vs_weight_{i}"] = to_float(w[i])
            if train_metrics:
                for key, value in train_metrics.items():
                    logs.setdefault(f"{key}", value) # Train/ is put by default
        #TODO THIS MIGHT NOT WORK
        return super().log(logs, start_time)

    """def create_optimizer(self, model: MORMForSequenceClassification =None) -> th.optim.Optimizer:
        self.optimizer = super().create_optimizer(model)
        return self.optimizer"""

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

        
    
    def compute_metrics(eval_pred, config: MORMForSequenceClassificationConfig, training_variables: MORMTrainingVariables) -> Dict[str, float]:
        with th.no_grad():
            result = {}
            
            #print("EVAL PREDICTIONS SHAPE:", eval_pred.predictions.shape)
            #print("EVAL LABELS SHAPE:", eval_pred.label_ids.shape)
            #print("EVAL KEYS", dir(eval_pred))
            
            # We assume that the first sample is preferred by default in groundtruth
            logits_shortened = eval_pred.predictions
            labels_shortened = eval_pred.label_ids

            #print("EVAL PREDICTIONS", logits_shortened)
            #print("EVAL LABELS:", labels_shortened)
            
            loss_all = eval_pred.losses
            losses = np.mean(loss_all, axis=0)
            loss_vs = losses[-1]
            loss_gr = losses[0:-1]

            result['grounding_loss'] = to_float(loss_gr)
            for i in range(len(loss_gr)):
                result[f'grounding_loss_{i}'] = to_float(loss_gr[i])    
            result['value_system_loss'] = to_float(loss_vs)

            represent = accuracy_logits(logits_shortened[..., -1], labels_shortened[..., -1], assume_torch=False)
            result['representativeness'] = represent

            chr = accuracy_logits(logits_shortened[..., 0:-1], labels_shortened[..., 0:-1], assume_torch=False)
            result['coherences'] = chr.tolist()
            for i, ch in enumerate(result['coherences']):
                result[f'coherence_{i}'] = float(ch)
            result['avg_coherence'] = np.mean(chr)
            assert chr.shape == (logits_shortened.shape[-1]-1,), f"Coherence shape: {result['coherence'].shape}, Expected shape: {(logits_shortened.shape[-1]-1,)}"
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
        assert l > 1
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
        #print("LOSS GR:", loss_gr)
        #print("LOSS GR IDEAL:", loss_gr_ideal)
        #print("LOSS VS:", loss_vs)
        #print_tensor_and_grad_fn(loss_vs.grad_fn)
        #exit(0)

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
            
        logits, labels, _ = rewards_and_labels_to_logits_and_targets(logits, labels, config=self.model.config, assume_torch=True)
        
        
        return (loss, logits, labels)
    

    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                # This is the only change in this function.
                if len(losses.shape) == 0:
                    losses = self.gather_function(losses.repeat(batch_size))
                elif len(losses.shape) == 1:
                    losses = self.gather_function(losses.unsqueeze_(0).repeat(batch_size, 1))
                    if not self.accelerator.gradient_state.end_of_dataloader:
                        assert losses.shape[0] == batch_size, f"Expected losses to have shape ({batch_size},) after gather, got {losses.shape}. Make sure your model is returning a loss tensor of shape (batch_size,) for evaluation."
                    assert losses.shape[1] == self.model.num_values + 1, f"Expected losses to have shape ({batch_size} (or smaller), {self.model.num_values + 1}) after gather, got {losses.shape}. Make sure your model is returning a loss tensor of shape (batch_size, {self.model.num_values + 1}) for evaluation where the first num_values entries correspond to the grounding loss and the last entry corresponds to the value system loss."
                
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        #print("LIBRARY ALL LOSSES", all_losses.shape)
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
