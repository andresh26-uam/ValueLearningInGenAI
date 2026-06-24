import enum
from typing import Any, Dict, NamedTuple, Optional
import numpy as np
import torch as th
from torch.optim.optimizer import Optimizer as Optimizer

from transformers.trainer import *

from transformers.optimization import get_scheduler

from transformers.trainer_utils import SchedulerType, TrainOutput, _is_peft_model
from vsllib.reward_models import AbstractCtxDependentAlignmentLayer, MORMForClassification, MORMForSequenceClassification, MORMForClassificationConfig, accuracy_logits, accuracy_logits_smooth, rewards_and_labels_to_logits_and_targets
from vsllib.training_utils import ConstrainedLRScheduler, ConstrainedOptimizer, MORMTrainingVariables


from accelerate.optimizer import AcceleratedOptimizer
from accelerate import Accelerator
from vsllib.utils import to_float
from vsllib.defines import MIN_EPSILON


from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk




class EvalPredictionWithExtraLabels(EvalPrediction):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*): Input data passed to the model.
        losses (`np.ndarray`, *optional*): Loss values computed during evaluation.
    """

    def __init__(
        self,
        predictions: np.ndarray | tuple[np.ndarray],
        label_ids: np.ndarray | tuple[np.ndarray],
        #labels_ql: np.ndarray | tuple[np.ndarray],
        #labels_qt: np.ndarray | tuple[np.ndarray],
        inputs: np.ndarray | tuple[np.ndarray] | None = None,
        losses: np.ndarray | tuple[np.ndarray] | None = None,
        others: Dict[str,np.ndarray]={}
    ):
        super().__init__(predictions=predictions, label_ids=label_ids, inputs=inputs, losses=losses)
        self.labels_ql = others.pop("target_probs_quantitative")
        self.labels_qt = others.pop("target_probs_qualitative")
        self.others = others
        self.elements = (*self.elements, self.labels_ql, self.labels_qt)


class EvalLoopOutputWithExtraLabels(NamedTuple):
    predictions: np.ndarray | tuple[np.ndarray]
    label_ids: np.ndarray | tuple[np.ndarray] | None
    metrics: dict[str, float] | None
    num_samples: int | None
    others: dict | None
    

    @property
    def labels_qt(self):
        return self.others["target_probs_quantitative"]

    @property
    def labels_ql(self):
        return self.others["target_probs_qualitative"]

    #labels_qt: np.ndarray | tuple[np.ndarray] | None
    #labels_ql: np.ndarray | tuple[np.ndarray] | None


class MORewardTrainer(Trainer):
    training_variables: MORMTrainingVariables
    model: MORMForSequenceClassification
    accelerator: Accelerator
    keys_to_save_in_prediction=["target_probs_quantitative", "target_probs_qualitative"]

    def __init__(self, **kwargs: Any) -> None:
        args = kwargs.get("args")
        if "loss" not in args.include_for_metrics:
            args.include_for_metrics.append("loss")
        kwargs["args"] = args
        super().__init__(**kwargs)
        self.model.loss_function = self.compute_loss_func

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        This overrides the original logging process to include new train metrics.
        """
        is_eval_log = any(k.startswith("eval_") for k in logs.keys())
        if self.model.training and not is_eval_log:

            train_metrics = self.model.training_variables._collect_train_metrics_for_logging()

            if self.model.value_system_layer is not None:
                w = self.model.value_system_layer.get_value_system_info()

                for i in range(self.model.num_values):
                    train_metrics[f"vs_weight_{i}"] = to_float(w[i])
            if train_metrics:
                for key, value in train_metrics.items():
                    # Train/ is put by default
                    logs.setdefault(f"{key}", value)
        
        return super().log(logs, start_time)

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
            raise ValueError(
                "Optimizer must be an instance of ConstrainedOptimizer or AcceleratedOptimizer wrapping a ConstrainedOptimizer. Unregistered optimizer type: {}".format(type(optimizer)))
            return super().create_scheduler(num_training_steps, optimizer)
        scheduler_name = SchedulerType(self.args.lr_scheduler_type)
        warmup_steps = self.args.get_warmup_steps(num_training_steps)

        sched_x = None
        if constrained_optim.optimx is not None:
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

    def compute_metrics_custom(eval_pred: EvalPredictionWithExtraLabels, config: MORMForClassificationConfig, training_variables: MORMTrainingVariables) -> Dict[str, float]:
        with th.no_grad():
            epsilon_list = set([0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5])
            epsilon_list.add(config.discordance_epsilon)

            result = {}
            logits_shortened = eval_pred.predictions
            labels_shortened = eval_pred.label_ids
            labels_quantitative = eval_pred.labels_qt
            labels_qualitative = eval_pred.labels_ql


            loss_all = eval_pred.losses
            losses = np.mean(loss_all, axis=0)
            loss_vs = losses[-1]
            loss_gr = losses[0:-1]

            result['grounding_loss'] = loss_gr.tolist()
            for i in range(len(loss_gr)):
                result[f'grounding_loss_{i}'] = to_float(loss_gr[i])
            result['value_system_loss'] = to_float(loss_vs)

            represent = accuracy_logits(
                logits_shortened[..., -1], labels_shortened[..., -1], assume_torch=False, discordance_epsilon=config.discordance_epsilon)
            result['representativeness'] = represent

            represent_usual = accuracy_logits(
                logits_shortened[..., -1], labels_shortened[..., -1], assume_torch=False, discordance_epsilon=config.discordance_epsilon, hard_classification=False)
            result['representativeness_usual'] = represent_usual

            represent_smooth = accuracy_logits_smooth(
                logits_shortened[..., -1], labels_quantitative[..., -1], assume_torch=False)
            result['representativeness_smooth'] = represent_smooth

            for epsilon in epsilon_list:
                represent = accuracy_logits(
                    logits_shortened[..., -1], labels_shortened[..., -1], assume_torch=False, discordance_epsilon=epsilon)
                result[f'representativeness_e{epsilon}'] = represent

            chr = accuracy_logits(
                logits_shortened[..., 0:-1], labels_shortened[..., 0:-1], assume_torch=False, discordance_epsilon=config.discordance_epsilon)
            coherences = chr.tolist()
            chr_usual = accuracy_logits(
                logits_shortened[..., 0:-1], labels_shortened[..., 0:-1], assume_torch=False, discordance_epsilon=config.discordance_epsilon, hard_classification=False)
            coherences_usual = chr_usual.tolist()

            chr_smooth = accuracy_logits_smooth(
                logits_shortened[..., 0:-1], labels_quantitative[..., 0:-1], assume_torch=False)
            coherences_smooth = chr_smooth.tolist()

            per_epsilon_chr = []
            for epsilon in epsilon_list:
                cohr = accuracy_logits(
                    logits_shortened[..., 0:-1], labels_shortened[..., 0:-1], assume_torch=False, discordance_epsilon=epsilon)
                per_epsilon_chr.append(cohr)

            result["coherences"] = coherences
            for i, ch in enumerate(coherences):
                result[f'coherence_{i}'] = float(ch)
                for j, epsilon in enumerate(epsilon_list):
                    result[f'coherence_e{epsilon}_{i}'] = float(per_epsilon_chr[j][i])
                

            for i, ch in enumerate(coherences_smooth):
                result[f'coherence_smooth_{i}'] = float(ch)
            for i, ch in enumerate(coherences_usual):
                result[f'coherence_usual_{i}'] = float(ch)

            result['avg_coherence'] = np.mean(coherences)
            result['avg_coherence_smooth'] = np.mean(coherences_smooth)
            result['avg_coherence_usual'] = np.mean(coherences_usual)
            for j, epsilon in enumerate(epsilon_list):
                result[f'avg_coherence_e{epsilon}'] = np.mean([float(per_epsilon_chr[j][i]) for i in range(len(coherences))])
            assert chr.shape == (
                logits_shortened.shape[-1]-1,), f"Coherence shape: {coherences.shape}, Expected shape: {(logits_shortened.shape[-1]-1,)}"

            training_variables.record_metrics(result, metric_type='validation')
            training_variables.record_grounding_loss(gr_loss_detached=th.tensor(loss_gr, requires_grad=False, device=training_variables.lagrange_multipliers.device, dtype=training_variables.lagrange_multipliers.dtype) if loss_gr is not None else None, 
                                                     vs_loss_detached=th.tensor(loss_vs, requires_grad=False, device=training_variables.lagrange_multipliers.device, dtype=training_variables.lagrange_multipliers.dtype), gr_loss_ideal_detached=None, loss_type="validation")
            
            return result

    # overriden
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
        epoch=None,
    ) -> torch.Tensor:
        """
        Taken from the library. It has changes to handle multiple losses.
        """
        # Prepare buffers for context parallelism
        cp_context, inputs = self._prepare_context_parallel_inputs(
            model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                raise NotImplementedError(
                    "Sagemaker model parallelism is not currently supported for MORewardTrainer.")
                loss_mb = smp_forward_backward(
                    model, inputs, self.args.gradient_accumulation_steps)

                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():

                loss = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch,epoch=epoch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}

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

            loss_single = self._gradients(loss=loss, epoch = epoch, **kwargs)

            return loss_single.detach()

    def _run_epoch(
        self,
        model,
        epoch,
        train_dataloader,
        steps_in_epoch,
        num_update_steps_per_epoch,
        trial,
        ignore_keys_for_eval,
        start_time,
        resume_from_checkpoint,
        epochs_trained,
        steps_trained_in_current_epoch,
    ):
        """Run one full pass over the dataloader."""

        step = -1
        grad_norm = None
        learning_rate = None
        rng_to_sync = False

        # Handle resumption from checkpoint: skip already-trained batches in the resumed epoch
        num_update_steps_trained = 0
        if epoch == epochs_trained and resume_from_checkpoint is not None:
            if steps_trained_in_current_epoch > 0 and not self.args.ignore_data_skip:
                train_dataloader = skip_first_batches(train_dataloader, steps_trained_in_current_epoch)
                step = steps_trained_in_current_epoch - 1
                num_update_steps_trained = steps_trained_in_current_epoch // self.args.gradient_accumulation_steps
                rng_to_sync = True
            elif steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        epoch_iterator = iter(train_dataloader)

        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % self.args.gradient_accumulation_steps
        if remainder == 0:
            remainder = self.args.gradient_accumulation_steps

        # Outer loop: one iteration per optimizer step. Each iteration prefetches
        # `gradient_accumulation_steps` batches (fewer for the last step if the epoch
        # doesn't divide evenly).
        for update_step in range(num_update_steps_trained, num_update_steps_per_epoch):
            num_batches = (
                self.args.gradient_accumulation_steps if update_step != (num_update_steps_per_epoch - 1) else remainder
            )
            batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, self.args.device)

            # This is used to correctly scale the loss when the last accumulation step has fewer batches.
            # Not used if `num_items_in_batch` is not None.
            self.current_gradient_accumulation_steps = len(batch_samples)

            # need to sync after if we skipped the batches in `get_batch_samples` for shuffle order reason
            if rng_to_sync:
                self._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False

            # Inner loop: forward + backward for each micro-batch. Gradients are
            # accumulated without syncing until the last micro-batch, then we clip,
            # step the optimizer, and log/save/evaluate.
            for i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if step % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # We sync the gradients in the following cases: 1. sync_each_batch set to True 2. Using deepspeed 3. when we are at the last batch sample
                if (
                    self.accelerator.gradient_state.plugin_kwargs.get("sync_each_batch", False)
                    or self.accelerator.distributed_type == DistributedType.DEEPSPEED
                    or i == len(batch_samples) - 1
                ):
                    sync_context = contextlib.nullcontext
                else:
                    sync_context = functools.partial(self.accelerator.no_sync, model=model)
                with sync_context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch, epoch=epoch)

                if (
                    self.args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    self._tr_loss += self._tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if self._tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {self._tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    self._tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                self._track_num_input_tokens(inputs)

                if do_sync_step:
                    grad_norm = None
                    if self.args.max_grad_norm > 0:
                        grad_norm = self._clip_grad_norm(model)
                        
                    grad_norm = self._get_grad_norm(model, grad_norm=grad_norm)
                    
                    self.control = self.callback_handler.on_pre_optimizer_step(self.args, self.state, self.control)
                    self.optimizer.step()
                    self.control = self.callback_handler.on_optimizer_step(self.args, self.state, self.control)

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    self._maybe_log_save_evaluate(
                        self._tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(self.args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        # PyTorch/XLA relies on the dataloader to insert mark_step each iteration.
        # When we break out of the loop early, we flush the pending graph manually.
        if is_torch_xla_available():
            xm.mark_step()

        if step < 0:
            logger.warning(
                "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({self.state.max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
        self._maybe_log_save_evaluate(
            self._tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )


    def _gradients(self, loss: th.Tensor, epoch: int, **kwargs):
        # Compute gradients for grounding and value system losses separately
        # Taken from accelerate.backward.

        learning_rate = kwargs.get("learning_rate")

        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss /= self.accelerator.gradient_accumulation_steps
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            raise NotImplementedError(
                "DeepSpeed is not currently supported for MORewardTrainer.")
            self.deepspeed_engine_wrapped.backward(
                loss, sync_gradients=self.sync_gradients, **kwargs)
        elif self.accelerator.distributed_type == DistributedType.MEGATRON_LM:
            raise NotImplementedError(
                "Megatron-LM is not currently supported for MORewardTrainer.")
            return
        elif self.accelerator.scaler is not None:

            loss = self.accelerator.scaler.scale(loss)
        elif learning_rate is not None and self.has_lomo_optimizer:
            raise NotImplementedError(
                "LOMO optimizers are not currently supported for MORewardTrainer.")
            self.accelerator.lomo_backward(loss, learning_rate)

        l = loss.shape[0]
        assert l > 1
        if l == self.model.num_values*2 + 1:
            loss_gr = loss[0:l//2]
            loss_gr_ideal = loss[l//2:l-1]
            assert len(loss_gr) == len(
                loss_gr_ideal), f"Grounding loss and ideal grounding loss must have the same number of samples. Got {len(loss_gr)} and {len(loss_gr_ideal)}."

        else:
            assert l == self.model.num_values + \
                1, f"Expected loss tensor to have shape (num_values + 1,), got {loss.shape}. Make sure your model is returning a loss tensor of shape (num_values + 1,) where the first num_values entries correspond to the grounding loss and the last entry corresponds to the value system loss."
            loss_gr = loss[0:self.model.num_values]
            loss_gr_ideal = None

        loss_vs = loss[-1]

        optimizer = self.optimizer
        if (isinstance(optimizer, AcceleratedOptimizer) and isinstance(optimizer.optimizer, ConstrainedOptimizer)):
            constrained_optim = optimizer.optimizer
        elif isinstance(optimizer, ConstrainedOptimizer):
            constrained_optim = optimizer
        else:
            raise ValueError(
                "Optimizer must be an instance of ConstrainedOptimizer or AcceleratedOptimizer wrapping a ConstrainedOptimizer. Unregistered optimizer type: {}".format(type(optimizer)))

        loss_combined = constrained_optim.custom_backward(
            loss_gr, loss_gr_ideal, loss_vs, epoch=epoch, **kwargs)
        
        return loss_combined

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
        epoch="EVAL",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Taken from the library. It has changes to handle multiple losses. Some implementations may raise errors as they were not tested
        """
        has_labels = False if len(self.label_names) == 0 else all(
            inputs.get(k) is not None for k in self.label_names)
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
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name)
                                   for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raise NotImplementedError(
                    "Sagemaker is not currently supported for MORewardTrainer.")
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items(
                        ) if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        num_items_in_batch = self._get_num_items_in_batch(
                            [inputs], self.args.device)
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch,epoch=epoch
                        )
                    if len(loss.shape) > 1:
                        loss = loss.detach().mean(dim=0)  # CHANGED FOR MULTILABEL LOSS!
                    else:
                        loss = loss.detach()
                    assert len(
                        loss.shape) == 1,  f"Expected loss to be a 1d vector, got {loss.shape}"
                    
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items()
                                       if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items()
                                       if k not in ignore_keys)
                    else:
                        logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        logits, labels, others = rewards_and_labels_to_logits_and_targets(
            logits, labels, config=self.model.config, assume_torch=True)

        return (loss, logits, labels, {k: others[k] for k in self.keys_to_save_in_prediction})

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
        epoch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, torch.Tensor | Any]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If not passed, the loss is computed
                using the default batch size reduction logic.

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculation might be slightly inaccurate when performing gradient accumulation.
        """
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled and self.model.training:
            return deepspeed_sp_compute_loss(self.accelerator, model, inputs, return_outputs, pc)

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        outputs = model(**inputs)
        

        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            if labels is None:
                logger.warning(
                    "Trainer: `compute_loss_func` is defined but `labels=None`. "
                    "Your custom loss function will still be called with labels=None. "
                )
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,epoch=epoch
            )
        # Default HF loss handling (label smoothing) if no custom loss function
        elif labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = (
                unwrapped_model.base_model.model._get_name()
                if _is_peft_model(unwrapped_model)
                else unwrapped_model._get_name()
            )
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutputWithExtraLabels:
        """
        taken from the library. It has changes to handle multiple losses.
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
        all_losses = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100)
        
        all_others = dict()
        for extra_key in self.keys_to_save_in_prediction:
            all_others[extra_key] = EvalLoopContainer(
                self.args.eval_do_concat_batches, padding_index=-100)
            
        all_inputs = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100)

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
            losses, logits, labels, others = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys, epoch="EVAL")
            main_input_name = getattr(
                self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(
                    inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                # This is the only change in this function.
                if len(losses.shape) == 0:
                    losses = self.gather_function(losses.repeat(batch_size))
                elif len(losses.shape) == 1:
                    losses = self.gather_function(
                        losses.unsqueeze_(0).repeat(batch_size, 1))
                    if __debug__ and not self.accelerator.gradient_state.end_of_dataloader:
                        assert losses.shape[0] % batch_size == 0, f"Expected losses to have shape ({batch_size} times number of GPUs (or smaller),) after gather, got {losses.shape}. Make sure your model is returning a loss tensor of shape (batch_size,) for evaluation."
                    assert losses.shape[1] == self.model.num_values + \
                        1, f"Expected losses to have shape ({batch_size} times number of GPUs (or smaller), {self.model.num_values + 1}) after gather, got {losses.shape}. Make sure your model is returning a loss tensor of shape (batch_size, {self.model.num_values + 1}) for evaluation where the first num_values entries correspond to the grounding loss and the last entry corresponds to the value system loss."

                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100)
                
            for extra_key in self.keys_to_save_in_prediction:
                if others.get(extra_key, None) is not None:
                    others[extra_key] = self.accelerator.pad_across_processes(
                    others[extra_key], dim=1, pad_index=-100)
            """if labels_ql is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels_ql = self.accelerator.pad_across_processes(
                    labels_ql, dim=1, pad_index=-100)
            if labels_qt is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels_qt = self.accelerator.pad_across_processes(
                    labels_qt, dim=1, pad_index=-100)"""
            if logits is not None:
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            for extra_key in self.keys_to_save_in_prediction:
                evalue =others.get(extra_key, None)
                if evalue is not None:
                    others[extra_key] = self.gather_function(evalue)
                    if not self.args.batch_eval_metrics or description == "Prediction":
                        all_others[extra_key].add(others[extra_key])
            """if labels_ql is not None:
                labels_ql = self.gather_function(labels_ql)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels_ql.add(labels_ql)
            if labels_qt is not None:
                labels_qt = self.gather_function(labels_qt)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels_qt.add(labels_qt)"""


            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        #EvalPredictionWithExtraLabels(predictions=logits,label_ids=labels, labels_ql=labels_ql, labels_qt=labels_qt, **batch_kwargs)
                        EvalPredictionWithExtraLabels(predictions=logits,
                                                       label_ids=labels, others=others, **batch_kwargs),

                        compute_result=is_last_step,
                    )

                del losses, logits, labels, labels_ql, labels_qt, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                for extra_key in self.keys_to_save_in_prediction:
                    all_others[extra_key].to_cpu_and_numpy()
                #all_labels_ql.to_cpu_and_numpy()
                #all_labels_qt.to_cpu_and_numpy()

                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, others, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        # print("LIBRARY ALL LOSSES", all_losses.shape)
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()

        for extra_key in self.keys_to_save_in_prediction:
            all_others[extra_key] = all_others[extra_key].get_arrays()
        #all_labels_ql = all_labels_ql.get_arrays()
        #all_labels_qt = all_labels_qt.get_arrays()
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
                #EvalPredictionWithExtraLabels(predictions=all_preds,label_ids=all_labels, labels_ql=all_labels_ql, labels_qt=all_labels_qt, **eval_set_kwargs)
                EvalPredictionWithExtraLabels(predictions=all_preds,
                                                       label_ids=all_labels, others=all_others, **eval_set_kwargs),
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(
                all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        #return EvalLoopOutputWithExtraLabels(predictions=all_preds, label_ids=all_labels, labels_ql=all_labels_ql, labels_qt=all_labels_qt, metrics=metrics, num_samples=num_samples)
        return EvalLoopOutputWithExtraLabels(predictions=all_preds, label_ids=all_labels, others=all_others, metrics=metrics, num_samples=num_samples)

    def save_with_seed(self, checkpoint_name: str = "last_checkpoint"):
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save_state()  # saved in the self.args.output_dir.
        #  save inside the checkpoint_name directory inside the output_dir.
        self.save_model(checkpoint_dir)

        # tokenizer.save_pretrained(checkpoint_dir)

        seed_info = {
            "seed": self.args.seed,
            "dataseed": self.args.data_seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "torch_initial_seed": int(th.initial_seed()),
        }
        with open(os.path.join(checkpoint_dir, "seed_info.json"), "w", encoding="utf-8") as fp:
            json.dump(seed_info, fp, indent=2, sort_keys=True)
        return checkpoint_dir


class CtxMORewardTrainer(MORewardTrainer):
    training_variables: MORMTrainingVariables
    model: MORMForClassification
    accelerator: Accelerator
    keys_to_save_in_prediction=["target_probs_quantitative", "target_probs_qualitative", "ctx"]
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        This overrides the original logging process to include new train metrics.
        """
        is_eval_log = any(k.startswith("eval_") for k in logs.keys())
        if self.model.training and not is_eval_log:

            train_metrics = self.model.training_variables._collect_train_metrics_for_logging()

            if self.model.value_system_layer is not None:
                assert isinstance(self.model.value_system_layer, AbstractCtxDependentAlignmentLayer)
                """contexts_to_vc = self.get_context_ids_mapped_to_value_system(vi)
            vs_tuple = transform_weights_to_tuple(vc)
            data={
                "contexts": contexts_to_vc,
                "share_of_data": float(th.sum(self.running_context_training_data.frequencies[contexts_to_vc]).numpy())}
            for iv, v in enumerate(vs_tuple):
                data[f"vs_w{iv}"] = v
            per_vs_contexts[f"vs_{vi}"] = data"""
                
                w_info = self.model.value_system_layer.get_value_system_info()
                
                """for i in range(self.model.num_values):
                    train_metrics[f"vs_weight_{i}"] = to_float(w[i])"""
                for vs_key, vs_data in w_info.items():
                    train_metrics[vs_key] = dict()
                    train_metrics[vs_key]["ncontexts"] = len(vs_data["contexts"])
                    train_metrics[vs_key]["share"] = vs_data["share_of_data"]
                    for k,v in vs_data.items():
                        if "vs_w" in k:
                            train_metrics[vs_key][k] = v # Weights of this VS.
            if train_metrics:
                for key, value in train_metrics.items():
                    # Train/ is put by default
                    logs.setdefault(f"{key}", value)
        
        return Trainer.log(self, logs, start_time)


    def train_initialization(self) -> None:
        if self.model.config.training_initialization_data_size != "all":
            training_initialization_data_size = min(len(self.train_dataset), self.model.config.training_initialization_data_size)
            indices_ = np.random.choice(len(self.train_dataset), size=training_initialization_data_size, replace=False)
            subset = self.train_dataset.select(indices_)
        else:
            subset = self.train_dataset
        self.model.train_initialization(subset)
    
    def train(self, resume_from_checkpoint: str | bool | None = None, trial: Any | Dict[str, Any] | None = None, ignore_keys_for_eval: list[str] | None = None) -> TrainOutput:
        
        if resume_from_checkpoint is None and self.accelerator.is_main_process:
            self.train_initialization()

        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval)

