from typing import Any, Dict, NamedTuple, Optional
import json
import re
from importlib import import_module
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats as scipy_stats
from wordcloud import WordCloud
import torch as th
from torch.optim.optimizer import Optimizer as Optimizer

from transformers.trainer import *

from transformers.optimization import get_scheduler

from transformers.trainer_utils import SchedulerType, TrainOutput, _is_peft_model
from transformers.trainer_utils import SchedulerType, TrainOutput, _is_peft_model
from kNLPmeans.summaryCentroids import build_sentence_corpus, embed_sentences, summarize_textrank
from vsllib.reward_models import AbstractCtxDependentAlignmentLayer, CtxData, MORMForClassification, MORMForSequenceClassification, rewards_and_labels_to_logits_and_targets
from vsllib.model_utils import CustomVAE, MORMForClassificationConfig, accuracy_logits, accuracy_logits_smooth
from vsllib.training_utils import ConstrainedLRScheduler, ConstrainedOptimizer, MORMTrainingVariables


from accelerate.optimizer import AcceleratedOptimizer
from accelerate import Accelerator
from vsllib.utils import auto_tsne, flatten_metrics_for_csv, kmeans_clustering, plot_alternative_clusterings, to_float
from vsllib.defines import LLM_MODEL_EVAL, ContextImplementations


from datasets import Dataset




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
        self.labels_ql = others.get("target_probs_qualitative")
        self.labels_qt = others.get("target_probs_quantitative")
        self.others = others
        self.elements = (*self.elements, self.labels_ql, self.labels_qt)


class EvalLoopOutputWithExtraLabels(NamedTuple):
    predictions: np.ndarray | tuple[np.ndarray]
    label_ids: np.ndarray | tuple[np.ndarray] | None
    metrics: dict[str, float] | None
    num_samples: int | None
    others: dict | None
    

    @property
    def labels_qt(self) -> th.Tensor | np.ndarray | tuple[np.ndarray] | None:
        return self.others["target_probs_quantitative"]

    @property
    def labels_ql(self) -> th.Tensor | np.ndarray | tuple[np.ndarray] | None:
        return self.others["target_probs_qualitative"]

    #labels_qt: np.ndarray | tuple[np.ndarray] | None
    #labels_ql: np.ndarray | tuple[np.ndarray] | None


def _pairwise_accuracy_metrics(logits_shortened, labels_shortened, labels_quantitative, config: MORMForClassificationConfig) -> Dict[str, float]:
    """Grounding/value-system accuracy metrics (representativeness*, coherence_{i}*, avg_coherence*)
    for a set of preference pairs. Shared by `compute_metrics_custom` (whole split) and
    `compute_metrics_per_cluster` (one call per cluster subset) so both stay in sync.
    """
    with th.no_grad():
        epsilon_list = set([0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5])
        epsilon_list.add(config.discordance_epsilon)

        result = {}

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

        return result


def compute_metrics_per_cluster(logits_shortened, labels_shortened, labels_quantitative, cluster_ids, config: MORMForClassificationConfig) -> Dict[Any, Dict[str, float]]:
    """Applies `_pairwise_accuracy_metrics` separately to each cluster's subset of pairs.

    `cluster_ids` must be a per-pair array (one label per row of `logits_shortened`/
    `labels_shortened`), i.e. already deduplicated from the interleaved (chosen, rejected)
    per-sample order the same way `CtxMORewardTrainer.remove_duplicates` does.
    """
    if isinstance(cluster_ids, th.Tensor):
        cluster_ids = cluster_ids.detach().cpu().numpy()
    cluster_ids = np.asarray(cluster_ids)
    results = {}
    for c in sorted(np.unique(cluster_ids).tolist()):
        mask = cluster_ids == c
        metrics = _pairwise_accuracy_metrics(
            logits_shortened[mask], labels_shortened[mask],
            labels_quantitative[mask] if labels_quantitative is not None else None, config)
        metrics.pop("coherences", None)
        metrics["size"] = int(mask.sum())
        results[c] = metrics
    return results


def _to_numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson/Spearman correlation, returning NaN instead of raising on degenerate
    input (fewer than 2 samples, or a constant array -- both make correlation undefined).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    pearson_r = float(scipy_stats.pearsonr(x, y)[0])
    spearman_r = float(scipy_stats.spearmanr(x, y)[0])
    return pearson_r, spearman_r


def _length_correlation_metrics(groundings: np.ndarray, value_system_reward: np.ndarray, response_lengths: np.ndarray) -> Dict[str, float]:
    """Pearson/Spearman correlation between response length (in tokens) and (a) each
    value's predicted grounding reward, (b) the model's predicted overall value-system
    reward, over a set of individual responses (one row per response, not per pair --
    `groundings`/`value_system_reward`/`response_lengths` must all be in the same
    per-sample order).
    """
    result = {}
    num_values = groundings.shape[1]
    for i in range(num_values):
        pearson_r, spearman_r = _safe_corr(response_lengths, groundings[:, i])
        result[f"length_pearson_{i}"] = pearson_r
        result[f"length_spearman_{i}"] = spearman_r
    pearson_vs, spearman_vs = _safe_corr(response_lengths, value_system_reward)
    result["length_pearson_vs"] = pearson_vs
    result["length_spearman_vs"] = spearman_vs
    return result


def compute_length_correlations_per_cluster(groundings: np.ndarray, value_system_reward: np.ndarray, response_lengths: np.ndarray, cluster_ids) -> Dict[Any, Dict[str, float]]:
    """Applies `_length_correlation_metrics` separately to each cluster's subset of
    individual responses. `cluster_ids` must be a per-sample array, aligned one-per-row
    with `groundings`/`value_system_reward`/`response_lengths` (unlike the per-pair
    `cluster_ids` used by `compute_metrics_per_cluster` -- repeat a per-pair cluster
    label array twice, e.g. `np.repeat(pair_cluster_ids, 2)`, to get this).
    """
    if isinstance(cluster_ids, th.Tensor):
        cluster_ids = cluster_ids.detach().cpu().numpy()
    cluster_ids = np.asarray(cluster_ids)
    results = {}
    for c in sorted(np.unique(cluster_ids).tolist()):
        mask = cluster_ids == c
        results[c] = _length_correlation_metrics(groundings[mask], value_system_reward[mask], response_lengths[mask])
    return results


def _coherence_and_representativeness_title_lines(cm: Dict[str, float]) -> list:
    """Coherence_{i}/representativeness lines for a wordcloud/bar-plot title, in value
    order (no per-value labels -- just the values, in the same order as the value-system
    weights), and the length-vs-reward Pearson/Spearman correlation lines that go with
    them, if present.
    """
    lines = []
    coherence_keys = sorted(
        (k for k in cm if re.fullmatch(r"coherence_\d+", k)),
        key=lambda k: int(k.split("_")[1]))
    if coherence_keys:
        lines.append("coherence: [" + ", ".join(f"{cm[k]:.3f}" for k in coherence_keys) + "]")
    if "representativeness" in cm:
        lines.append(f"representativeness: {cm['representativeness']:.3f}")

    pearson_keys = sorted(
        (k for k in cm if re.fullmatch(r"length_pearson_\d+", k)),
        key=lambda k: int(k.rsplit("_", 1)[-1]))
    spearman_keys = sorted(
        (k for k in cm if re.fullmatch(r"length_spearman_\d+", k)),
        key=lambda k: int(k.rsplit("_", 1)[-1]))
    if pearson_keys:
        lines.append("length-reward pearson: [" + ", ".join(f"{cm[k]:.3f}" for k in pearson_keys) + "]")
    if spearman_keys:
        lines.append("length-reward spearman: [" + ", ".join(f"{cm[k]:.3f}" for k in spearman_keys) + "]")
    if "length_pearson_vs" in cm or "length_spearman_vs" in cm:
        pr_vs = cm.get("length_pearson_vs", float("nan"))
        sr_vs = cm.get("length_spearman_vs", float("nan"))
        lines.append(f"length-VS reward pearson: {pr_vs:.3f}, spearman: {sr_vs:.3f}")
    return lines


class MORewardTrainer(Trainer):
    training_variables: MORMTrainingVariables
    model: MORMForSequenceClassification
    accelerator: Accelerator
    keys_to_save_in_prediction=["target_probs_quantitative", "target_probs_qualitative", "groundings"]

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
        sched_z = None
        if constrained_optim.optimz is not None:
            sched_z = get_scheduler(
                name=scheduler_name,
                optimizer=constrained_optim.optimz,
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
            sched_z=sched_z,
            sched_lambda=sched_lambda,
        )
        print("Optimizer and schedulers created successfully.")

        return self.lr_scheduler

    def compute_metrics_custom(eval_pred: EvalPredictionWithExtraLabels, config: MORMForClassificationConfig, training_variables: MORMTrainingVariables) -> Dict[str, float]:
        with th.no_grad():
            logits_shortened = eval_pred.predictions
            labels_shortened = eval_pred.label_ids
            labels_quantitative = eval_pred.labels_qt
            labels_qualitative = eval_pred.labels_ql

            loss_all = eval_pred.losses
            losses = np.mean(loss_all, axis=0)
            loss_vs = losses[-1]
            loss_gr = losses[0:-1]

            result = {}
            result['grounding_loss'] = loss_gr.tolist()
            for i in range(len(loss_gr)):
                result[f'grounding_loss_{i}'] = to_float(loss_gr[i])
            result['value_system_loss'] = to_float(loss_vs)

            result.update(_pairwise_accuracy_metrics(logits_shortened, labels_shortened, labels_quantitative, config))

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
        epoch: int,
        train_dataloader: DataLoader,
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
                    if extra_key == "groundings":
                        others[extra_key] = {
                            "values": others[extra_key],
                            "dummy": th.tensor(0.0, device=others[extra_key].device),
                        }
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
                evalue = others.get(extra_key, None)
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
        if isinstance(all_others.get("groundings"), dict):
            all_others["groundings"] = all_others["groundings"]["values"]
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
    keys_to_save_in_prediction=["target_probs_quantitative", "target_probs_qualitative", "groundings", "ctx"]
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        This overrides the original logging process to include new train metrics.
        """
        is_eval_log = any(k.startswith("eval_") for k in logs.keys())

        if self.model.training and not is_eval_log:
            if self.state.global_step % 1000 == 0:
                path = os.path.join(self.args.output_dir, "images")
                os.makedirs(path, exist_ok=True)
                if ContextImplementations(self.model.config.context_implementation) in [ContextImplementations.GMM, ]:
                    self.model.plot_matrices(t=self.state.global_step, filename=os.path.join(path,  f"context_matrices_{self.state.global_step}"), low_res=True)
                    
            train_metrics = self.model.training_variables._collect_train_metrics_for_logging()

            if self.model.value_system_layer is not None:
                assert isinstance(self.model.value_system_layer, AbstractCtxDependentAlignmentLayer)
                
                w_info = self.model.value_system_layer.get_value_system_info()
                
                for vs_key, vs_data in w_info.items():
                    if str(vs_key).startswith("vs"):
                        train_metrics[vs_key] = dict()
                        train_metrics[vs_key]["ncontexts"] = len(vs_data["contexts"])
                        train_metrics[vs_key]["share_ctx"] = vs_data["share_of_ctxdata"]
                        train_metrics[vs_key]["share_vs"] = vs_data["share_of_vsdata"]
                        for k,v in vs_data.items():
                            if "vs_w" in k:
                                train_metrics[vs_key][k] = v # Weights of this VS.

                    else:
                        train_metrics[vs_key] = float(vs_data)
            if train_metrics:
                for key, value in train_metrics.items():
                    # Train/ is put by default
                    logs.setdefault(f"{key}", value)
        
        return Trainer.log(self, logs, start_time)

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics. 

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset` | dict[str, `Dataset`], *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        #print(output.num_samples, output.others["ctx"]["context_features"].shape)
        #print("..." ,[output.others[k].shape for k in self.keys_to_save_in_prediction if k != "ctx"])
        #exit()

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        output.others["pairwise_predictions"] = output.predictions
        output.others["pairwise_labels"] = output.label_ids
        output.metrics["others"] = output.others # Just changed this.  

        return output.metrics 
    

    def train_initialization(self) -> None:

        
        if self.model.config.training_initialization_data_size != "all":
            training_initialization_data_size = min(len(self.train_dataset), self.model.config.training_initialization_data_size)
            indices_ = np.random.choice(len(self.train_dataset), size=training_initialization_data_size, replace=False)
            subset = self.train_dataset.select(indices_)
        else:
            subset = self.train_dataset
        self.model.train_initialization(subset, eval_set=self.eval_dataset, args=self.args, total_dataset_size=len(self.train_dataset))

    @staticmethod
    def plot_cluster_word_clouds(texts, labels, output_path: str, clustering_name: str, vs_predicted=None, label_names=None, descriptions=None, cluster_metrics=None) -> None:
        texts = np.asarray(texts, dtype=object)
        if isinstance(labels, th.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and {clustering_name} labels must have the same length, "
                f"got {len(texts)} and {len(labels)}"
            )
        if vs_predicted is not None:
            if isinstance(vs_predicted, th.Tensor):
                vs_predicted = vs_predicted.detach().cpu().numpy()
            vs_predicted = np.asarray(vs_predicted)
            if len(vs_predicted) != len(labels):
                raise ValueError(
                    f"vs_predicted and {clustering_name} labels must have the same length, "
                    f"got {len(vs_predicted)} and {len(labels)}"
                )

        panels = []
        for cluster_label in np.unique(labels):
            cluster_mask = labels == cluster_label
            cluster_texts = [str(text) for text in texts[cluster_mask] if str(text).strip()]
            if cluster_texts:
                average_vs = None
                if vs_predicted is not None:
                    average_vs = np.mean(vs_predicted[cluster_mask], axis=0)
                panels.append((cluster_label, len(cluster_texts), " ".join(cluster_texts), average_vs))

        panels.sort(key=lambda panel: -panel[1])

        if not panels:
            return

        columns = min(4, len(panels))
        rows = (len(panels) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5 * columns, 4.5 * rows),
            squeeze=False,
        )
        for axis, (cluster_label, cluster_size, text, average_vs) in zip(axes.flat, panels):
            word_cloud = WordCloud(
                width=800,
                height=600,
                background_color="white",
                random_state=0,
            ).generate(text)
            axis.imshow(word_cloud, interpolation="bilinear")
            axis.axis("off")
            category = label_names.get(cluster_label, f"Cluster {cluster_label}") if label_names else f"Cluster {cluster_label}"
            title = f"{clustering_name}, {category} (n={cluster_size})"
            if average_vs is not None:
                average_vs_text = ", ".join(f"{weight:.3f}" for weight in np.ravel(average_vs))
                title += f"\nmean predicted VS: [{average_vs_text}]"
            if cluster_metrics is not None and cluster_label in cluster_metrics:
                for line in _coherence_and_representativeness_title_lines(cluster_metrics[cluster_label]):
                    title += f"\n{line}"
            """THIS IS TOO MUCH INFO... if descriptions is not None and cluster_label in descriptions:
                title += f"\n{descriptions[cluster_label]}"
            """
            axis.set_title(title)
        for axis in axes.flat[len(panels):]:
            axis.axis("off")
        figure.tight_layout(pad=1)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def plot_cluster_metrics_bars(cluster_metrics: Dict[Any, Dict[str, float]], value_names, output_path: str, clustering_name: str) -> None:
        """Bar-plot grid (one subplot per cluster) of per-value grounding accuracy
        (`coherence_{i}`) and value-system accuracy (`representativeness`) for a single
        clustering (as produced by `compute_metrics_per_cluster`).
        """
        value_names = list(value_names)
        panels = sorted(cluster_metrics.items(), key=lambda kv: -kv[1].get("size", 0))
        if not panels:
            return

        bar_labels = value_names + ["representativeness"]
        colors = ["tab:blue"] * len(value_names) + ["tab:orange"]

        columns = min(4, len(panels))
        rows = (len(panels) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4 * rows), squeeze=False)
        for axis, (cluster_label, metrics) in zip(axes.flat, panels):
            values = [metrics.get(f"coherence_{i}", np.nan) for i in range(len(value_names))] + [metrics.get("representativeness", np.nan)]
            bars = axis.bar(range(len(bar_labels)), values, color=colors)
            axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=2, fontsize=8)
            axis.set_xticks(range(len(bar_labels)))
            axis.set_xticklabels(bar_labels, rotation=45, ha="right")
            axis.set_ylim(0, 1.08)
            axis.set_ylabel("accuracy")
            label_text = f"Cluster {cluster_label}" if isinstance(cluster_label, (int, np.integer)) else str(cluster_label)
            title = f"{clustering_name}, {label_text} (n={metrics.get('size', 0)})"
            for line in _coherence_and_representativeness_title_lines(metrics):
                if not line.startswith("coherence:") and not line.startswith("representativeness:"):
                    title += f"\n{line}"
            axis.set_title(title)
        for axis in axes.flat[len(panels):]:
            axis.axis("off")
        figure.tight_layout(pad=1)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def save_cluster_metrics(cluster_metrics_by_clustering: Dict[str, Dict[Any, Dict[str, float]]], output_dir: str) -> pd.DataFrame:
        """Writes every accuracy metric from `compute_metrics_per_cluster`, for every
        clustering, to `{output_dir}/cluster_metrics.csv` and `.json` -- one row per
        (clustering, cluster) pair.
        """
        rows = []
        for clustering_name, per_cluster in cluster_metrics_by_clustering.items():
            for cluster_label, metrics in per_cluster.items():
                row = {"clustering": clustering_name, "cluster": cluster_label}
                row.update(flatten_metrics_for_csv(metrics))
                rows.append(row)
        descriptions = pd.DataFrame(rows)
        os.makedirs(output_dir, exist_ok=True)
        descriptions.to_csv(os.path.join(output_dir, "cluster_metrics.csv"), index=False)
        descriptions.to_json(os.path.join(output_dir, "cluster_metrics.json"), orient="records", indent=2)
        return descriptions

    @staticmethod
    def describe_clusters_with_llm(texts, label_sets, clustering_names, output_dir: str, model_name: str = LLM_MODEL_EVAL, max_documents: int = 20, cluster_metrics: Optional[Dict[str, Dict[Any, Dict[str, float]]]] = None):
        """Generate category and description metadata for each text clustering."""
        try:
            ChatGroq = import_module("langchain_groq").ChatGroq
        except ImportError as error:
            raise ImportError("Install langchain-groq to generate cluster descriptions with Groq.") from error

        texts = np.asarray(texts, dtype=object)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set; cannot generate cluster descriptions.")
        llm = ChatGroq(model=model_name, temperature=0, api_key=groq_api_key)
        rows = []
        category_maps = []
        for labels, clustering_name in zip(label_sets, clustering_names):
            labels = np.asarray(labels)
            category_map = {}
            for cluster_label in sorted(np.unique(labels), key=lambda label: (-np.sum(labels == label), str(label))):
                cluster_texts = [str(text) for text in texts[labels == cluster_label] if str(text).strip()]
                sample = cluster_texts[:max_documents]
                if not sample:
                    category = f"Cluster {cluster_label}"
                    description = "No text was available for this cluster."
                else:
                    prompt = (
                        "Analyze the following documents from one cluster.\n"
                        "Return exactly two lines:\n"
                        "CATEGORY: a concise descriptive label of at most six words\n"
                        "DESCRIPTION: one concise sentence describing the common themes\n\n"
                        + "\n---\n".join(sample)
                    )
                    response = llm.invoke(prompt)
                    response_text = getattr(response, "content", str(response)).strip()
                    parsed = {}
                    for line in response_text.splitlines():
                        key, separator, value = line.partition(":")
                        if separator:
                            parsed[key.strip().upper()] = value.strip()
                    category = parsed.get("CATEGORY", response_text.splitlines()[0]).strip()
                    description = parsed.get("DESCRIPTION", response_text).strip()

                category_map[cluster_label] = category
                textrank_summary = CtxMORewardTrainer.summarize_cluster_with_textrank(cluster_texts)

                row = {
                    "clustering": clustering_name,
                    "cluster": cluster_label,
                    "category": category,
                    "description": description,
                    "summary_textrank": textrank_summary,
                    "size": len(cluster_texts),
                }
                cm = (cluster_metrics or {}).get(clustering_name, {}).get(cluster_label, {})
                for key, value in cm.items():
                    if key == "representativeness" or re.fullmatch(r"coherence_\d+", key):
                        row[key] = value
                rows.append(row)
            category_maps.append(category_map)

        descriptions = pd.DataFrame(rows)
        os.makedirs(output_dir, exist_ok=True)
        descriptions.to_csv(os.path.join(output_dir, "cluster_descriptions.csv"), index=False)
        descriptions.to_json(os.path.join(output_dir, "cluster_descriptions.json"), orient="records", indent=2)
        return descriptions, category_maps

    @staticmethod
    def summarize_cluster_with_textrank(cluster_texts, top_k: int = 5, emb_type: str = "all-MiniLM-L6-v2") -> str:
        if not cluster_texts:
            return "No text was available for this cluster."

        sentences, _ = build_sentence_corpus(cluster_texts)
        if not sentences:
            return "No sentence was available for this cluster."

        sentence_embeddings, _ = embed_sentences(sentences, emb_type=emb_type)
        summary = summarize_textrank(sentences, sentence_embeddings, top_k=top_k)
        return " ".join(sentence for sentence, _ in summary)
    
    def evaluate_contexts(self, eval_dataset, test_dataset, train_dataset, validation_output: Dict = None, test_output: Dict =None, output_dir: str = "", reducer_kwargs: dict = {}, value_names=None, eval_response_lengths=None, test_response_lengths=None) -> None:
        
        train_set_contexts = np.array(train_dataset.select_columns([self.model.vs_features_name])[self.model.vs_features_name])
        train_set_contexts = self.remove_duplicates(train_set_contexts)

        kmeans = kmeans_clustering(train_set_contexts, K= self.model.config.max_contexts)
        # --- Dimensionality reduction ---

        assert CtxData.from_dict(validation_output["ctx"], to_tensor=True).context_features.shape[0] == len(eval_dataset)*2, f"Validation dataset size {len(eval_dataset)} does not match validation output size {CtxData.from_dict(validation_output['ctx'], to_tensor=True).context_features.shape[0]}"
        validation_data = self.remove_duplicates(CtxData.from_dict(validation_output["ctx"], to_tensor=True).context_features)
        test_data = self.remove_duplicates(CtxData.from_dict(test_output["ctx"], to_tensor=True).context_features)
    
        X = train_set_contexts
        X_EVAL_TEST = np.concatenate([train_set_contexts, validation_data, test_data], axis=0)
        assert X_EVAL_TEST.shape == (len(train_set_contexts) + len(validation_data) + len(test_data), train_set_contexts.shape[1])
        needs_reduction = X.shape[1] > 2
        
        
        reducer_pca = None
        output = {"validation": None if validation_output is None else {}, "test": None if test_output is None else {}}
        for otype, output_per_type, context_data, response_lengths in zip(
            ("validation", "test",), (validation_output, test_output), (validation_data, test_data),
            (eval_response_lengths, test_response_lengths),
        ):
            if output_per_type is not None:
                ctxdata: CtxData = CtxData.from_dict(output_per_type["ctx"], to_tensor=True)
                stats = self.model.value_system_layer.calculate_statistics(ctxdata)
                
                #dataset_ctxs = np.array(val_dataset.select_columns([self.model.vs_features_name])[self.model.vs_features_name])
                original_context = np.array(eval_dataset.select_columns(["context"])["context"] if otype == "validation" else test_dataset.select_columns(["context"])["context"])
                
                features = self.remove_duplicates(context_data)
                assert len(features) == len(original_context), f"Got: {len(features)} and {len(original_context)}"

            
                print("OG", original_context[0:5], original_context.shape)
                print("FEAT", features[0:5], features.shape)
                
                kmeans_labels = kmeans.predict(features)
                kmeans_clusters = kmeans.cluster_centers_
                labels_1 = kmeans_labels
                
                labels_2 = self.remove_duplicates(ctxdata.vs_assignments)

                label1_name = f"Kmeans K={len(np.unique(np.array(labels_1)))}/{self.model.config.max_contexts}"
                label2_name = f"Value Systems {self.model.config.context_implementation} K={len(np.unique(np.array(labels_2)))}/{self.model.config.max_value_systems}"
                labels = [labels_1, labels_2]
                labels_set_names = [label1_name, label2_name]
                if ContextImplementations(self.model.config.context_implementation) in [ContextImplementations.GMM, ContextImplementations.GMM_AND_CLASSIFIER, ContextImplementations.VAE_AND_KMEANS]:
                     labels_3 = self.remove_duplicates(ctxdata.ctx_assignments)
                     label3_name = f"Contexts {self.model.config.context_implementation} K={len(np.unique(np.array(labels_2)))}/{self.model.config.max_value_systems}"              
                     labels.append(labels_3)
                     labels_set_names.append(label3_name)

                category_maps = [{label: f"Cluster {label}" for label in np.unique(labels_1)},
                                 {label: f"Cluster {label}" for label in np.unique(labels_2)}]
                if len(labels) == 3:
                    category_maps.append({label: f"Cluster {label}" for label in np.unique(labels_3)})

                clustering_names_used = ["kmeans", "value_system", "context"][:len(labels)]

                # Per-cluster grounding/value-system accuracy: replicates the accuracy
                # portion of compute_metrics_custom (representativeness*, coherence_{i}*,
                # avg_coherence*) separately for each cluster of each clustering, without
                # touching training_variables. Requires the raw pairwise predictions/labels
                # that CtxMORewardTrainer.evaluate() stashes into the "ctx"-sibling "others"
                # dict (pairwise_predictions/pairwise_labels), aligned pair-for-pair with
                # labels_1/labels_2/labels_3 above (all deduplicated the same way).
                pairwise_predictions = output_per_type.get("pairwise_predictions")
                pairwise_labels = output_per_type.get("pairwise_labels")
                pairwise_labels_quantitative = output_per_type.get("target_probs_quantitative")
                assert pairwise_predictions is not None and pairwise_labels is not None, (
                    f"{otype}: CtxMORewardTrainer.evaluate() did not expose "
                    "pairwise_predictions/pairwise_labels in its \"others\" output -- "
                    "per-cluster accuracy metrics cannot be computed."
                )

                # Per-sample (interleaved chosen/rejected) grounding rewards and the
                # model's predicted overall value-system reward for that same sample,
                # used for the response-length-vs-reward correlation below. groundings is
                # already per-sample; vs_predicted (the predicted value-system weights) is
                # also per-sample here (only `remove_duplicates`d elsewhere in this method
                # when a per-*pair* quantity is needed) -- both come from the same "ctx"
                # others dict, so they're aligned row-for-row by construction.
                groundings_per_sample = _to_numpy(output_per_type.get("groundings"))
                vs_predicted_per_sample = _to_numpy(ctxdata.vs_predicted)
                assert groundings_per_sample is not None and vs_predicted_per_sample is not None, (
                    f"{otype}: groundings/vs_predicted are required to compute the "
                    "value-system reward for the length-correlation analysis."
                )
                value_system_reward_per_sample = np.sum(groundings_per_sample * vs_predicted_per_sample, axis=1)
                assert response_lengths is not None, (
                    f"{otype}: response_lengths was not provided -- required for the "
                    "length-vs-reward correlation metrics. Pass eval_response_lengths/"
                    "test_response_lengths (computed via compute_response_token_lengths, "
                    "which itself requires an nlp_based tokenizer and response1/response2 "
                    "dataset columns -- re-run preprocessing with --repostprocess if missing)."
                )
                assert len(response_lengths) == len(groundings_per_sample), (
                    f"response_lengths ({len(response_lengths)}) and groundings "
                    f"({len(groundings_per_sample)}) must be aligned one-per-sample."
                )

                cluster_metrics_by_clustering = {}
                for label_array, clustering_name in zip(labels, clustering_names_used):
                    assert len(label_array) == len(pairwise_predictions), (
                        f"{clustering_name} labels ({len(label_array)}) and pairwise predictions "
                        f"({len(pairwise_predictions)}) must be aligned one-per-pair."
                    )
                    cluster_metrics_by_clustering[clustering_name] = compute_metrics_per_cluster(
                        pairwise_predictions, pairwise_labels, pairwise_labels_quantitative,
                        label_array, self.model.config,
                    )
                    # label_array is per-pair; repeat each pair's label across its two
                    # samples (chosen, rejected always share a cluster -- both derive
                    # from the same prompt's context) to mask per-sample data.
                    sample_label_array = np.repeat(_to_numpy(label_array), 2)
                    length_metrics = compute_length_correlations_per_cluster(
                        groundings_per_sample, value_system_reward_per_sample, response_lengths,
                        sample_label_array,
                    )
                    for cluster_label, metrics in cluster_metrics_by_clustering[clustering_name].items():
                        assert cluster_label in length_metrics, (
                            f"{clustering_name} cluster {cluster_label}: no length-correlation "
                            "metrics were computed for this cluster."
                        )
                        metrics.update(length_metrics[cluster_label])
                    self.plot_cluster_metrics_bars(
                        cluster_metrics_by_clustering[clustering_name],
                        value_names if value_names is not None else [],
                        os.path.join(output_dir, f"{otype}_{clustering_name}_cluster_metrics_bars.png"),
                        clustering_name,
                    )

                # Same accuracy + length-correlation metrics, but over the whole split
                # at once (no clustering) -- one extra "overall" bar-plot panel per
                # split, in addition to (not instead of) the per-cluster ones above.
                n_pairs = len(pairwise_predictions)
                overall_metrics = _pairwise_accuracy_metrics(
                    pairwise_predictions, pairwise_labels, pairwise_labels_quantitative, self.model.config)
                overall_metrics.pop("coherences", None)
                overall_metrics["size"] = n_pairs
                overall_metrics.update(_length_correlation_metrics(
                    groundings_per_sample, value_system_reward_per_sample, response_lengths))
                cluster_metrics_by_clustering["overall"] = {"all": overall_metrics}
                self.plot_cluster_metrics_bars(
                    cluster_metrics_by_clustering["overall"],
                    value_names if value_names is not None else [],
                    os.path.join(output_dir, f"{otype}_overall_cluster_metrics_bars.png"),
                    "overall",
                )

                self.save_cluster_metrics(
                    cluster_metrics_by_clustering,
                    os.path.join(output_dir, f"{otype}_cluster_metrics"),
                )

                if isinstance(original_context[0], str):
                    try:
                        descriptions, category_maps = self.describe_clusters_with_llm(
                            original_context,
                            labels,
                            clustering_names_used,
                            os.path.join(output_dir, f"{otype}_cluster_descriptions"),
                            cluster_metrics=cluster_metrics_by_clustering,
                        )
                    except (ImportError, RuntimeError) as error:
                        print(f"Could not generate LLM cluster descriptions: {error}")
                        descriptions = None
                    description_text_maps = {name: {} for name in clustering_names_used}
                    if descriptions is not None:
                        for clustering_name in clustering_names_used:
                            subset = descriptions[descriptions["clustering"] == clustering_name]
                            description_text_maps[clustering_name] = dict(zip(subset["cluster"], subset["description"]))
                    self.plot_cluster_word_clouds(
                        texts=original_context,
                        labels=labels_1,
                        output_path=os.path.join(output_dir, f"{otype}_kmeans_word_clouds.png"),
                        clustering_name="kmeans",
                        vs_predicted=self.remove_duplicates(ctxdata.vs_predicted),
                        label_names=category_maps[0],
                        descriptions=description_text_maps.get("kmeans"),
                        cluster_metrics=cluster_metrics_by_clustering.get("kmeans"),
                    )
                    self.plot_cluster_word_clouds(
                        texts=original_context,
                        labels=labels_2,
                        output_path=os.path.join(output_dir, f"{otype}_value_system_word_clouds.png"),
                        clustering_name="value_system",
                        vs_predicted=self.remove_duplicates(ctxdata.vs_predicted),
                        label_names=category_maps[1],
                        descriptions=description_text_maps.get("value_system"),
                        cluster_metrics=cluster_metrics_by_clustering.get("value_system"),
                    )
                    if len(labels) == 3:
                        self.plot_cluster_word_clouds(
                            texts=original_context,
                            labels=labels_3,
                            output_path=os.path.join(output_dir, f"{otype}_context_word_clouds.png"),
                            clustering_name="context",
                            vs_predicted=self.remove_duplicates(ctxdata.vs_predicted),
                            label_names=category_maps[2],
                            descriptions=description_text_maps.get("context"),
                            cluster_metrics=cluster_metrics_by_clustering.get("context"),
                        )
                if needs_reduction and reducer_pca is None:
                    reducer_pca: PCA = PCA(n_components=2, svd_solver= "full", whiten= True,)
                    reducer_pca.fit(X)
                    reduction_tsne, reducer_tsne, best_perp, best_metric = auto_tsne(X_EVAL_TEST, **reducer_kwargs)
                
                plot_alternative_clusterings(reducer_pca.transform(features) if needs_reduction else features, labels, 
                                             dim_reduction="pca", 
                                             #reduction_kwargs={"svd_solver": "full", "whiten": True},
                                             label_set_names=labels_set_names,
                                             label_display_sets=category_maps,
                                             output_path=os.path.join(output_dir, f"{otype}_PCA_context_clustering.pdf"))

                if needs_reduction:
                    eval_reduction = reduction_tsne[len(train_set_contexts):len(train_set_contexts)+len(validation_data)]
                    test_reduction = reduction_tsne[len(train_set_contexts)+len(validation_data):]
                    assert len(test_data) == len(test_reduction)
                    assert len(validation_data) == len(eval_reduction)
                    if otype == "validation":
                        tsne_features = eval_reduction
                    else:
                        tsne_features = test_reduction
                else:
                    tsne_features = features
                    
                plot_alternative_clusterings(tsne_features if needs_reduction else features, 
                                             labels,
                                             dim_reduction=f"tsne_p{best_perp}",
                                             label_set_names=labels_set_names,
                                             label_display_sets=category_maps,
                                             output_path=os.path.join(output_dir, f"{otype}_TSNE_context_clustering.pdf"))

                # If using VAE_KMEANS, now plot the latent space, the centroids, and the vs assignments. 
                if ContextImplementations(self.model.config.context_implementation) in [ContextImplementations.VAE_AND_KMEANS]:
                    vae_or_ae = self.model.value_system_layer.context_logits
                    vae_or_ae: CustomVAE
                    vae_or_ae.plot_embedding_space(sample_data=features, sample_labels=self.remove_duplicates(ctxdata.ctx_assignments), sample_label_names=category_maps[2], original_space_centroids=th.as_tensor(kmeans_clusters, dtype=features.dtype, device=features.device), save_path=os.path.join(output_dir, f"{otype}_VAE_latent_space"), low_res=False)
                output[otype] = stats.to_dict()
                if ContextImplementations(self.model.config.context_implementation) not in [ContextImplementations.DIRECT_VS]:
                    self.model.plot_matrices(t=0, filename=os.path.join(output_dir, f"{otype}_context_matrices"), low_res=False, ctx_data=ctxdata)
        return output

    def remove_duplicates(self, train_set_contexts):
        if np.allclose(train_set_contexts[::2], train_set_contexts[1::2]) and len(train_set_contexts) % 2 == 0:
            bsz = train_set_contexts.size(0)                
            jidx = th.arange(0, bsz, 2, device=train_set_contexts.device)
            kidx = jidx + 1
            train_set_contexts = train_set_contexts[jidx]
        return train_set_contexts

    
    

    def train(self, resume_from_checkpoint: str | bool | None = None, trial: Any | Dict[str, Any] | None = None, ignore_keys_for_eval: list[str] | None = None) -> TrainOutput:
        if self.model.config.do_initialization:
            if resume_from_checkpoint is None and self.accelerator.is_main_process:
                self.train_initialization()

        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval)
