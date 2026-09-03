from collections.abc import Iterator
import math

from pythae.data import BaseDataset
from pythae.models.nn import BaseDecoder, BaseEncoder
import tqdm

from transformers.configuration_utils import PretrainedConfig
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from transformers import TrainingArguments

from pythae.models.vae import VAE, VAEConfig
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP

from pythae.models.base.base_utils import ModelOutput

from torch.distributions import Normal, Independent


from sklearn.cluster import KMeans

from pythae.data.datasets import BaseDataset
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline


from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
                
from vsllib.defines import MIN_EPSILON, NO_RATING_MASK, VALUE_LAYER_ACTIVATIONS, ContextImplementations, MOLossFunctions, MOLossManagement
from vsllib.utils import kmeans_clustering

THRESHOLD = 50.0
ACTIVATE_THRESHOLD_VS =  False
THRESHOLD_CTX = 50.0
TEMP_GMM = 1.0
ACTIVATE_TEMPERATURE_GMM = False

def gaussian_prob(
            z: th.Tensor,
            mu: th.Tensor,
            log_var: th.Tensor,
        ) -> th.Tensor:
            std = th.exp(0.5 * log_var)
            dist = Independent(Normal(mu, std), 1)
            return dist.log_prob(z).exp()

def compute_mutual_information(context_to_vs_logit_probs: th.Tensor):
    context_log_probs = th.log_softmax(
                context_to_vs_logit_probs, dim=1
            )
    context_probs = context_log_probs.exp()
    value_system_probs = context_probs.mean(dim=0)  # axis 1: value systems
    value_system_log_probs = value_system_probs.clamp_min(1e-30).log()
    value_system_entropy = -th.sum(
                value_system_probs * value_system_log_probs
            )
    conditional_value_system_entropy = -th.sum(
                context_probs * context_log_probs, dim=1
            ).mean()  # axis 0: contexts
    mutual_information = value_system_entropy - conditional_value_system_entropy
            
    return context_log_probs,context_probs,mutual_information

def compute_mutual_information_from_alternative_distributions(log_probs1: th.Tensor, log_probs2: th.Tensor):
    assert th.allclose(log_probs1.exp().sum(dim=-1), th.ones_like(log_probs1[..., 0])), f"Expected log_probs1 to sum to 1.0, but got {log_probs1.exp().sum(dim=-1)}"
    assert th.allclose(log_probs2.exp().sum(dim=-1), th.ones_like(log_probs2[..., 0])), f"Expected log_probs2 to sum to 1.0, but got {log_probs2.exp().sum(dim=-1)}"
    """
    log_probs1: [B, N1]
    log_probs2: [B, N2]

    Treats each batch element as an observation and estimates
    P(V1, V2) from the two distributions.
    """

    p1 = log_probs1.exp()
    p2 = log_probs2.exp()

    joint = th.einsum("bi,bj->ij", p1, p2) / p1.shape[0]

    marginal1 = joint.sum(dim=1, keepdim=True)
    marginal2 = joint.sum(dim=0, keepdim=True)

    # Only evaluate log where joint > 0.
    log_ratio = (
        th.log(joint.clamp_min(th.finfo(joint.dtype).tiny))
        - th.log(marginal1.clamp_min(th.finfo(joint.dtype).tiny))
        - th.log(marginal2.clamp_min(th.finfo(joint.dtype).tiny))
    )

    return (joint * log_ratio).sum()

def apply_discordance_epsilon_to_logits(missing_mask: th.Tensor, targets, bt: th.Tensor, activate_discordance_epsilon_for_loss: bool = True, discordance_epsilon: Optional[float] = 0.0):
        if activate_discordance_epsilon_for_loss and discordance_epsilon is not None:
                with th.no_grad():
                    discordance = th.full_like(bt, fill_value=0.0)
                    discordance = discordance.masked_fill(targets < 0.5, -discordance_epsilon)
                    discordance = discordance.masked_fill(targets > 0.5, discordance_epsilon)
                    if missing_mask is not None:
                        discordance = discordance.masked_fill(missing_mask, 0.0)
                return bt-discordance
        else:
            return bt
        

def accuracy_logits_smooth(logits: th.Tensor, target_probs: th.Tensor, missing_mask=None, assume_torch=True) -> th.Tensor:
    with th.no_grad():
        
        missing_mask = get_missing_rating_mask(
            target_probs) if missing_mask is None else missing_mask
        if missing_mask is not None:
            all_defined_cases = ~missing_mask
            logits_of_smoothing_equal_cases = logits[all_defined_cases]
            targets_of_smoothing_equal_cases = target_probs[all_defined_cases]
        else:
            all_defined_cases = True
            logits_of_smoothing_equal_cases = logits
            targets_of_smoothing_equal_cases = target_probs


        if assume_torch:
            f = th.nn.functional.sigmoid(logits_of_smoothing_equal_cases)
            score = 1.0-th.abs(f - targets_of_smoothing_equal_cases)
            # assert th.all(score <= 1.0) and th.all(score >= 0.0), f"Score values must be between 0 and 1.0, but got min {th.min(score)}, max {th.max(score)}"
        else:
            f = 1.0/(1.0+np.exp(-logits_of_smoothing_equal_cases))
            score = 1.0-np.abs(f - targets_of_smoothing_equal_cases)
            # assert np.all(score <= 1.0) and np.all(score >= 0.0), f"Score values must be between 0 and 1.0, but got min {np.min(score)}, max {np.max(score)}"

        if assume_torch:
            mask = all_defined_cases.float()
            factor = mask.sum(dim=0)
        else:
            mask = all_defined_cases.astype(float)
            factor = mask.sum(axis=0)
        mask[all_defined_cases] = score

        if assume_torch:
            positive_cases = mask.sum(dim=0)
        else:
            positive_cases = mask.sum(axis=0)

        return positive_cases/factor


def scores_to_target_probs(scores1: th.Tensor, scores2: th.Tensor, reward_diff_threshold: int = 50.0, assume_qualitative_labels=False, check_undefined_label=True, missing_mask=None, assume_torch=True) -> th.Tensor:

    with th.no_grad():

        if assume_qualitative_labels:
            # model probability of first one being preferred.
            mask_greater = scores1 > scores2
            mask_less = scores1 < scores2
            mask_equal = ~(mask_greater | mask_less)
            if assume_torch:
                target_probs = mask_greater.float()
            else:
                target_probs = mask_greater.astype(float)
            target_probs[mask_equal] = 0.5

            
        
        else:
            log = logits_BT(scores1, scores2, threshold=reward_diff_threshold,
                            check_undefined_label=check_undefined_label, missing_mask=missing_mask, assume_torch=assume_torch)
            if assume_torch:
                target_probs = th.sigmoid(log)
            else:
                target_probs = 1 / (1 + np.exp(-log))
        mask = True
        if check_undefined_label:
            # If either score is NO_RATING_MASK, set target_prob to 0.5 (indicating no preference)
            mask = get_missing_rating_mask(
                scores1, scores2) if missing_mask is None else missing_mask
            if assume_torch:
                target_probs.masked_fill_(mask, NO_RATING_MASK)
                #assert th.allclose(target_probs[mask_less], th.zeros_like(target_probs[mask_less]))
                #assert th.allclose(target_probs[mask_greater], th.ones_like(target_probs[mask_greater]))
                #assert th.allclose(target_probs[mask_equal], 0.5 * th.ones_like(target_probs[mask_equal]))
            else:
                target_probs[mask] = NO_RATING_MASK

    """if assume_qualitative_labels:
        assert np.allclose(target_probs[mask_less & mask], np.zeros_like(target_probs[mask_less & mask]))
        assert np.allclose(target_probs[mask_greater & mask], np.ones_like(target_probs[mask_greater & mask]))
        assert np.allclose(target_probs[mask_equal & mask], 0.5 * np.ones_like(target_probs[mask_equal & mask]))
    """
    return target_probs


def print_logits_target_mismatches(logits: th.Tensor, target_probs: th.Tensor, logits1: th.Tensor, target1: th.Tensor, all_defined_cases, assume_torch=True) -> None:
    """Debug helper: print where `logits1` differs from `target1` and show values.

    Kept as a standalone function so callers can enable/disable it easily.
    """
    try:
        if assume_torch:
            mismatch = (logits1 != target1) & all_defined_cases
            if mismatch.any():
                idx = mismatch.nonzero(as_tuple=False)
                if idx.numel() > 0:
                    if logits.ndim == 1:
                        rows = idx.squeeze(1)
                        vals_logits = logits[rows].cpu().numpy()
                        vals_target_probs = target_probs[rows].cpu().numpy()
                        vals_logits1 = logits1[rows].cpu().numpy()
                        vals_target1 = target1[rows].cpu().numpy()
                        print(f"accuracy_logits mismatch at indices: {rows.cpu().numpy()}")
                        print("logits:", vals_logits)
                        print("target_probs:", vals_target_probs)
                        print("logits1:", vals_logits1, "target1:", vals_target1)
                        input()
                    else:
                        rows = idx[:, 0]
                        cols = idx[:, 1]
                        vals_logits = logits[rows, cols].cpu().numpy()
                        vals_target_probs = target_probs[rows, cols].cpu().numpy()
                        vals_logits1 = logits1[rows, cols].cpu().numpy()
                        vals_target1 = target1[rows, cols].cpu().numpy()
                        print("accuracy_logits mismatches at (row,col):", list(zip(rows.cpu().numpy(), cols.cpu().numpy())))
                        print("logits:", vals_logits)
                        print("target_probs:", vals_target_probs)
                        print("logits1:", vals_logits1, "target1:", vals_target1)
                        input()
        else:
            # numpy branch
            mismatch = (logits1 != target1) & all_defined_cases
            if mismatch.any():
                idx = np.nonzero(mismatch)
                print("accuracy_logits mismatches at indices:", idx)
                print("logits values:", logits[idx])
                print("target_probs values:", target_probs[idx])
                print("logits1:", logits1[idx], "target1:", target1[idx])
                input()
    except Exception as e:
        # Do not raise from the debug helper; report and continue
        try:
            print("print_logits_target_mismatches failed:", e)
        except Exception:
            pass


def accuracy_logits(logits: th.Tensor, target_probs: th.Tensor, missing_mask=None, assume_torch=True, discordance_epsilon=MIN_EPSILON, hard_classification=True) -> th.Tensor:

    with th.no_grad():
        score_diff_epsilon = 1.0/(1.0+np.exp(-discordance_epsilon)) -0.5 if discordance_epsilon > 0 else 0.0
        discordance_epsilon = max(discordance_epsilon, 1e-5)

        missing_mask = get_missing_rating_mask(
            target_probs) if missing_mask is None else missing_mask
        if missing_mask is not None:
            all_defined_cases = ~missing_mask
        else:
            all_defined_cases = th.ones_like(target_probs, dtype=th.bool) if assume_torch else np.ones_like(target_probs, dtype=bool)
       
        # & (target_probs != NO_RATING_MASK))

        if hard_classification:
            logits1 = (logits > discordance_epsilon) 
            logits0 = (logits < -discordance_epsilon) 
            target1 = (target_probs > 0.5 + score_diff_epsilon) 
            target0 = (target_probs < 0.5 - score_diff_epsilon)

            mask1_1 = logits1 & target1
            mask1_2 = logits0 & target0
            
            equal_cases_logits = ~(logits1  | logits0) #if not hard_classification else (logits <= discordance_epsilon) & (logits >= -discordance_epsilon)
            equal_cases_targets = ~(target1  |target0) #if not hard_classification else (target_probs <= 0.5 + score_diff_epsilon) & (target_probs >= 0.5 - score_diff_epsilon)
            mask1_3 = equal_cases_logits & equal_cases_targets

            #print_logits_target_mismatches(logits, target_probs, equal_cases_logits, equal_cases_targets, all_defined_cases, assume_torch=assume_torch)
            mask05_1 = equal_cases_logits & ~equal_cases_targets
            mask05_2 = equal_cases_targets & ~equal_cases_logits

            mask05 = (mask05_1 | mask05_2) & all_defined_cases
            mask1 = (mask1_1 | mask1_2  | mask1_3) & all_defined_cases

        else:
            repr1 = (logits > 0) & (target_probs > 0.5)
            repr2 = (logits < 0) & (target_probs < 0.5)
            equal_targets = (target_probs >= 0.5 - score_diff_epsilon) & (target_probs <= 0.5 + score_diff_epsilon)
            equal_logits = (logits >= -discordance_epsilon) & (logits <= discordance_epsilon)
            repr3 = equal_targets & equal_logits
            mask1 = (repr1 | repr2 | repr3) & all_defined_cases
            mask05 = all_defined_cases & ((equal_targets & ~equal_logits) | (~equal_targets & equal_logits))

        if assume_torch:
            mask = mask1.float()
            mask[mask05 & ~mask1] = 0.5   
            factor = (all_defined_cases).float().sum(dim=0)
            positive_cases = mask.sum(dim=0)
        else:
            mask = mask1.astype(float)
            mask[mask05 & ~mask1] = 0.5   
            factor = (all_defined_cases).astype(float).sum(axis=0)
            positive_cases = mask.sum(axis=0)
        accuracy = positive_cases / factor

    if len(logits.shape) >= 2:
        assert accuracy.shape == (
            logits.shape[-1],), f"Expected loss shape {(logits.shape[-1],)}, got {accuracy.shape}"
    else:
        assert accuracy.shape == (
        ), f"Expected loss shape (), got {accuracy.shape}"
    return accuracy


def logits_BT(x: th.Tensor, y: th.Tensor, threshold=50.0, check_undefined_label=False, missing_mask=None, assume_torch=True) -> th.Tensor:
    
    returns_diff = x - y
    
    if check_undefined_label:
        if missing_mask is None:
            missing_mask = get_missing_rating_mask(x, y)
    else:
        if missing_mask is not None:
            raise ValueError(
                "check_undefined_label should be True to use missing_mask or get_missing_rating_mask")

    if assume_torch:
        returns_diff = th.clip(returns_diff, -threshold, threshold)
    else:
        returns_diff = np.clip(returns_diff, -threshold, threshold)
    if missing_mask is not None and check_undefined_label:
        if assume_torch:
            returns_diff.masked_fill_(missing_mask, NO_RATING_MASK)
        else:
            returns_diff[missing_mask] = NO_RATING_MASK
    """if assume_torch:
        assert th.max(returns_diff[~missing_mask]) <= threshold and th.min(returns_diff[~missing_mask]) >= - \
            threshold, f"Clipping failed: max {th.max(returns_diff[~missing_mask])}, min {th.min(returns_diff[~missing_mask])}, threshold {threshold}"
    """
    return returns_diff


def get_missing_rating_mask(x_or_probs: th.Tensor, y: th.Tensor =None) -> th.Tensor:
    if y is not None:
        missing_mask = (x_or_probs == NO_RATING_MASK) | (y == NO_RATING_MASK)
    else:
        missing_mask = (x_or_probs == NO_RATING_MASK)

    return missing_mask



def calculate_training_constants(args: TrainingArguments, total_dataset_size: int, len_dataset: int, epoch_multiplier=0.1) -> Tuple[tqdm.tqdm, int, int, int]:
        
            
        iterations_total = int((total_dataset_size//args.train_batch_size)*args.num_train_epochs*epoch_multiplier)
        print("Iterations total:", iterations_total)
            
                
        pbar = tqdm.tqdm(range(iterations_total))
        batch_size = min(args.train_batch_size, len_dataset)
        if batch_size <= 0:
            raise ValueError("No valid examples available for value-system pretraining")
        batches_per_epoch = (len_dataset + batch_size - 1) // batch_size
        return pbar,iterations_total,batch_size,batches_per_epoch

def random_argmax(x: th.Tensor, dim: int = -1) -> th.Tensor:
    #return th.argmax(x, dim) # TODO !!!!!!!
    max_val = x.amax(dim=dim, keepdim=True)
    mask = x == max_val

    # Random number only determines ordering among maxima.
    r = th.rand(x.shape, device=x.device, dtype=th.float32)
    r.masked_fill_(~mask, -1.0)

    return r.argmax(dim=dim)
t = th.tensor([[1.0, 2.0, 3.0, 3.0], [3.0, 2.0, 1.0, 3.0], [1.0, 3.0, 2.0, 3.0]])

def compute_per_centroid_variance(
    data: th.Tensor,
    centroids: th.Tensor,
    assignments: th.Tensor,
    min_var: float = 1e-6,
    return_log: bool = False
) -> th.Tensor:
    """Compute per-centroid variance from data and cluster assignments.
    
    Calculates the variance of data points within each cluster by computing
    the mean squared difference from the centroid for each dimension.
    
    Args:
        data: [B, D] - input data points
        centroids: [K, D] - cluster centroids
        assignments: [B] - hard cluster assignment for each point (values in 0..K-1)
        min_var: minimum variance to prevent log(0) issues
        return_log: if True, return log(variance); else return variance
        
    Returns:
        variance: [K, D] - per-centroid, per-dimension variance
                  or log(variance) if return_log=True
    """
    num_components = centroids.shape[0]
    input_size = centroids.shape[1]
    device = centroids.device
    dtype = centroids.dtype
    
    # Initialize variance accumulator
    variance = th.zeros(
        num_components,
        input_size,
        device=device,
        dtype=dtype
    )
    
    # Compute variance for each component
    for k in range(num_components):
        mask = assignments == k
        if mask.sum() > 0:
            # Get points assigned to this centroid
            points_k = data[mask]  # [n_k, D]
            centroid_k = centroids[k]  # [D]
            
            # Compute squared differences and take mean
            diff_squared = (points_k - centroid_k).pow(2)  # [n_k, D]
            variance[k] = diff_squared.mean(dim=0).clamp(min=min_var)
        else:
            # No points assigned, use minimum variance
            variance[k] = min_var
    
    if return_log:
        return th.log(variance)
    return variance


def construct_layers(input_dim: int, hidden_sizes: Tuple, intermediate_activation: str, dropout: float, device, dtype, n_outputs: int, final_activation: str, final_activation_kwargs: dict):
    layers=[]
    try:
        intermediate_activation = VALUE_LAYER_ACTIVATIONS[
            intermediate_activation]
    except KeyError:
        raise ValueError(
            f"Unsupported intermediate activation: {intermediate_activation}")

    if intermediate_activation is None:
        raise ValueError(
            f"Unsupported intermediate activation: {intermediate_activation}")
    final_size = input_dim
    input_aux = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(input_aux, hidden_size,
                      dtype=dtype, device=device))

        layers.append(intermediate_activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        final_size = hidden_size
        input_aux = hidden_size
    layers.append(nn.Linear(final_size, n_outputs,
                  dtype=dtype, device=device))

    try:
        final_activation = VALUE_LAYER_ACTIVATIONS[final_activation]
    except KeyError:
        raise ValueError(
            f"Unsupported final activation: {final_activation}")
    if final_activation is not None:
        layers.append(final_activation(**final_activation_kwargs))
    return layers

class FastGaussianMixture(nn.Module):

    def __init__(
        self,
        input_size: int,
        num_components: int,
        l_entropy = 1e-1,
        activate_threshold: bool = False,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.l_entropy = l_entropy
        self.input_size = input_size
        self.num_components = num_components
        self.activate_threshold = activate_threshold
        self.logits = nn.Parameter(
            th.zeros(num_components, device=device, dtype=dtype)
        )

        self.centroids = nn.Parameter(
            th.randn(
                num_components,
                input_size,
                device=device,
                dtype=dtype
            ) #* 1e-3
        )

        self.log_var = nn.Parameter(
            th.zeros(
                num_components,
                input_size,
                device=device,
                dtype=dtype
            )
        )


    def forward_all(self, x) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        # x: [B,D]

        diff = (
            x[:, None, :]
            -
            self.centroids[None, :, :]
        )
        # [B,K,D]


        inv_var = th.exp(
            -self.log_var
        )


        mahalanobis = (
            diff * diff * inv_var
        ).sum(-1)
        # [B,K]


        log_det = self.log_var.sum(-1)
        # [K]


        norm = (
            self.input_size *
            th.log(
                th.tensor(
                    2 * th.pi,
                    device=x.device,
                    dtype=x.dtype
                )
            )
        )


        component_log_prob = (
            -0.5 *
            (
                mahalanobis
                +
                log_det
                +
                norm
            )
        )
        logits = th.log_softmax(self.logits, dim=0)

        thresholded_gmm_logit = th.clamp((component_log_prob
                            +
                            logits)/TEMP_GMM, min=-THRESHOLD_CTX, max=THRESHOLD_CTX)
        if not self.activate_threshold:
            logit_sumexp = component_log_prob + logits
        else:
            logit_sumexp = thresholded_gmm_logit
        return th.logsumexp(
            logit_sumexp,
            dim=1
        ), component_log_prob, logits, thresholded_gmm_logit

    def forward(self, x: th.Tensor):
        point_logprob, per_component_logprob, component_logprobs, csum = self.forward_all(x)
        #per_component_logprob: [B, K]
        #component_logprobs: [K]
        #assert th.testing.assert_close(th.sum(ind_probs, dim=1) , th.ones((x.shape[0],), device=x.device, dtype=x.dtype), rtol=1e-3, atol=1e-3)

        return point_logprob
        #return per_component_logprob + component_logprobs

    def component_generation_logprob(self, batch: th.Tensor) -> th.Tensor:
        """Compute log-probabilities for each sample under each GMM component.

        Returns a tensor with shape [B, K] where each entry is:
            log p(x_b | component_k) + log p(component_k)
        """

        if batch.ndim != 2 or batch.shape[1] != self.input_size:
            raise ValueError(
                f"Expected batch shape [B, {self.input_size}], got {tuple(batch.shape)}"
            )

        centered = batch[:, None, :] - self.centroids[None, :, :]
        variance = th.exp(self.log_var)
        inv_variance = 1.0 / variance

        quadratic_term = (centered.pow(2) * inv_variance).sum(dim=-1)
        log_det_term = self.log_var.sum(dim=-1)
        normalizer = self.input_size * th.log(
            th.tensor(2.0 * th.pi, device=batch.device, dtype=batch.dtype)
        )

        conditional_logprob = -0.5 * (quadratic_term + log_det_term + normalizer)
        mixture_logprob = th.log_softmax(self.logits, dim=0)

        return conditional_logprob + mixture_logprob.unsqueeze(0)
        
    def set_centroids(self, centroids: th.Tensor):

        with th.no_grad():

            centroids = centroids.to(
                device=self.centroids.device,
                dtype=self.centroids.dtype
            )

            if centroids.shape != self.centroids.shape:
                raise ValueError(
                    f"Expected {self.centroids.shape}, got {centroids.shape}"
                )

            self.centroids.copy_(centroids)

    def initialize_from_data(
            self,
            centroids: th.Tensor,
            data: th.Tensor,
            assignments: th.Tensor,
            min_var: float = 1e-8,
        ):
            """Initialize diagonal-GMM parameters from hard cluster assignments.
    
            Args:
                centroids: [K, D] centroids to copy into the model.
                data: [B, D] data points.
                assignments: [B] hard cluster ids in [0, K-1].
                min_var: diagonal variance floor for stability.
            """
    
            with th.no_grad():
                self.set_centroids(centroids)
    
                data = data.to(device=self.centroids.device, dtype=self.centroids.dtype)
                assignments = assignments.to(device=self.centroids.device, dtype=th.long)
    
                if data.ndim != 2 or data.shape[1] != self.input_size:
                    raise ValueError(
                        f"Expected data shape [B, {self.input_size}], got {tuple(data.shape)}"
                    )
                if assignments.ndim != 1 or assignments.shape[0] != data.shape[0]:
                    raise ValueError("assignments must be [B] with the same B as data")
    
                total_points = data.shape[0]
                if total_points == 0:
                    raise ValueError("Cannot initialize from empty data")
    
                counts = th.bincount(assignments, minlength=self.num_components).to(self.centroids.dtype)
                probs = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-12)
                self.logits.copy_(th.log(probs))
    
                per_centroid_var = th.empty(
                    self.num_components,
                    self.input_size,
                    device=self.centroids.device,
                    dtype=self.centroids.dtype,
                )
    
                fallback_var = th.full(
                    (self.input_size,),
                    min_var,
                    device=self.centroids.device,
                    dtype=self.centroids.dtype,
                )
    
                for k in range(self.num_components):
                    mask = assignments == k
                    n_k = int(mask.sum().item())
    
                    if n_k > 1:
                        points_k = data[mask]
                        centered = points_k - self.centroids[k]
                        var_k = centered.pow(2).mean(dim=0).clamp_min(min_var)
                        per_centroid_var[k] = var_k
                    else:
                        per_centroid_var[k] = fallback_var
    
                self.log_var.copy_(th.log(per_centroid_var))
                

    def sample(self, n: int) -> Tuple[th.Tensor, th.Tensor]:

        with th.no_grad():

            ids = th.multinomial(
                th.softmax(self.logits, dim=0),
                n,
                replacement=True
            )


            std = th.exp(
                0.5 *
                self.log_var[ids]
            )


            return (
                self.centroids[ids]
                +
                th.randn_like(std) * std,
                ids
            )
    def sample_with_predicted_cluster(self, n: int):

        with th.no_grad():

            mixture_probs = th.softmax(self.logits, dim=0)

            # Generate samples
            ids = th.multinomial(
                mixture_probs,
                n,
                replacement=True
            )

            std = th.exp(0.5 * self.log_var[ids])

            samples = (
                self.centroids[ids]
                + th.randn_like(std) * std
            )

            # Compute log p(x | k) for every component
            diff = samples[:, None, :] - self.centroids[None, :, :]
            inv_var = th.exp(-self.log_var)

            mahalanobis = (diff.pow(2) * inv_var).sum(dim=-1)
            log_det = self.log_var.sum(dim=-1)
            norm = self.input_size * th.log(
                th.tensor(2 * th.pi, device=samples.device, dtype=samples.dtype)
            )

            component_log_prob = -0.5 * (
                mahalanobis + log_det + norm
            )

            # Add log mixture weights
            log_post: th.Tensor = component_log_prob + th.log_softmax(self.logits, dim=0)

            # MAP component
            predicted_ids = log_post.argmax(dim=-1)
            log_prob, _ = th.max(log_post, dim=-1)

            return samples, log_prob, predicted_ids




class MORMForClassificationConfig(PretrainedConfig):
    model_type = "morm_for_sequence_classification"
    has_no_defaults_at_init = True

    def vae_args(self) -> dict:
        # Return all arguments that start with "vae_" as a dictionary
        return {k: v for k, v in self.__dict__.items() if k.startswith("vae_")}
    
    @property
    def loss_management(self) -> MOLossManagement:
        loss_func_enum = MOLossFunctions(self.loss_func_type)
        return MOLossManagement(loss_func_enum, self.loss_func_type_kwargs)

    def __init__(
        self,
        # This will be set properly in the model init based on the tokenizer
        context_implementation: str = ContextImplementations.NO_CONTEXT.value,
        vs_weight_initialization: Literal['dirichlet',
                                     'span', 'equal'] = "dirichlet",
        do_initialization: bool = True,
        do_vs_initialization: bool = True,
        normalize_context: bool =False,
        smooth_evaluation: bool = True,
                                                        
        vae_pretrain_epochs: int = 10,
        vae_initial_temperature: float = 1.0,
        vae_lambda_clustering: float = 1.0,
        vae_beta: float = 1.0,
        vae_latent_dim: int =32,
        vae_type: str = "vae",
        vae_dropout: int = 0.0,
        vae_layer_activation: str = "ReLU",
        vae_reconstruction_loss: str = "mse",
        vae_n_hidden_layers: int = 4,
        vae_hidden_dim: int = 512,
        vae_final_encoder_layer_activation: str = "none",
        vae_resampling_iterations: int = 10,
        vae_similarity: str = "cosine",
        entropy_coefficient: float = 0.0,
        ctx_coefficient: float = 0.0,
        vs_selection_coefficient: float = 0.0,
        training_initialization_data_size: int|str = "all",
        sharp_context_classification: bool = True,
        direct_context_to_vs_relation: bool = False,
        detach_context_selection_for_value_system_selection: bool = False,
        detach_vs_selection_for_value_system_weight_training: bool = False,
        pad_token_id: int = "UNKNOWN",
        num_values: int = 3,
        input_size: int = "infer", 
        input_size_vs: int = "infer",
        hidden_sizes: list[int] = [1024, 1024, 1024],
        vs_layer_hidden_sizes: list[int] = [],
        vs_layer_dropout: float = 0.0,
        vs_layer_intermediate_activation: str = "ReLU",
        value_layer_dropout: float = 0.1,
        value_layer_intermediate_activation: str = "ReLU",
        value_layer_final_activation: str = "none",
        sentence_transformer_name: str = "all-MiniLM-L6-v2",
        use_sentence_transformer: bool = False,
        layer_normalization: Literal['LayerNorm',
                                     'BatchNorm', 'none'] = 'LayerNorm',
                            
        reward_diff_threshold: float = 50.0,
        assume_qualitative_labels: bool = True,
        discordance_epsilon=MIN_EPSILON,
        activate_discordance_epsilon_for_loss: bool = False,
        check_undefined_label: bool = True,
        grounding_loss_tendency_update_ratio: float = 0.001,
        update_tendencies_every_n_steps: int = 1,
        use_validation_for_tendencies: bool = False,
        rew_center_coefficient: float = 0.0,
        gradient_accumulation_steps: int = 2,
        use_metrics_or_losses_for_lagrange_updates: str = "metrics",
        use_exponential_moving_average_or_optimum_targets: str = "optimum",
        grad_on_only_worst_value: bool = False,
        zero_constraint: bool = True,
        lambda_decay: float = 0.0,
        gather_train_metrics: bool = False,
        use_ideal_grounding_model: bool = False,
        dtype: str = "float32",
        base_model_name_or_path: Optional[str] = None,
        base_model_trust_remote_code: bool = True,
        base_model_num_labels: int = 1,
        use_base_model_heads: bool = False,
        base_model_reward_heads_module_name: str = None,
        base_model_value_system_module_name: str = None,
        base_model_reward_head_indices: list = None,
        loss_func_type: str = MOLossFunctions.DEFAULT.value,
        loss_func_kwargs: dict = None,
        lr_grounding: Optional[float] = 0.0001,
        lr_value_system: Optional[float] = 0.0001,
        lr_context: Optional[float] = 0.0,
        lr_lambda: Optional[float] = 0.0,
        max_contexts: Optional[float]=5,
        max_value_systems: Optional[float]=3,
        **kwargs,
    ):
        assert num_values > 0, "num_values must be greater than 0"
        # assert len(hidden_sizes) > 0, "hidden_sizes must be a non-empty list"

        if value_layer_intermediate_activation not in VALUE_LAYER_ACTIVATIONS.keys():
            raise ValueError(
                f"value_layer_intermediate_activation must be one of {list(VALUE_LAYER_ACTIVATIONS.keys())}, but got {value_layer_intermediate_activation}")
        if value_layer_final_activation not in VALUE_LAYER_ACTIVATIONS.keys():
            raise ValueError(
                f"value_layer_final_activation must be one of {list(VALUE_LAYER_ACTIVATIONS.keys())}, but got {value_layer_final_activation}")

        if layer_normalization not in ['LayerNorm', 'BatchNorm', 'none']:
            raise ValueError(
                f"layer_normalization must be one of 'LayerNorm', 'BatchNorm', 'none', but got {layer_normalization}")

        default_id2label = {
            index: f"VALUE_{index}" for index in range(num_values)
        }
        default_id2label[num_values] = "VALUE_SYSTEM"
        id2label = kwargs.pop("id2label", default_id2label)
        label2id = kwargs.pop(
            "label2id", {label: index for index, label in id2label.items()})

        """if pad_token_id == "UNKNOWN":
            raise ValueError("pad_token_id must be set to a valid integer value corresponding to the tokenizer's pad token ID. It is currently set to 'UNKNOWN', which is not valid. Please set it to the correct value when initializing the config.")
        """
        self.direct_context_to_vs_relation=direct_context_to_vs_relation
        self.vs_layer_hidden_sizes = vs_layer_hidden_sizes
        self.vs_layer_dropout = vs_layer_dropout
        self.vs_layer_intermediate_activation = vs_layer_intermediate_activation
        self.vs_weight_initialization = vs_weight_initialization

        self.normalize_context = normalize_context
        self.smooth_evaluation=smooth_evaluation
        self.sentence_transformer_name = sentence_transformer_name
        self.use_sentence_transformer = use_sentence_transformer

        self.vae_pretrain_epochs = vae_pretrain_epochs
        self.vae_initial_temperature = vae_initial_temperature
        self.vae_lambda_clustering = vae_lambda_clustering
        self.vae_beta = vae_beta
        self.vae_latent_dim=vae_latent_dim
        self.vae_type=vae_type
        self.vae_dropout=vae_dropout
        self.vae_layer_activation=vae_layer_activation
        self.vae_reconstruction_loss=vae_reconstruction_loss
        self.vae_n_hidden_layers=vae_n_hidden_layers
        self.vae_hidden_dim=vae_hidden_dim
        self.vae_resampling_iterations=vae_resampling_iterations
        self.vae_similarity=vae_similarity
        self.vae_final_encoder_layer_activation = vae_final_encoder_layer_activation

        self.context_implementation = context_implementation
        self.sharp_context_classification = sharp_context_classification
        self.do_initialization = do_initialization
        self.do_vs_initialization = do_vs_initialization
        self.vs_selection_coefficient = vs_selection_coefficient
        self.ctx_coefficient = ctx_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_contexts = max_contexts
        self.detach_context_selection_for_value_system_selection = detach_context_selection_for_value_system_selection
        self.detach_vs_selection_for_value_system_weight_training = detach_vs_selection_for_value_system_weight_training

        self.max_value_systems = max_value_systems
        self.pad_token_id = pad_token_id
        self.num_values = num_values
        self.gather_train_metrics = gather_train_metrics
        self.hidden_sizes = hidden_sizes
        self.value_layer_dropout = value_layer_dropout
        self.value_layer_intermediate_activation = value_layer_intermediate_activation
        self.activate_discordance_epsilon_for_loss = activate_discordance_epsilon_for_loss
        self.value_layer_final_activation = value_layer_final_activation
        self.reward_diff_threshold = reward_diff_threshold
        self.assume_qualitative_labels = assume_qualitative_labels
        self.check_undefined_label = check_undefined_label
        self.grounding_loss_tendency_update_ratio = grounding_loss_tendency_update_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_metrics_or_losses_for_lagrange_updates = use_metrics_or_losses_for_lagrange_updates
        self.use_exponential_moving_average_or_optimum_targets = use_exponential_moving_average_or_optimum_targets
        self.grad_on_only_worst_value = grad_on_only_worst_value
        self.update_tendencies_every_n_steps = update_tendencies_every_n_steps
        self.use_validation_for_tendencies = use_validation_for_tendencies
        self.zero_constraint = zero_constraint
        self.rew_center_coefficient = rew_center_coefficient
        self.discordance_epsilon = discordance_epsilon
        self.input_size = input_size
        self.input_size_vs = input_size_vs
        self.training_initialization_data_size = training_initialization_data_size
        if isinstance(dtype, th.dtype):
            self.dtype = str(dtype).replace("th.", "")
        else:
            self.dtype = str(dtype)
        self.use_ideal_grounding_model = use_ideal_grounding_model
        self.layer_normalization = layer_normalization
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_trust_remote_code = bool(base_model_trust_remote_code)
        self.base_model_num_labels = int(base_model_num_labels)
        self.use_base_model_heads = use_base_model_heads
        self.base_model_reward_heads_module_name = base_model_reward_heads_module_name
        self.base_model_value_system_module_name = base_model_value_system_module_name
        # Store loss_func_type as string value for JSON serialization compatibility
        if isinstance(loss_func_type, MOLossFunctions):
            self.loss_func_type = loss_func_type.value
        else:
            self.loss_func_type = loss_func_type
        self.loss_func_type_kwargs = loss_func_kwargs if loss_func_kwargs is not None else {}

        loss_manage = self.loss_management # requires self. loss_functype and loss_functypekwargs.

        if not loss_manage.should_apply_grad_on_grounding_parameters():
            self.lr_grounding = 0.0
        else:
            assert lr_grounding is not None and lr_grounding > 0.0, f"Loss function type {loss_func_type} requires applying gradients on grounding parameters, but lr_grounding is set to {lr_grounding}. Please set lr_grounding to a positive value to enable optimization of grounding parameters."
            self.lr_grounding = lr_grounding

        if not loss_manage.should_apply_grad_on_value_system_weights():
            self.lr_value_system = 0.0
        else:
            assert lr_value_system is not None and lr_value_system > 0.0, f"Loss function type {loss_func_type} requires applying gradients on value system parameters, but lr_value_system is set to {lr_value_system}. Please set lr_value_system to a positive value to enable optimization of value system parameters."
            self.lr_value_system = lr_value_system
        
        if not loss_manage.should_apply_grad_on_context_parameters():
            self.lr_context = 0.0
        else:
            assert lr_context is not None and lr_context > 0.0, f"Loss function type {loss_func_type} requires applying gradients on context parameters, but lr_context is set to {lr_context}. Please set lr_context to a positive value to enable optimization of value system parameters."
            self.lr_context = lr_context
        

        self.lambda_decay = lambda_decay
        if not loss_manage.should_apply_grad_on_lagrange_multipliers():
            self.lr_lambda = 0.0
            self.lambda_decay = 0.0
        else:
            if lr_lambda is None:
                lr_lambda = lr_value_system
            assert lr_lambda is not None and lr_lambda > 0.0, f"Loss function type {loss_func_type} requires applying gradients on Lagrange multipliers, but lr_lambda is set to {lr_lambda}. Please set lr_lambda to a positive value to enable optimization of Lagrange multipliers."
            self.lr_lambda = lr_lambda

        self.base_model_reward_head_indices = base_model_reward_head_indices if base_model_reward_head_indices is not None else "use_base_model_value_system_module_name"

        
        super().__init__(num_labels=num_values + 1,
                         id2label=id2label, label2id=label2id, **kwargs)
        
class CustomVAEConfig(VAEConfig):

    def __init__(self, input_dim: int, 
                 vae_latent_dim: int,
                vae_reconstruction_loss: str,
                vae_type: str, 
                vae_dropout: int,
                vae_layer_activation: str,
                vae_n_hidden_layers: int , 
                vae_final_encoder_layer_activation: str,
                vae_resampling_iterations: int,
                vae_similarity: str,
                vae_hidden_dim: int , 
                vae_initial_temperature: float, 
                vae_lambda_clustering: float,
                vae_is_binary: bool = False,
                vae_pretrain_epochs: int = 10,
                  **kwargs):
        super().__init__(input_dim=input_dim, latent_dim=vae_latent_dim, reconstruction_loss=vae_reconstruction_loss, **kwargs)
        self.n_hidden_layers = vae_n_hidden_layers
        self.hidden_dim = vae_hidden_dim
        self.pretrain_epochs = vae_pretrain_epochs
        self.type = vae_type
        self.dropout = vae_dropout
        self.layer_activation = vae_layer_activation
        self.resampling_iterations = vae_resampling_iterations if vae_type == "vae" else 1
        self.final_layer_activation = vae_final_encoder_layer_activation
        self.similarity = vae_similarity
        self.lambda_clustering = vae_lambda_clustering
        self.initial_temperature = vae_initial_temperature
        self.binary = vae_is_binary

class CustomEncoder(Encoder_VAE_MLP):
    def __init__(self, args: CustomVAEConfig, device, dtype):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.n_hidden_layers = args.n_hidden_layers
        self.layer_activation = args.layer_activation
        self.final_layer_activation = args.final_layer_activation
        self.vae_type = args.type
        layers = nn.ModuleList()

        encoder = nn.Sequential(*construct_layers(input_dim=np.prod(args.input_dim),
                         hidden_sizes=[self.hidden_dim] * (self.n_hidden_layers-1),
                         intermediate_activation=self.layer_activation,
                         dropout=args.dropout,
                         device=device,
                         dtype=dtype,
                         n_outputs=self.hidden_dim,
                         final_activation=self.final_layer_activation, # TODO ??
                         final_activation_kwargs={}))
        
        layers.append(encoder)

        self.layers = layers
        self.depth = len(layers)

        if args.type == "vae":    
            self.embedding = nn.Linear(self.hidden_dim, self.latent_dim, device=device, dtype=dtype)
            self.log_var = nn.Linear(self.hidden_dim, self.latent_dim, device=device, dtype=dtype)
        elif args.type == "ae":
            self.embedding = nn.Linear(self.hidden_dim, self.latent_dim, device=device, dtype=dtype)
            self.log_var = None
        else:
            raise ValueError(f"Unsupported VAE type: {args.type}")
        print("ENCODER", self)

    def forward(self, x: th.Tensor, output_layer_levels: List[int] = None) -> ModelOutput:
        if self.vae_type == "ae":
            output = ModelOutput()
    
            max_depth = self.depth
    
            if output_layer_levels is not None:
    
                assert all(
                    self.depth >= levels > 0 or levels == -1
                    for levels in output_layer_levels
                ), (
                    f"Cannot output layer deeper than depth ({self.depth}). "
                    f"Got ({output_layer_levels})."
                )
    
                if -1 in output_layer_levels:
                    max_depth = self.depth
                else:
                    max_depth = max(output_layer_levels)
    
            out = x.reshape(-1, np.prod(self.input_dim))
    
            for i in range(max_depth):
                out = self.layers[i](out)
    
                if output_layer_levels is not None:
                    if i + 1 in output_layer_levels:
                        output[f"embedding_layer_{i+1}"] = out
                if i + 1 == self.depth:
                    output["embedding"] = self.embedding(out)
                    #output["log_covariance"] = self.log_var(out)
            return output
        elif self.vae_type == "vae":
            return super().forward(x)

class CustomDecoder(Decoder_AE_MLP):
    def __init__(self, args: CustomVAEConfig, device, dtype):
        BaseDecoder.__init__(self)
        
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.n_hidden_layers = args.n_hidden_layers
        self.layer_activation = args.layer_activation
        
        layers = nn.ModuleList()

        decoder = nn.Sequential(*construct_layers(input_dim=self.latent_dim,
                         hidden_sizes=[self.hidden_dim] * self.n_hidden_layers,
                         intermediate_activation=self.layer_activation,
                         dropout=args.dropout,
                         device=device,
                         dtype=dtype,
                         n_outputs=np.prod(self.input_dim),
                         final_activation="none", # TODO ??
                         final_activation_kwargs={}))
        layers.append(decoder)
        """layers.append(nn.Sequential(nn.Linear(args.latent_dim, args.hidden_dim, device=device, dtype=dtype), nn.ReLU()))
        for _ in range(args.n_hidden_layers - 1):
                    layers.append(nn.Sequential(nn.Linear(self.hidden_dim_size, self.hidden_dim_size, device=device, dtype=dtype), nn.ReLU()))
              """  
        """layers.append(
            nn.Sequential(nn.Linear(args.hidden_dim, int(np.prod(args.input_dim)), device=device, dtype=dtype), nn.Sigmoid())
        )"""

        self.layers = layers
        self.depth = len(layers)
        print("DECODER", self)


class CustomVAE(VAE):
    def extra_parameters1(self) -> List[nn.Parameter]:
        return [self._latent_centroids]
    def extra_parameters2(self) -> List[nn.Parameter]:
        return None
    def context_parameters(self) -> List[nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    # THIS IS INSPIRED BY: https://arxiv.org/pdf/1806.10069
    def plot_embedding_space(self, sample_data: th.Tensor, 
                             sample_labels: th.Tensor=None, 
                             original_space_centroids: th.Tensor=None, 
                             save_path: str = None, 
                             output: ModelOutput = None, 
                             sampling_reps = 5):
        
        with th.no_grad(): 
            encoder_output = self.encoder(sample_data)
            if original_space_centroids is not None:
                # Project original space centroids to latent space
                original_space_centroids = original_space_centroids.to(sample_data.device, sample_data.dtype)
                latent_original_centroids = self.encoder(original_space_centroids).embedding
                
            std_m=None
            if self.model_config.type == "vae":
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                    
                std = th.exp(0.5 * log_var)
                std_m = th.mean(std).item()

                z_all = []
                for r in range(sampling_reps):
                    z, eps = self._sample_gauss(mu, std)
                    z = z.cpu().numpy()
                    z_all.append(z)
                z = np.concatenate(z_all, axis=0)
            else:
                z = encoder_output.embedding.cpu().numpy()
            # Fit a tsne model to the latent space and the centroids
            
            umap = UMAP(n_components=2, random_state=42)
            z_with_centroids = np.concatenate([z, self.latent_centroids.detach().cpu().numpy()], axis=0)
            if self.latent_dim > 2:
                z_umap_with_centroids = umap.fit_transform(z_with_centroids)
            else:
                z_umap_with_centroids = z_with_centroids
            z_umap = z_umap_with_centroids[:-self.num_contexts]
            z_umap_centroids = z_umap_with_centroids[-self.num_contexts:]
            if sample_labels is None:
                #labels set to be the centroid closest to each point in the latent space
                closest_centroids, _ = pairwise_distances_argmin_min(z, self.latent_centroids.detach().cpu().numpy())
                sample_labels = closest_centroids 
            if original_space_centroids is not None:
                # Project original space centroids to latent space and then to UMAP space
                latent_original_centroids = self.encoder(original_space_centroids).embedding
                if self.latent_dim > 2:
                    umap_original_centroids = umap.transform(latent_original_centroids)
                else:
                    umap_original_centroids = latent_original_centroids.detach().cpu().numpy()
            plt.figure(figsize=(8, 6))
            if sample_labels.shape[0] != z_umap.shape[0]:
                sample_labels_expanded = np.zeros((len(sample_labels) * sampling_reps,))
                assert len(sample_labels_expanded) == z_umap.shape[0], f"sample_labels_expanded shape {len(sample_labels_expanded)} does not match z_umap shape {z_umap.shape[0]}"

                for i in range(sample_labels.shape[0]):
                    sample_labels_expanded[i*sampling_reps:(i+1)*sampling_reps] = sample_labels[i]
                    
                sample_labels = sample_labels_expanded
            plt.scatter(z_umap[:, 0], z_umap[:, 1], c=sample_labels, cmap='viridis', s=5, alpha=0.8)
            # plot centroids the same color as the closest points in the latent space:
            plt.scatter(z_umap_centroids[:, 0], z_umap_centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
            if original_space_centroids is not None:
                plt.scatter(umap_original_centroids[:, 0], umap_original_centroids[:, 1], c='blue', marker='o', s=100, label='Original Centroids')
            #plt.scatter(z_umap_centroids[:, 0], z_umap_centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
            if output is None:
                plt.title('Latent Space UMAP Projection')
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')
            else:
                plt.title(f"Loss: {output.loss.item():.4f}, SDMEAN {std_m if self.model_config.type != "ae" else "None"}\n KL: {output.reg_loss.item():.4f} Recon: {output.recon_loss.item():.4f} loss_vae_pure {output.loss_vae_pure.item():.4f} loss_clustering {output.loss_clustering:.4f}")
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')
            plt.legend()
            if save_path is not None:
                plt.savefig(save_path + ".png", dpi=50)
            plt.show() 

    def init_centroids(self, device, dtype) -> th.Tensor:
        return nn.Parameter(
            (th.rand(
                (self.num_contexts, self.latent_dim),
                device=device,
                dtype=dtype
            )*2-1.0), requires_grad=True
        )

    @property
    def latent_centroids(self) -> th.Tensor:
        return self._latent_centroids

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        if not self._pretrain_mode:
            return super().parameters(recurse=recurse)
        else:
            return self.context_parameters()
        
    def __init__(self, vae_config: CustomVAEConfig, encoder: BaseEncoder, decoder: BaseDecoder, 
                num_contexts: int, device, dtype, **kwargs):
        super().__init__(vae_config, encoder, decoder)
        self.train()
        self.num_contexts = num_contexts
        self._temperature = self.model_config.initial_temperature
        self._latent_centroids = self.init_centroids(device, dtype)
        self.set_pretrain_mode(False)
    
    def loss_function_unreduced(self, recon_x: th.Tensor, x: th.Tensor, mu: th.Tensor, log_var: th.Tensor=None, z: th.Tensor=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    
            if self.model_config.reconstruction_loss == "mse":
                recon_loss = (
                    0.5
                    * nn.functional.mse_loss(
                        recon_x.reshape(x.shape[0], -1),
                        x.reshape(x.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)
                )
    
            elif self.model_config.reconstruction_loss == "bce":
    
                recon_loss = nn.functional.binary_cross_entropy(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
            if self.model_config.type == "vae":
                KLD = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
            else:
                KLD = th.zeros_like(recon_loss)
            return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0), recon_loss + KLD 

    def loss_function(self, recon_x: th.Tensor, x: th.Tensor, mu: th.Tensor, log_var: th.Tensor=None, z: th.Tensor=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        ret = self.loss_function_unreduced(recon_x, x, mu, log_var, z)
        return ret[0], ret[1], ret[2]
    
    def similarity(self, x: th.Tensor,y: th.Tensor, dim:int=1) -> th.Tensor:
            if self.model_config.similarity=="cosine":
                return th.cosine_similarity(x, y, dim=dim, eps=1e-2)
            elif self.model_config.similarity=="euclidean":
                return -th.sum((x-y)**2, dim=dim)#/(th.norm(x, dim=dim) + th.norm(y, dim=dim) + 1e-8)
                
    def update_temperature(self) -> None:
            if self.training:
                self._temperature = max(0.99*self._temperature, 0.001)
            else:
                raise ValueError("Temperature can only be set when gradients are enabled.")
            return self.temperature
    
    @property
    def temperature(self) -> float:
        return self._temperature

    def forward(self, hidden_state: th.Tensor|dict, loss_clustering= None, epoch=None, **kwargs):
           
            if isinstance(hidden_state, dict):
                to_basic = hidden_state
                hidden_state = hidden_state["data"]
            else:
                to_basic = {"data": hidden_state}
            model_output = self.forward_basic(to_basic)
            z = model_output["z"]
            mu = model_output["mu"]
            log_var = model_output["log_var"]
                    #prob_z = gaussian_prob(z, mu, log_var).detach()
                    #assert prob_z.shape==(z.shape[0],)
    
            assert z.shape[1] == self.latent_centroids.shape[1]
                                    
            ucentroids = self.latent_centroids.unsqueeze(0)
            zrepresentation = z.unsqueeze(1)
                    
            #context_logit_detached_repr = self.similarity(zrepresentation.detach(), ucentroids, dim=2)
            context_logit = self.similarity(zrepresentation, ucentroids, dim=2)
    
            #print("Z: ", z[0:5])
            #print("LATENT CENTROIDS: ", self.latent_centroids[0:5])
            #print(self.similarity(z[0], self.latent_centroids[0], dim=0), context_logit[0][0])
            #print(self.similarity(z[0], self.latent_centroids[1], dim=0), context_logit[0][1])
            #print("CONTEXT LOGIT", context_logit[0:5], context_logit[0][0])
            #print("Similarity check", self.similarity(z[0], self.latent_centroids[0], dim=0), context_logit[0][0])
            assert th.allclose(context_logit[1][4], self.similarity(z[1], self.latent_centroids[4], dim=0))
            assert context_logit.shape == (hidden_state.shape[0], self.num_contexts)
                    #per_component_logprob = per_component_logprob/th.max(th.abs(per_component_logprob))
                    #SOFT: context_logprobs = th.log_softmax(per_component_logprob, dim=1) + component_logprobs
                    #HARD: context_logprobs = per_component_logprob + component_logprobs
                    
            #context_logprobs_detached_repr = self.sim_to_logprob(context_logit_detached_repr)
            context_logit_with_detached_centroids = self.similarity(zrepresentation,  ucentroids.detach(), dim=2)
            context_logprobs_detached_centroid = self.sim_to_logprob(context_logit_with_detached_centroids)            
            context_logprobs = self.sim_to_logprob(context_logit_with_detached_centroids)
            
                    # TODO: ?? self.update_temperature()
            assert context_logprobs.shape == (hidden_state.shape[0], self.num_contexts)
                    
                    #print("SHOULD BE", th.log(per_component_logprob.exp() * component_logprobs.exp()))
                    #print("IT GOES:", context_logprobs)
                    #assert th.allclose(th.log(per_component_logprob.exp() * component_logprobs.exp()), context_logprobs, atol=1e-3, rtol=0.03)
                    
            with th.no_grad():
                ctx_assignments = random_argmax(context_logprobs, dim=1)
                    
            if loss_clustering is None:
                loss_clustering, repeat_inference = self.clustering_algorithm_loss_or_update(model_output, ctx_assignments, context_logprobs)
                if repeat_inference:
                    return self.forward(hidden_state, loss_clustering=loss_clustering)


            loss_pure_unreduced = model_output.pop("loss_unreduced")
            loss_unreduced = loss_pure_unreduced + self.model_config.lambda_clustering * loss_clustering
            loss_clustering = th.mean(loss_clustering) if isinstance(loss_clustering, th.Tensor) else float(loss_clustering)
            loss_vae_pure = model_output.pop("loss")
            if __debug__:
                 assert th.allclose(th.mean(loss_pure_unreduced), loss_vae_pure, atol=1e-4, rtol=1e-3), f"Loss mismatch: loss_pure_unreduced {loss_pure_unreduced}, loss_vae_pure {loss_vae_pure}"
                        
            loss_vae = loss_vae_pure + self.model_config.lambda_clustering * loss_clustering
            if self._pretrain_mode:
                assert self.model_config.lambda_clustering == 0.0, "During pretraining, lambda_clustering should be set to 0.0"
            
            return ModelOutput(loss_clustering=loss_clustering, 
                               loss_unreduced=loss_unreduced,
                               loss=loss_vae,
                               loss_vae_pure=loss_vae_pure,
                               context_logprobs=context_logprobs,
                               context_logprobs_detached_centroid=context_logprobs_detached_centroid,
                               ctx_assignments=ctx_assignments,
                               **model_output)

    def sim_to_logprob(self, logit_sim: th.Tensor, direct_prob=False) -> th.Tensor:
        if self.model_config.similarity == "euclidean":
            assert th.all(logit_sim <= 0), "Euclidean similarity should be non-positive"

            logit_distance = -logit_sim
            if logit_sim.shape == (logit_distance.shape[0], self.num_contexts):
                min_log = logit_distance.min(dim=1).values.unsqueeze(1)
                assert min_log.shape == (logit_distance.shape[0], 1)
                assert logit_distance.shape == (logit_distance.shape[0], self.num_contexts)
                assert all([logit_distance[0,i] >= min_log[0] for i in range(self.num_contexts)])
                
            else:
                assert logit_distance.shape == (self.num_contexts,)
                min_log = logit_distance.min().values
                assert min_log.shape == ()
                assert logit_sim.shape == (self.num_contexts,)

            if direct_prob:
                return th.softmax(-(logit_distance-min_log)/self.temperature, dim=1)
            else:
                return th.log_softmax(-(logit_distance-min_log)/self.temperature, dim=1)
        elif self.model_config.similarity == "cosine":
            if direct_prob:
                return th.softmax(logit_sim/self.temperature, dim=1)
            else:
                return th.log_softmax(logit_sim/self.temperature, dim=1)
    
    def clustering_algorithm_loss_or_update(self, model_output: ModelOutput, ctx_assignments, context_logprobs):
        ucentroids, mu_representation = self.latent_centroids.unsqueeze(0), model_output["mu"].unsqueeze(1)
        
        context_logit_of_mu = self.similarity(mu_representation.detach(), ucentroids, dim=2) # THIS IS DISTANCE
        # minimum distance of each sample to any centroid
        context_distance_of_mu = -context_logit_of_mu
        context_probs_of_mu = self.sim_to_logprob(context_logit_of_mu, direct_prob=True)

        if __debug__:
            with th.no_grad():
                min_distance = (-context_logit_of_mu).min(dim=1).values
                assert min_distance.shape == (model_output["mu"].shape[0],)
                assert context_logit_of_mu.shape == (model_output["mu"].shape[0], self.num_contexts)
                context_probs_of_mu_test = th.softmax(-((-context_logit_of_mu)-min_distance.unsqueeze(1))/self.temperature, dim=1) # THIS SHOULD BE "CLOSENES"
                assert th.allclose(context_probs_of_mu, context_probs_of_mu_test, atol=1e-4, rtol=1e-3), f"Context probs of mu do not match expected softmax values. Got {context_probs_of_mu}, expected {th.log(context_probs_of_mu_test)}"
        
        #print("CONTEXT LOGIT OF MU", context_logit_of_mu[0:5])
        #print("CONTEXT LOGPROBS OF MU", context_logprobs_of_mu[0:5])
        
        loss_clustering_unreduced = (th.sum(context_distance_of_mu*(context_probs_of_mu), dim=-1))
        assert loss_clustering_unreduced.shape == (model_output["mu"].shape[0],)
        return loss_clustering_unreduced, False

    def set_pretrain_mode(self, pretrain: bool):
        self._pretrain_mode = pretrain
        if pretrain:
            self._orig_lambda = self.model_config.lambda_clustering
            self.model_config.lambda_clustering = 0.0
            self.latent_centroids.requires_grad_(False)
        else:
            if hasattr(self, "_orig_lambda") and self._orig_lambda is not None:
                self.model_config.lambda_clustering = self._orig_lambda
            
            self.latent_centroids.requires_grad_(True)

    def initialize_from_data(self, centroids: th.Tensor, data: th.Tensor, assignments: th.Tensor,  eval_data: th.Tensor, eval_assignments: th.Tensor, eval_ground_truth: th.Tensor=None, args: TrainingArguments=None, config: MORMForClassificationConfig=None, **kwargs):
        # This code pretrains the VAE/AE model on the data and initializes the latent centroids with Kmeans
        self.requires_grad_(True)
        
        self.set_pretrain_mode(True)
        
        w = th.is_grad_enabled()
        assert w
        train_config = BaseTrainerConfig(
            output_dir='my_model',
            learning_rate=config.lr_context,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            num_epochs=max(int(0.01*args.num_train_epochs), 1), # Change this to train the model a bit more
            optimizer_cls="AdamW",
            optimizer_params={"weight_decay": args.weight_decay}
        )
        
        pipeline = TrainingPipeline(
            training_config=train_config,
            model=self
        )
        with th.enable_grad():
            probe_output = self({"data": data[:10]})
        assert probe_output.loss.requires_grad, "VAE initialization loss is detached from trainable parameters"
        dataset = BaseDataset(data, data)
        eval_dataset = BaseDataset(eval_data, eval_data)
        
        if eval_ground_truth is not None:
                        eval_ground_truth_centroids = th.stack([
                            eval_data[eval_ground_truth == i].mean(dim=0) if th.any(eval_ground_truth == i) else th.mean(eval_data, dim=0)
                            for i in range(self.num_contexts)
                        ])
        eval_centroids = th.stack([
            eval_data[eval_assignments == i].mean(dim=0) if th.any(eval_assignments == i) else th.mean(eval_data, dim=0)
                        
                        for i in range(self.num_contexts)
                    ])
        for r in range(self.model_config.pretrain_epochs):
            print(f"Pretraining VAE/AE model, epoch {r+1}/{self.model_config.pretrain_epochs}...")
            # at each 10% of the pretraining epochs, plot the embedding space and the centroids
            if r % max(1, self.model_config.pretrain_epochs // 10) == 0:
                self.eval()
                output = self.forward(eval_data)
                self.plot_embedding_space(eval_data, save_path=f"pretrain_plots/vae_{self.__class__.__name__}_{self.model_config.type}_before_epoch_{r}", output=output)
                # Compute the eval centroids as the mean of the hidden states for each context assignment
                
                
                assert th.allclose(eval_centroids[0], eval_data[eval_assignments==0].mean(dim=0))
                self.plot_embedding_space(eval_data, save_path=f"pretrain_plots/vae_{self.__class__.__name__}_{self.model_config.type}_before_epoch_{r}_shouldbe", output=output, sample_labels=eval_assignments, original_space_centroids=eval_centroids)

                if eval_ground_truth is not None:
                    #eval_ground_truth_centroids = eval_data[eval_ground_truth].reshape(self.num_contexts, -1, *eval_data.shape[1:]).mean(dim=1)
                    self.plot_embedding_space(eval_data, save_path=f"pretrain_plots/vae_{self.__class__.__name__}_{self.model_config.type}_before_epoch_{r}_shouldbe_groundtruth", output=output, sample_labels=eval_ground_truth, original_space_centroids=eval_ground_truth_centroids)

            self.train()
            
            pipeline(
                train_data = dataset,
                eval_data = eval_dataset
            )

            #print("C After", self.latent_centroids[0])
            
            encoded_data = self.encoder(data).embedding.detach()
            self.latent_centroids.copy_(th.tensor(
                kmeans_clustering(encoded_data.cpu().numpy(), self.num_contexts).cluster_centers_, 
                device=self.latent_centroids.device, dtype=self.latent_centroids.dtype))
                 
        print("Running k-means on the learned embeddings...")
        encoded_data = self.encoder(data).embedding.detach()
        kmeans_model = kmeans_clustering(encoded_data.cpu().numpy(),
            K=self.num_contexts,
        )

        with th.no_grad():
            encoded_centroids = kmeans_model.cluster_centers_
            self.latent_centroids.copy_(th.tensor(encoded_centroids, device=self.latent_centroids.device, dtype=self.latent_centroids.dtype))
        self.latent_centroids.requires_grad_(not isinstance(self, CustomVAENoLoss))
        
        self.plot_embedding_space(
            sample_data=eval_data,
            sample_labels=eval_ground_truth if eval_ground_truth is not None else eval_assignments,
            original_space_centroids=eval_ground_truth_centroids if eval_ground_truth is not None else eval_centroids,
            #original_space_centroids=th.as_tensor(kmeans_model.cluster_centers_, device=device, dtype=th.float32),
            save_path=f"pretrain_plots/vae_{self.__class__.__name__}_{self.model_config.type}_before_epoch_{r}_shouldbe_final",
        )
        
        self.plot_embedding_space(
            sample_data=eval_data,
            sample_labels=None,
            original_space_centroids=eval_ground_truth_centroids,
            #original_space_centroids=th.as_tensor(kmeans_model.cluster_centers_, device=device, dtype=th.float32),
            save_path=f"pretrain_plots/vae_{self.__class__.__name__}_{self.model_config.type}_before_epoch_{r}_shouldbe_final",
        )
        
        self.set_pretrain_mode(False)
        #self._temperature = 0.1 smaller is risky
    def forward_basic(self, inputs: BaseDataset, **kwargs):
            """
            The VAE model
    
            Args:
                inputs (BaseDataset): The training dataset with labels
    
            Returns:
                ModelOutput: An instance of ModelOutput containing all the relevant parameters
    
            """
    
            x = inputs["data"]
    
            encoder_output = self.encoder(x)
            if self.model_config.type == "vae":
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = th.exp(0.5 * log_var)
            else:
                mu, log_var = encoder_output.embedding, None
            
            loss_total = 0.0
            recon_loss_total = 0.0
            kld_total = 0.0
            loss_unreduced_total = 0.0
            assert isinstance(self.model_config, CustomVAEConfig) #and self.model_config.resampling_iterations > 0, "resampling_iterations must be a positive integer"
            for _ in range(self.model_config.resampling_iterations):
                if self.model_config.type == "vae":
                    z, eps = self._sample_gauss(mu, std)
                else:
                    z = mu
                #z = mu
                recon_x = self.decoder(z)["reconstruction"]
        
                loss, recon_loss, kld, loss_unreduced = self.loss_function_unreduced(recon_x, x, mu, log_var, z)
                loss_total = loss + loss_total
                recon_loss_total = recon_loss + recon_loss_total
                kld_total = kld + kld_total
                loss_unreduced_total = loss_unreduced + loss_unreduced_total
            loss = loss_total / self.model_config.resampling_iterations
            recon_loss = recon_loss_total / self.model_config.resampling_iterations
            kld = kld_total / self.model_config.resampling_iterations
            loss_unreduced = loss_unreduced_total / self.model_config.resampling_iterations

            output = ModelOutput(
                loss_unreduced=loss_unreduced,
                recon_loss=recon_loss,
                mu=mu,
                log_var=log_var,
                reg_loss=kld,
                loss=loss,
                recon_x=recon_x,
                z=z,
            )
    
            return output

class CustomVAENoLoss(CustomVAE):
    update_factor = 0.90

    def __init__(self, vae_config: CustomVAEConfig, encoder: BaseEncoder, decoder: BaseDecoder, num_contexts: int, device, dtype, **kwargs):
        super().__init__(vae_config, encoder, decoder, num_contexts, device, dtype, **kwargs)
        
    def clustering_algorithm_loss_or_update(self, model_output: ModelOutput, ctx_assignments, context_logprobs):
        # Sample cluster label according to multinomial of context_logprobs (no grad) DONT USE RANDOM ARGMAX:
        
        
        with th.no_grad():
            total_distance = 0.0
            rand_ctx_assignments = th.multinomial(context_logprobs.exp(), num_samples=1).squeeze(1)
            
            z = model_output["z"]
            # Update the centroids so they move towards the mean of the assigned latent representations z:
            for c in range(self.num_contexts):
                assigned_z = z[rand_ctx_assignments == c]
                if len(assigned_z) > 0:
                    new_centroid = assigned_z.mean(dim=0)
                    distance = th.norm(new_centroid - self.latent_centroids[c])
                    total_distance += distance
                    if self.training:
                        self.latent_centroids.data[c] = (new_centroid*(1.0-self.update_factor) + self.latent_centroids.data[c]*self.update_factor).detach().clone()
        loss_clustering = th.norm(self.latent_centroids @ self.latent_centroids.T - th.eye(self.num_contexts, device=self.latent_centroids.device, dtype=self.latent_centroids.dtype))
        return loss_clustering, self.training


class VaDEDecoder(Decoder_AE_MLP):
    def __init__(self, args: CustomVAEConfig, device, dtype, dec_act="none"):
        super().__init__(args)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.n_hidden_layers = args.n_hidden_layers
        self.layer_activation = args.layer_activation
        
        self._decoder = nn.Sequential(*construct_layers(input_dim=self.latent_dim,
                            hidden_sizes=[self.hidden_dim] * (self.n_hidden_layers-1),
                            intermediate_activation=self.layer_activation,
                            dropout=args.dropout,
                            device=device,
                            dtype=dtype,
                            n_outputs=self.hidden_dim,
                            final_activation="none", # TODO ??
                            final_activation_kwargs={}))
        
        """layers.append(nn.Sequential(nn.Linear(args.latent_dim, args.hidden_dim, device=device, dtype=dtype), nn.ReLU()))
        for _ in range(args.n_hidden_layers - 1):
                    layers.append(nn.Sequential(nn.Linear(self.hidden_dim_size, self.hidden_dim_size, device=device, dtype=dtype), nn.ReLU()))
                """  
        """layers.append(
            nn.Sequential(nn.Linear(args.hidden_dim, int(np.prod(args.input_dim)), device=device, dtype=dtype), nn.Sigmoid())
        )"""

        self.depth = 1
        input_dim_flat = np.prod(self.input_dim)
        
        self._dec_mu = nn.Linear(args.hidden_dim, input_dim_flat, device=device, dtype=dtype)
        self._dec_log_sigma = nn.Linear(args.hidden_dim, input_dim_flat, device=device, dtype=dtype)
        self._dec_act = VALUE_LAYER_ACTIVATIONS[dec_act] if dec_act is not None else None
        print("DECODER", self)
    
    def forward(self, z: th.Tensor) -> dict:
        
        h = self._decoder(z)
        x_mu = self._dec_mu(h)
        x_logvar = self._dec_log_sigma(h)
        if self._dec_act is not None:
            x_mu = self._dec_act(x_mu)

        return ModelOutput(x_mu=x_mu,x_logvar=x_logvar, reconstruction=x_mu)


from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import vsllib.vade_metrics as metrics

pi2 = 2 * math.pi
logpi2 = math.log(pi2)

class CustomVaDE(CustomVAE):

    @property
    def latent_centroids(self) -> th.Tensor:
        return self.u_p.T

    def init_centroids(self, device, dtype) -> th.Tensor:
        self.create_gmmparam(self.num_contexts, self.latent_dim, device, dtype)
        return None

    def extra_parameters1(self) -> List[nn.Parameter]:
        return None
    def extra_parameters2(self) -> List[nn.Parameter]:
        return None
    def context_parameters(self) -> List[nn.Parameter]:
        return self.parameters()
    
    def __init__(self, vae_config: CustomVAEConfig, encoder: BaseEncoder, decoder: VaDEDecoder, num_contexts: int, device, dtype, **kwargs):
        vae_config.type = "vae"
        super().__init__(vae_config, encoder, decoder, num_contexts, device, dtype, **kwargs)
        #self.create_gmmparam(self.num_contexts, self.latent_dim)
        
    def create_gmmparam(self, n_centroids: int, z_dim: int, device, dtype)-> None:
        self.theta_p = th.nn.Parameter(th.ones(n_centroids, device=device, dtype=dtype)/n_centroids, requires_grad=True) # mixture weights
        self.u_p = th.nn.Parameter(th.zeros(z_dim, n_centroids, device=device, dtype=dtype), requires_grad=True) #mean
        self.lambda_p = th.nn.Parameter(th.ones(z_dim, n_centroids, device=device, dtype=dtype),  requires_grad=True) #variance

    def reparameterize(self, mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
        if self.training:
            eps = th.randn_like(mu)
            return mu + eps * (0.5 * logvar).exp()

        return mu
    def clustering_algorithm_loss_or_update(self, model_output: ModelOutput, ctx_assignments, context_logprobs):
            return 0.0, False
    
    def initialize_gmm(self, dataloader) -> None:
            use_cuda = th.cuda.is_available()
            if use_cuda:
                self.cuda()
    
            self.eval()
            data = []
            for batch_idx, data_dict in enumerate(dataloader):
                inputs = data_dict["data"]
                labels = data_dict.get("labels", None)
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                output = self.forward(inputs)
                z = output.z
                data.append(z.data.cpu().numpy())
            data = np.concatenate(data)
            gmm = GaussianMixture(n_components=self.num_contexts,covariance_type='diag')
            gmm.fit(data)
            self.u_p.data.copy_(th.from_numpy(gmm.means_.T.astype(np.float32)))  # why transpose?
            self.lambda_p.data.copy_(th.from_numpy(gmm.covariances_.T.astype(np.float32)))

    def gmm_kmeans_cluster(self, dataloader) -> None:
            use_cuda = th.cuda.is_available()
            if use_cuda:
                self.cuda()
    
            self.eval()
            data = []
            Y = []
            for batch_idx, data_dict in enumerate(dataloader):
                inputs = data_dict["data"]
                y = data_dict.get("labels", None)
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                output = self.forward(inputs)
                mu = output.mu
                data.append(mu.data.cpu().numpy())
                Y.append(y.numpy())
            data = np.concatenate(data)
            Y = np.concatenate(Y)
            gmm = GaussianMixture(n_components=self.num_contexts, covariance_type='full')
            gmm.fit(data)
            y_pred_gmm = gmm.predict(data)
            acc = np.round(metrics.acc(Y, y_pred_gmm), 5)
            nmi = np.round(metrics.nmi(Y, y_pred_gmm), 5)
            ari = np.round(metrics.ari(Y, y_pred_gmm), 5)
            print('GMM fit of AutoEncoder embedding: acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
    
            km = KMeans(n_clusters=self.num_contexts, n_init=20)
            y_pred_kmeans = km.fit_predict(data)
            acc = np.round(metrics.acc(Y, y_pred_kmeans), 5)
            nmi = np.round(metrics.nmi(Y, y_pred_kmeans), 5)
            ari = np.round(metrics.ari(Y, y_pred_kmeans), 5)
            print('Kmeans clustering of AutoEncoder embedding: acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))

    
    
    def forward_original(self, x) -> ModelOutput:
            print("FORWARD ORIGINAL")
            print("X", x[0:5], x.shape)
            h = self.encoder(x)
            
            mu, logvar = h.embedding, h.log_covariance
            print("H", mu.grad_fn, logvar.grad_fn)
            z = self.reparameterize(mu, logvar)
            
            out_dec = self.decoder(z)
            x_mu,x_logvar = out_dec.x_mu, out_dec.x_logvar
            recon_x = x_mu
            return ModelOutput(z=z, recon_x=recon_x,x_logvar=x_logvar, 
                               mu=mu, logvar=logvar)

    def loss_function(self, recon_x_mu, recon_x_logvar, x, z, z_mean, z_log_var) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.num_contexts) # NxDxK
            z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.num_contexts)
            z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.num_contexts)
            u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
            lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
            theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.num_contexts) # NxK
            
            p_c_z = th.exp(th.log(theta_tensor2) - th.sum(0.5*th.log(pi2*lambda_tensor3)+\
                (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
            gamma = p_c_z / th.sum(p_c_z, dim=1, keepdim=True) # NxK
    
    
    
            #NX1
            if self.model_config.binary:
                BCE = -th.sum(x*th.log(th.clamp(recon_x_mu, min=1e-10))+(1-x)*th.log(th.clamp(1-recon_x_mu, min=1e-10)), 1)
            else:
                BCE = th.sum(0.5*logpi2+0.5*recon_x_logvar+0.5*(x-recon_x_mu)**2/th.exp(recon_x_logvar),1)
            logpzc = th.sum(0.5*gamma*th.sum(logpi2+th.log(lambda_tensor3)+
                th.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
            qentropy = -0.5*th.sum(1+z_log_var+logpi2, 1)
            logpc = -th.sum(th.log(theta_tensor2)*gamma, 1)
            logqcx = th.sum(th.log(gamma)*gamma, 1)
    
            # Normalise by same number of elements as in reconstruction
            kld_maybe = logpzc + qentropy + logpc + logqcx
            loss_unreduced = BCE + kld_maybe
            loss = th.mean(loss_unreduced)
    
            return loss, BCE.mean(), kld_maybe.mean(), loss_unreduced
    
    def forward_basic(self, inputs: BaseDataset, **kwargs):
                """
                The VAE model
        
                Args:
                    inputs (BaseDataset): The training dataset with labels
        
                Returns:
                    ModelOutput: An instance of ModelOutput containing all the relevant parameters
        
                """
        
                x = inputs["data"]
                print("FORWARD BASIC")
                out_original = self.forward_original(x)
                recon_x_mu, recon_x_logvar = out_original.recon_x, out_original.x_logvar
                z, mu, log_var = out_original.z, out_original.mu, out_original.logvar

                assert isinstance(self.model_config, CustomVAEConfig) #and self.model_config.resampling_iterations > 0, "resampling_iterations must be a positive integer"
                
        
                loss, recon_loss, kld, loss_unreduced = self.loss_function(recon_x_mu=recon_x_mu, 
                                                                           recon_x_logvar=recon_x_logvar, 
                                                                           x=x, z=z, z_mean=mu, z_log_var=log_var)
                    
                output = ModelOutput(
                    loss_unreduced=loss_unreduced,
                    recon_loss=recon_loss,
                    mu=mu,
                    log_var=log_var,
                    reg_loss=kld,
                    loss=loss,
                    recon_x=recon_x_mu,
                    z=z,
                )
        
                return output
    def forward(self, hidden_state: th.Tensor|dict, loss_clustering= None, epoch=None, **kwargs):
        
        if isinstance(hidden_state, dict):
            to_basic = hidden_state
            hidden_state = hidden_state["data"]
        else:
            to_basic = {"data": hidden_state}
        model_output = self.forward_basic(to_basic)
        z = model_output["z"]
        mu = model_output["mu"]
        log_var = model_output["log_var"]
                #prob_z = gaussian_prob(z, mu, log_var).detach()
                #assert prob_z.shape==(z.shape[0],)

                
        context_logit = self.predict_logits_z(z, mu, log_var)
        context_logprobs = self.sim_to_logprob(context_logit)
        #context_logprobs_detached_repr = self.sim_to_logprob(self.predict_logits_z(z, mu, log_var))
        assert context_logprobs.shape == (hidden_state.shape[0], self.num_contexts)
        
        with th.no_grad():
            ctx_assignments = random_argmax(context_logprobs, dim=1)
                
        """if loss_clustering is None:
            loss_clustering, repeat_inference = self.clustering_algorithm_loss_or_update(model_output, ctx_assignments, context_logprobs, context_logprobs_detached_repr)
            if repeat_inference:
                return self.forward(hidden_state, loss_clustering=loss_clustering)"""
    
        loss_unreduced = model_output.pop("loss_unreduced") #+ self.model_config.lambda_clustering * loss_clustering
        loss_vae = th.mean(loss_unreduced)
        #loss_clustering = th.mean(loss_clustering) if isinstance(loss_clustering, th.Tensor) else float(loss_clustering)
        return ModelOutput(loss_clustering=0.0, 
                            loss_unreduced=loss_unreduced,
                            loss=loss_vae,
                            loss_vae_pure=model_output.pop("loss"),
                            context_logprobs=context_logprobs,
                            context_logprobs_detached_centroid=context_logprobs,
                            ctx_assignments=ctx_assignments,
                            **model_output)
    
    def predict_logits(self, x) -> th.Tensor:
        assert x.shape[1] == np.prod(self.input_dim)
        out_original = self.forward_original(x)
        z, mu, log_var = out_original.z, out_original.mu, out_original.logvar
        return self.predict_logits_z(z, mu, log_var)

    def predict_logits_z(self, z, mu, log_var) -> th.Tensor:
        return self.get_logit_gamma(z, mu, log_var)
    
    def predict_proba(self, x) -> th.Tensor:
        return self.softmax(self.predict_logits(x), dim=1)

    def get_gamma(self, z, z_mean, z_log_var):
            Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.num_contexts) # NxDxK
            z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.num_contexts)
            z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.num_contexts)
            u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
            lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
            theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.num_contexts) # NxK
    
            p_c_z = th.exp(th.log(theta_tensor2) - th.sum(0.5*th.log(pi2*lambda_tensor3)+\
                (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
            gamma = p_c_z / th.sum(p_c_z, dim=1, keepdim=True)
    
            return gamma
    
    def get_logit_gamma(self, z: th.Tensor, z_mean: th.Tensor, z_log_var: th.Tensor) -> th.Tensor:
            Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.num_contexts) # NxDxK
            #z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.num_contexts)
            #z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.num_contexts)
            u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
            lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
            theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.num_contexts) # NxK
    
            log_p_c_z = th.log(theta_tensor2) - th.sum(0.5*th.log(pi2*lambda_tensor3)+\
                (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)# NxK
            
            return log_p_c_z
    