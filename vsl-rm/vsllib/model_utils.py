from pythae.models.nn import BaseDecoder, BaseEncoder
import tqdm

from transformers.configuration_utils import PretrainedConfig
from typing import Literal, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from transformers import TrainingArguments


from vsllib.defines import MIN_EPSILON, NO_RATING_MASK, VALUE_LAYER_ACTIVATIONS, ContextImplementations, MOLossFunctions, MOLossManagement

THRESHOLD = 0.0
ACTIVATE_THRESHOLD_VS =  False
THRESHOLD_CTX = 50.0
TEMP_GMM = 1.0
ACTIVATE_TEMPERATURE_GMM = False


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


from pythae.models.vae import VAE, VAEConfig
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP

from pythae.models.base.base_utils import ModelOutput


class CustomVAEConfig(VAEConfig):
    def __init__(self, input_dim: int, 
                 vae_latent_dim: int,
                vae_reconstruction_loss: str, 
                vae_type: str, 
                vae_dropout: int,
                vae_layer_activation: str,
                vae_n_hidden_layers: int , 
                vae_hidden_dim: int , vae_detach_centroids: bool = False, **kwargs):
        super().__init__(input_dim=input_dim, latent_dim=vae_latent_dim, reconstruction_loss=vae_reconstruction_loss, **kwargs)
        self.n_hidden_layers = vae_n_hidden_layers
        self.hidden_dim = vae_hidden_dim
        self.type = vae_type
        self.dropout = vae_dropout
        self.layer_activation = vae_layer_activation

class CustomEncoder(Encoder_VAE_MLP):
    def __init__(self, args: CustomVAEConfig, device, dtype):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.n_hidden_layers = args.n_hidden_layers
        self.layer_activation = args.layer_activation
        
        layers = nn.ModuleList()

        encoder = nn.Sequential(*construct_layers(input_dim=np.prod(args.input_dim),
                         hidden_sizes=[self.hidden_dim] * (self.n_hidden_layers-1),
                         intermediate_activation=self.layer_activation,
                         dropout=args.dropout,
                         device=device,
                         dtype=dtype,
                         n_outputs=self.hidden_dim,
                         final_activation=args.layer_activation, # TODO ??
                         final_activation_kwargs={}))

        layers.append(encoder)

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(self.hidden_dim, self.latent_dim, device=device, dtype=dtype)
        self.log_var = nn.Linear(self.hidden_dim, self.latent_dim, device=device, dtype=dtype)
        print("ENCODER", self)


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
    def __init__(self, vae_config: CustomVAEConfig, encoder: BaseEncoder, decoder: BaseDecoder):
        super().__init__(vae_config, encoder, decoder)
        self.train()
    def forward(self, inputs: BaseDataset, **kwargs):
            """
            The VAE model
    
            Args:
                inputs (BaseDataset): The training dataset with labels
    
            Returns:
                ModelOutput: An instance of ModelOutput containing all the relevant parameters
    
            """
    
            x = inputs["data"]
    
            encoder_output = self.encoder(x)
    
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
    
            std = th.exp(0.5 * log_var)
            z, eps = self._sample_gauss(mu, std)
            recon_x = self.decoder(z)["reconstruction"]
    
            loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)
    
            output = ModelOutput(
                recon_loss=recon_loss,
                mu=mu,
                log_var=log_var,
                reg_loss=kld,
                loss=loss,
                recon_x=recon_x,
                z=z,
            )
    
            return output
    
class MORMForClassificationConfig(PretrainedConfig):
    model_type = "morm_for_sequence_classification"
    has_no_defaults_at_init = True

    def vae_args(self):
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

        initial_temperature: float = 1.0,
        lambda_clustering: float = 0.0,
        vae_beta: float = 1.0,
        vae_latent_dim: int =32,
        vae_type: str ="VAE",
        vae_dropout: int = 0.0,
        vae_layer_activation: str = "ReLU",
        vae_reconstruction_loss: str = "mse",
        vae_n_hidden_layers: int = 4,
        vae_hidden_dim: int = 512,

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
        lr_grounding: Optional[float] = None,
        lr_value_system: Optional[float] = None,
        lr_context: Optional[float] = None,
        lr_lambda: Optional[float] = None,
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

        self.initial_temperature = initial_temperature
        self.lambda_clustering = lambda_clustering
        self.vae_beta = vae_beta
        self.vae_latent_dim=vae_latent_dim
        self.vae_type=vae_type
        self.vae_dropout=vae_dropout
        self.vae_layer_activation=vae_layer_activation
        self.vae_reconstruction_loss=vae_reconstruction_loss
        self.vae_n_hidden_layers=vae_n_hidden_layers
        self.vae_hidden_dim=vae_hidden_dim

        self.context_implementation = context_implementation
        self.sharp_context_classification = sharp_context_classification
        self.do_initialization = do_initialization
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
            self.dtype = str(dtype).replace("torch.", "")
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