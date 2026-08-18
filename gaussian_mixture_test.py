
import math

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import torch as th
import numpy as np
import torch
from matplotlib.colors import Normalize
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader

import pyro.distributions as pyro_dist 
def make_cov(logvar: th.Tensor, tril: th.Tensor, cholesky=True) -> th.Tensor:
    """ make full covarance matrix
    logvar: log variance vector
    tril: unmaksed lower triangular matrix
    cholesky: return cholesky decomposition, default=False
    """
    var = th.exp(logvar.clip(math.log(1e-6), math.log(1e5)))
    L = th.tril(tril, diagonal=-1)
    L = L + th.diag_embed(var)
    
    if not cholesky:
        L = th.bmm(L, L.transpose(-1, -2))
    return L
def sample_gmm(pi: th.Tensor, mvn_dists: th.distributions.Distribution, num_samples: int) -> tuple[th.Tensor, th.Tensor]:
    c = th.multinomial(pi, num_samples, replacement=True)
    pi_samples = th.nn.functional.one_hot(c, num_classes=len(pi))
    mvn_samples = mvn_dists.sample((num_samples,))
    
    samples = th.sum(pi_samples.unsqueeze(-1) * mvn_samples, dim=1)
    return samples, c



class GaussianMixtureContextProbability(nn.Module):

    def __init__(
        self,
        input_size: int,
        num_contexts: int,
        device: th.device = None,
        dtype: th.dtype = None
    ) -> None:

        super().__init__()

        self.input_size = input_size
        self.num_contexts = num_contexts
        self.min_variance = 1e-3

        # mixture weights (unnormalized logits)
        self.mixture_logits = nn.Parameter(
            th.randn(num_contexts, device=device, dtype=dtype)
        )
        # diagonal of Cholesky factor
        self.log_scales = nn.Parameter(
            th.zeros(num_contexts, input_size, device=device, dtype=dtype)
        )

        # Gaussian means
        self.centroids = nn.Parameter(
            th.randn(num_contexts, input_size, device=device, dtype=dtype)
        )

        

        # covariance lower-triangular parameters
        self.tril = nn.Parameter(
            th.randn(
                num_contexts,
                input_size,
                input_size,
                device=device,
                dtype=dtype
            )
        )

        self.init_params()


    def init_params(self):

        nn.init.normal_(self.centroids, 0, 1e-5)

        # start close to unit variance
        #nn.init.normal_(self.log_variances, 0, 1e-3)
        nn.init.normal_(
            self.log_scales,
            0,
            1e-3
        )
        # small correlations
        nn.init.normal_(self.tril, 0, 1e-3)


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



    def get_obs_dist(self):

        scale_tril = make_cov(
            self.log_variances,
            self.tril,
            self.min_variance
        )

        return dist.MultivariateNormal(
            loc=self.centroids,
            scale_tril=scale_tril
        )


    def get_scale_tril(self):

        """
        Returns:
            L: [K,D,D]
            where Sigma = L L^T
        """

        L = th.tril(self.tril, diagonal=-1)

        diag = th.exp(
            self.log_scales
        ).clamp_min(
            self.min_variance
        )

        L = L + th.diag_embed(diag)

        return L
    def forward(self, context_embeddings: th.Tensor):

        """
        Returns log p(x) under the Gaussian mixture.

        Input:
            x: [..., D]

        Output:
            log probability: [...]
        """

        if context_embeddings.ndim == 1:
            context_embeddings = context_embeddings.unsqueeze(0)


        if context_embeddings.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected last dim {self.input_size}, "
                f"got {context_embeddings.shape[-1]}"
            )


        # distribution over components
        obs_dist = self.get_obs_dist()


        # evaluate each component
        #
        # x:
        #   [B,1,D]
        #
        # output:
        #   [B,K]
        #
        component_log_probs = obs_dist.log_prob(
            context_embeddings.unsqueeze(-2)
        )


        # normalized mixture weights
        log_weights = th.log_softmax(
            self.mixture_logits,
            dim=0
        )



        # log sum_k pi_k N(x|mu_k,Sigma_k)
        log_prob = th.logsumexp(
            component_log_probs + log_weights,
            dim=-1
        )


        return log_prob, component_log_probs



    def sample(self, num_samples):

        with th.no_grad():

            probs = th.softmax(
                self.mixture_logits,
                dim=0
            )


            ids = th.multinomial(
                probs,
                num_samples,
                replacement=True
            )


            L = self.get_scale_tril()[ids]


            eps = th.randn(
                num_samples,
                self.input_size,
                device=L.device,
                dtype=L.dtype
            )


            samples = (
                self.centroids[ids]
                +
                th.bmm(
                    L,
                    eps.unsqueeze(-1)
                )
                .squeeze(-1)
            )


        return samples, ids
    def sampleworks(self, num_samples: int):

        with th.no_grad():

            # choose mixture components
            mixture_probs = th.softmax(
                self.mixture_logits,
                dim=0
            )

            component_ids = th.multinomial(
                mixture_probs,
                num_samples,
                replacement=True
            )


            # select Gaussian parameters
            means = self.centroids[component_ids]

            scale_tril = make_cov(
                self.log_variances[component_ids],
                self.tril[component_ids],
                self.min_variance
            )


            gaussian = dist.MultivariateNormal(
                means,
                scale_tril=scale_tril
            )


            samples = gaussian.sample()


        return samples, component_ids


class FastGaussianMixture(nn.Module):

    def __init__(
        self,
        input_size: int,
        num_components: int,
        l_entropy = 1e-1,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.l_entropy = l_entropy
        self.input_size = input_size
        self.num_components = num_components

        self.logits = nn.Parameter(
            th.zeros(num_components, device=device, dtype=dtype)
        )

        self.centroids = nn.Parameter(
            th.randn(
                num_components,
                input_size,
                device=device,
                dtype=dtype
            ) * 1e-3
        )

        self.log_var = nn.Parameter(
            th.zeros(
                num_components,
                input_size,
                device=device,
                dtype=dtype
            )
        )


    def forward(self, x):

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

        return th.logsumexp(
            component_log_prob
            +
            logits,
            dim=1
        ), component_log_prob #+ logits

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


    def sample(self, n):

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
        
class GaussianMixture(nn.Module):
    def __init__(self, num_components, dim, cov="full", batch_norm=False):
        super().__init__()
        assert cov in ["diag", "mvn"]
        self.num_components = num_components
        self.dim = dim
        self.cov = cov
        self.batch_norm = batch_norm

        self.pi_logits = nn.Parameter(torch.randn(num_components), requires_grad=True)
        self.centroids = nn.Parameter(torch.randn(num_components, 1, dim), requires_grad=True)
        self.logvar = nn.Parameter(torch.randn(num_components, 1, dim), requires_grad=True)
        self.tril = nn.Parameter(torch.randn(num_components, 1, dim, dim), requires_grad=True)
        
        if batch_norm:
            self.bn = pyro_dist.transforms.BatchNorm(dim, momentum=0.1)
            self.bn.gamma.requires_grad = False
            self.bn.beta.requires_grad = False
            
        self.init_params()

    def set_centroids(self, centroids):
        with torch.no_grad():
            cent = centroids.to(device=self.centroids.device, dtype=self.centroids.dtype)
            self.centroids.copy_(cent.unsqueeze(1))
    def init_params(self):
        nn.init.normal_(self.centroids, 0, 1e-5)
        nn.init.normal_(self.logvar, 0, 1e-3)
        nn.init.normal_(self.tril, 0, 0.001)
    
    def get_obs_dist(self):
        if self.cov == "mvn":
            L = make_cov(self.logvar, self.tril)
        else:
            L = torch.diag_embed(self.logvar.exp())
        P = dist.MultivariateNormal(self.centroids, scale_tril=L)
        if self.batch_norm:
            P = pyro_dist.TransformedDistribution(P, [self.bn])
        return P

    def forward(self, x):
        # get dists
        pi = torch.softmax(self.pi_logits, dim=-1)
        P = self.get_obs_dist()

        logp_pi = torch.log(pi).unsqueeze(1)
        logp_P = P.log_prob(x)
        logp = torch.logsumexp(logp_pi + logp_P, dim=0)
        return logp

    def sample(self, num_samples: int):
        # get dists
        pi = torch.softmax(self.pi_logits, dim=0).view(-1)
        P = self.get_obs_dist()

        # sample
        c = torch.multinomial(pi, num_samples, replacement=True)
        pi_samples = F.one_hot(c, num_classes=len(pi))
        P_samples = P.sample((num_samples, )).squeeze(-2)

        samples = torch.sum(pi_samples.unsqueeze(-1) * P_samples, dim=1)
        logp = self.forward(samples)
        return samples, logp


class GMMDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]
def train(lr: float, epochs, loader, gmm, file_name, l2_lambda=0.0, l_entropy=0.0):
    optimizer = torch.optim.AdamW(gmm.parameters(), lr=lr, weight_decay=0.0)

    start_time = time.time()
    history = []
    bar = trange(epochs)

    for e in bar:
        epoch_loss, epoch_n = 0, 0
        epoch_true_loss = 0.0
        for data_batch in loader:
            entropy_loss = 0.0
            logp, component_log_prob = gmm(data_batch)
            logits = th.log_softmax(gmm.logits, dim=0)
            k = len(gmm.centroids)
            #if l_entropy != 0.0:
            # WORKS responsibilities = th.log_softmax(component_log_prob, dim=1)#*th.exp(logits)"""
            """
            WORKS
            entropy_loss = -(
                                responsibilities *
                                responsibilities.exp()
                            ).sum(dim=1).mean()
            """
            responsibilities = th.log_softmax(component_log_prob, dim=1) #+ logits
            entropy_loss = -(
                                responsibilities *
                                responsibilities.exp()
                            ).sum(dim=1).mean()
            #avg_resp = responsibilities.mean(dim=0)
            # Mean entropy over the batch
            
            
            
            
            l2 = (
                #(gmm.centroids-gmm.centroids.mean(dim=0)).square().sum()
                #+ gmm.log_var.square().sum()
                -(logits*th.exp(logits)).mean()
            )
            l3 = (gmm.centroids-gmm.centroids.mean(dim=0)).square().mean()
            nll = -logp.mean() 
            #l_entropy=0
            #l2_lambda=0
            loss = nll + l_entropy*entropy_loss*k + l2_lambda*l2*k

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += -logp.sum().item()
            epoch_true_loss += nll.item()
            epoch_n += len(data_batch)
        
        
        epoch_loss /= epoch_n
        history.append(epoch_loss)
        if e % 100 == 0:
            plot(gmm, history, file_name)

        bar.set_postfix(
            loss=f"{epoch_true_loss:.2f}",
            time=f"{time.time() - start_time:.2f}"
        )

    print(f"FINISHED IN {time.time() - start_time:.2f} seconds")
    
    plot(gmm, history, file_name)


def train_exp(lr: float, epochs, loader, gmm, file_name, l2_lambda=0.0, l_entropy=0.0):
    """Expectation-maximization training for the diagonal Gaussian mixture.

    This keeps the legacy heuristic gradient routine in `train()` and adds a
    separate EM-style optimizer that re-estimates the component weights,
    means, and diagonal variances from soft responsibilities.
    """
    del lr, l2_lambda, l_entropy

    start_time = time.time()
    history = []
    bar = trange(epochs)

    for e in bar:
        total_resp = th.zeros(
            gmm.num_components,
            device=gmm.centroids.device,
            dtype=gmm.centroids.dtype,
        )
        total_weighted_x = th.zeros_like(gmm.centroids)
        total_weighted_x2 = th.zeros_like(gmm.centroids)
        total_nll = 0.0
        total_n = 0

        for data_batch in loader:
            data_batch = data_batch.to(device=gmm.centroids.device, dtype=gmm.centroids.dtype)
            logp, component_log_prob = gmm(data_batch)
            log_pi = th.log_softmax(gmm.logits, dim=0)
            log_resp = component_log_prob + log_pi.unsqueeze(0)
            responsibilities = th.softmax(log_resp, dim=1)

            nk = responsibilities.sum(dim=0).clamp_min(1e-12)
            total_resp += nk
            total_weighted_x += responsibilities.t() @ data_batch
            total_weighted_x2 += (responsibilities.unsqueeze(-1) * data_batch.unsqueeze(1).pow(2)).sum(dim=0)

            total_nll += (-logp).sum().item()
            total_n += len(data_batch)

        nk = total_resp.clamp_min(1e-12)
        new_pi = nk / nk.sum().clamp_min(1e-12)

        with th.no_grad():
            gmm.logits.copy_(th.log(new_pi.clamp_min(1e-12)))
            gmm.centroids.copy_(total_weighted_x / nk.unsqueeze(-1))

            second_moment = total_weighted_x2 / nk.unsqueeze(-1)
            variance = second_moment - gmm.centroids.pow(2)
            gmm.log_var.copy_(variance.clamp_min(1e-6).log())

        epoch_loss = total_nll / max(total_n, 1)
        history.append(epoch_loss)

        if e % 100 == 0:
            plot(gmm, history, file_name)

        bar.set_postfix(
            loss=f"{epoch_loss:.4f}",
            time=f"{time.time() - start_time:.2f}",
        )

    print(f"FINISHED IN {time.time() - start_time:.2f} seconds")
    plot(gmm, history, file_name)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np
import matplotlib.pyplot as plt

def plot(gmm, history, file_name):
    # ---- Training history plot ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history)
    ax.set_title("Training history")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(file_name + "_history.png")
    plt.show()
    plt.close()

    # ---- Sampling ----
    num_samples = 1000
    predicted_cluster = None

    with torch.no_grad():
        if hasattr(gmm, "sample_with_predicted_cluster"):
            x_sample, _, predicted_cluster = gmm.sample_with_predicted_cluster(num_samples)
        else:
            x_sample, _ = gmm.sample(num_samples)

    x_sample_np = x_sample.cpu().numpy() if hasattr(x_sample, "cpu") else x_sample

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # ---- Left: real data ----
    ax[0].scatter(x[:, 0], x[:, 1])
    ax[0].set_title("Data samples")

    # ---- Right: model samples ----
    if predicted_cluster is not None:
        clusters = predicted_cluster.cpu().numpy()

        markers = ['o', 's', '^', 'v', 'D', 'P', '*']
        cmap = plt.cm.get_cmap('tab10')  # distinct cluster colors

        used_clusters = np.unique(clusters)

        # ---- Plot samples per cluster ----
        for k in used_clusters:
            idx = clusters == k

            ax[1].scatter(
                x_sample_np[idx, 0],
                x_sample_np[idx, 1],
                color=cmap(k % 10),
                marker=markers[k % len(markers)],
                label=f"Cluster {k}",
                alpha=0.8,
                edgecolors='black',
                linewidths=0.3
            )

        # ---- Plot centroids ----
        centroids = gmm.centroids
        centroids_np = centroids.detach().cpu().numpy() if hasattr(centroids, "cpu") else centroids

        for k in range(centroids_np.shape[0]):
            if k in used_clusters:
                color = cmap(k % 10)
                alpha = 1.0
            else:
                color = 'black'
                alpha = 0.3

            ax[1].scatter(
                centroids_np[k, 0],
                centroids_np[k, 1],
                color=color,
                marker=markers[k % len(markers)],
                s=180,
                alpha=alpha,
                edgecolors='white',
                linewidths=1.2
            )

        # ---- Legend ----
        ax[1].legend(title="Cluster", loc="best")

    else:
        ax[1].scatter(x_sample_np[:, 0], x_sample_np[:, 1])

    ax[1].set_title("Model samples")

    plt.tight_layout()
    plt.savefig(file_name + "_samples.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    seed = 453765
    th.manual_seed(seed)

    # make gmm
    K = 4
    dim = 2
    pi = th.softmax(th.randn(K), dim=-1)
    mu = th.rand(K, dim).uniform_(-20, 20)
    logvar = th.rand(K, dim).uniform_(-3, 0).exp()
    tril = 0.9 * th.randn(K, dim, dim)
    L = make_cov(logvar, tril, cholesky=True)

    mvn_dists = th.distributions.MultivariateNormal(mu, scale_tril=L)

    # sample from gmm
    num_samples = 1000
    x, c = sample_gmm(pi, mvn_dists, num_samples)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(x[:, 0], x[:, 1], c=c)
    plt.show()
    plt.savefig(f"gmm_data.png")
    plt.close()

    k = 32
    dataset = GMMDataset(x)
    random_state = 426
    kmeans = KMeans( n_clusters=k, tol=0.000001, n_init="auto", random_state=random_state, max_iter=3000)
    kmeans.fit(dataset)

    centroids = th.tensor(kmeans.cluster_centers_).cpu()


    cov = "diag"
    batch_norm = False
    lr = 5e-1
    batch_size = 32
    epochs = 1000

    loader = DataLoader(dataset, batch_size)

    gmm = GaussianMixture(k, dim, cov=cov, batch_norm=batch_norm)
    gmm.set_centroids(centroids)
    print("MU", gmm.centroids)
    gmm_copilot = FastGaussianMixture(dim, k, device="cpu", dtype=th.float32)
    gmm_copilot.set_centroids(centroids)
    print("Centroids", gmm_copilot.centroids)
    #gmm_copilot.initialize_from_data(x)

    #
    #train(lr, epochs, loader, gmm, "gmm_kaggle") #40/s
    # WORKS train(lr, epochs, loader, gmm_copilot, "gmm_copilot", l_entropy=-0.02, l2_lambda=0.05) # ~50/s
    train(lr, epochs, loader, gmm_copilot, "gmm_copilot", l_entropy=-0.02, l2_lambda=0.05) # ~50/s
        #train(lr, epochs, loader, gmm_copilot, "gmm_copilot_em")
