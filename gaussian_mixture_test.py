
import math

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import torch as th
import numpy as np
import torch

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


    def forward(self, x):

        """
        x:
            [...,D]

        returns:
            log p(x)
            [...]
        """

        if x.ndim == 1:
            x = x.unsqueeze(0)


        # x:
        #   [B,1,D]
        #
        # mu:
        #   [1,K,D]
        #

        diff = (
            x.unsqueeze(-2)
            -
            self.centroids
        )


        L = self.get_scale_tril()


        #
        # Solve:
        #
        # L z = x-mu
        #
        # diff:
        #   [B,K,D]
        #
        # L:
        #   [K,D,D]
        #

        z = th.linalg.solve_triangular(
            L.unsqueeze(0),
            diff.unsqueeze(-1),
            upper=False
        ).squeeze(-1)


        mahalanobis = (
            z.pow(2)
            .sum(dim=-1)
        )


        log_det = (
            2.0 *
            th.log(
                th.diagonal(L, dim1=-2, dim2=-1)
            )
            .sum(dim=-1)
        )


        normalization = (
            self.input_size *
            th.log(
                th.tensor(
                    2.0 * th.pi,
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
                log_det.unsqueeze(0)
                +
                normalization
            )
        )


        log_weights = th.log_softmax(
            self.mixture_logits,
            dim=0
        )


        return th.logsumexp(
            component_log_prob
            +
            log_weights,
            dim=-1
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
    def forwardworks(self, context_embeddings: th.Tensor):

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


        # add mixture probability
        component_log_probs = (
            component_log_probs
            + log_weights
        )


        # log sum_k pi_k N(x|mu_k,Sigma_k)
        log_prob = th.logsumexp(
            component_log_probs,
            dim=-1
        )


        return log_prob



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
        device=None,
        dtype=None
    ):
        super().__init__()

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


        return th.logsumexp(
            component_log_prob
            +
            th.log_softmax(self.logits, dim=0),
            dim=1
        )

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

def train(lr, epochs, loader, gmm, file_name):
    optimizer = torch.optim.Adam(gmm.parameters(), lr=lr)
    

    start_time = time.time()
    history = []
    bar = trange(epochs)
    for e in bar:
        epoch_loss, epoch_n = 0, 0
        for i, data_batch in enumerate(loader):
            logp = gmm.forward(data_batch)
            loss = -torch.mean(logp)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += -logp.sum().data.item()
            epoch_n += len(data_batch)
        
        epoch_loss /= epoch_n
        history.append(epoch_loss)

        bar.set_postfix(
            loss='{:.2f}'.format(epoch_loss), 
            time="{:.2f}".format(time.time() - start_time)
        )
    end_time = time.time()
    print(f"FINISHED IN {end_time - start_time:.2f} seconds")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history)
    plt.show()
    plt.savefig(file_name+"_history.png")
    plt.close()

    num_samples = 1000
    with torch.no_grad():
        x_sample, logp = gmm.sample(num_samples)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax[0].scatter(x[:, 0], x[:, 1])
    ax[0].set_title("Data samples")
    ax[1].scatter(x_sample[:, 0], x_sample[:, 1])
    ax[1].set_title("Model samples")
    plt.tight_layout()
    plt.show()
    plt.savefig(file_name+"_samples.png")
    plt.close()

if __name__ == "__main__":
    seed = 34254
    th.manual_seed(seed)

    # make gmm
    K = 3
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

    k = 4
    dataset = GMMDataset(x)
    random_state = 42
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    kmeans.fit(dataset)

    centroids = th.tensor(kmeans.cluster_centers_).cpu()


    cov = "diag"
    batch_norm = False
    lr = 1e-2
    batch_size = 128
    epochs = 1000

    loader = DataLoader(dataset, batch_size)

    gmm = GaussianMixture(k, dim, cov=cov, batch_norm=batch_norm)
    gmm.set_centroids(centroids)
    print("MU", gmm.centroids)
    gmm_copilot = FastGaussianMixture(dim, k, device="cpu", dtype=th.float32)
    gmm_copilot.set_centroids(centroids)
    print("Centroids", gmm_copilot.centroids)
    #gmm_copilot.initialize_from_data(x)

    #train(lr, epochs, loader, gmm, "gmm_kaggle")
    train(lr, epochs, loader, gmm_copilot, "gmm_copilot")
