from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import random
from typing import List

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from datasets import Dataset as HFDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from vsllib.defines import SYNTH_PROCESSED_PATH, NO_RATING_MASK, OASST_PROCESSED_PATH, OASSTFL_PROCESSED_PATH, SYNTH_PROCESSED_PATH, VALUES_OASST, VALUES_OASST_ORIG, DOWNLOADED_DATASETS_PATH, DatasetNames, save_processeddataset

load_dotenv()


N_FEATURES = 10
CONTEXT_FEATURES = 6
N_VALUES = 3
N_CONTEXTS = 2000
EXAMPLES_PER_CONTEXT = 10
VALUE_SYSTEM_VARIETY = 5


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


class GMMDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np
import matplotlib.pyplot as plt

def plot(samples=None, predicted_cluster=None, gmm=None, history=None, file_name="gmm_plot"):
    # ---- Training history plot ----

    if history is not None:
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

    if samples is None:
        predicted_cluster = None
        with torch.no_grad():
            if hasattr(gmm, "sample_with_predicted_cluster"):
                x_sample, _, predicted_cluster = gmm.sample_with_predicted_cluster(num_samples)
            else:
                x_sample, _ = gmm.sample(num_samples)
    else:
        x_sample = th.as_tensor(samples)
        predicted_cluster = th.as_tensor(predicted_cluster)

    x_sample_np = x_sample.cpu().numpy() if hasattr(x_sample, "cpu") else x_sample

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # ---- Left: real data ----
    ax[0].scatter(x_sample[:, 0], x_sample[:, 1])
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
            print(cmap(k % 10))
            ax[1].scatter(
                x_sample_np[idx, 0],
                x_sample_np[idx, 1],
                color=cmap(k % 10),
                #marker=markers[k % len(markers)],
                label=f"Cluster {k}",
                alpha=0.8,
                edgecolors='black',
                linewidths=0.3
            )

        # ---- Plot centroids ----
        if gmm is not None and hasattr(gmm, "centroids"):
            centroids = gmm.centroids 
        else:
            centroids = []
            for k in used_clusters:
                idx = clusters == k
                centroids.append(np.mean(x_sample_np[idx], axis=0))
            centroids = torch.tensor(np.array(centroids), dtype=torch.float32)   
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




def sample_example_profiles(profile_variety, n_values=3) -> List:
    ratios = np.linspace(0, 1, profile_variety)

    basic_profiles = [tuple(t) for t in np.eye(n_values, dtype=np.float32).tolist()]
    if n_values < 1:
        raise ValueError('Need more values: n_values must be bigger than 0')
    if n_values == 1:
        profile_set = list(ratios)
    if n_values == 2:
        profile_set = [(1.0-ratio, ratio) for ratio in ratios]
    if n_values == 3:
        profile_combinations = [set(itertools.permutations((ratios[i], ratios[j], ratios[-i-j-1])))
                                for i in range(len(ratios)) for j in range(i, (len(ratios)-i+1)//2)]
    else:
        def recursFind(N, nc=3, i=0, t=0, p=[]):
            if nc == 1:
                # No need to explore, last value is N-t
                if N-t >= i:
                    yield p+[N-t]
                else:
                    # p+[N-t] is a solution, but it has already been given in another order
                    pass
            elif i*nc+t > N:
                # impossible to find nc values>=i (there are some <i. But that would yield to already given solutions)
                return
            else:
                for j in range(i, N):
                    yield from recursFind(N, nc-1, j, t+j, p+[j])

        profile_combinations = [set(itertools.permutations(
            ratios[i] for i in idx)) for idx in recursFind(len(ratios)-1, n_values)]

    if n_values >= 3:
        profile_set = list(set(tuple(
            float(f"{a_i:0.3f}") for a_i in a) for l in profile_combinations for a in l))
        [profile_set.remove(pr) for pr in basic_profiles]
        for pr in reversed(basic_profiles):
            profile_set.insert(0, pr)

    a = np.array(profile_set, dtype=np.dtype(
        [(f'{i}', float) for i in range(n_values)]))
    sortedprofiles = a[np.argsort(
        a, axis=-1, order=tuple([f'{i}' for i in range(n_values)]), )]
    profile_set = list(tuple(t) for t in sortedprofiles.tolist())

    profile_set = [tuple(round(num, 2) for num in t) for t in profile_set]

    return profile_set

def process_dataset() -> tuple[pd.DataFrame, Path]:
    random.seed(42)
    """Download the ready trees file and load it into a pandas dataframe."""
    value_systems = sample_example_profiles(VALUE_SYSTEM_VARIETY, n_values=N_VALUES)
    features = np.random.randn(N_CONTEXTS*EXAMPLES_PER_CONTEXT, N_FEATURES, dtype=np.float32)
    context_features = np.random.randn(N_CONTEXTS, CONTEXT_FEATURES, dtype=np.float32)

    to_norm_features = ["hh_inc_abs", ("tt1", "tt2"), ("hw1", "hw2"), ("ch1", "ch2"), ("tc1", "tc2")]
    for f in to_norm_features:
        if isinstance(f, tuple):
            f1, f2 = f[0], f[1]
            all_data = np.concatenate([full_data[f1].to_numpy(), full_data[f2].to_numpy()])
            for f_ in (f1,f2):
                full_data[f_ + "_NORM"] = ((full_data[f_])-np.mean(all_data))/np.std(all_data)
        else:
            full_data[f + "_NORM"] = ((full_data[f])-full_data[f].mean())/full_data[f].std()
    
    to_scale_features = ["hh_inc_abs", ("tt1", "tt2"), ("hw1", "hw2"), ("ch1", "ch2"), ("tc1", "tc2")]
    for f in to_scale_features:
        if isinstance(f, tuple):
            f1, f2 = f[0], f[1]
            all_data = np.concatenate([full_data[f1].to_numpy(), full_data[f2].to_numpy()])
            for f_ in (f1,f2):
                full_data[f_ + "_SCALED"] = ((full_data[f_]))/max(all_data)
        else:
            full_data[f + "_SCALED"] = full_data[f]/max(full_data[f].to_numpy())

    print(full_data.head(5))

    rows = []
    for i, line in tqdm(full_data.iterrows()):
        rows.append(process_line(line,i))
        
    pdrows = pd.DataFrame(rows)
    pdrows = pdrows.reset_index(drop=True)
     # ID,choice,tt1,tc1,hw1,ch1,tt2,tc2,hw2,ch2,hh_inc_abs,car_availability,commute,shopping,business,leisure
    # shuffle the rows to avoid any ordering bias:
    pdrows_sh = pdrows.sample(frac=1, random_state=42).reset_index(drop=True)
    assert len(pdrows_sh) == len(pdrows), f"Shuffled rows length {len(pdrows_sh)} does not match original rows length {len(pdrows)}"
    return pdrows_sh

if __name__ == "__main__":
    seed = 45376
    th.manual_seed(seed)

    value_systems = sample_example_profiles(VALUE_SYSTEM_VARIETY, n_values=N_VALUES)
    # make gmm
    K = len(value_systems)
    dim = CONTEXT_FEATURES
    pi = th.softmax(th.randn(K), dim=-1)
    mu = th.rand(K, dim).uniform_(-10, 10)
    logvar = th.rand(K, dim).uniform_(-3, 0).exp()
    tril = 0.5 * th.randn(K, dim, dim)
    L = make_cov(logvar, tril, cholesky=True)

    mvn_dists = th.distributions.MultivariateNormal(mu, scale_tril=L)

    # sample from gmm
    num_samples = N_CONTEXTS
    x, c = sample_gmm(pi, mvn_dists, num_samples)

    pca = TSNE(n_components=2)
    x2d = pca.fit_transform(x)
    
    plot(samples=x2d, predicted_cluster=c,gmm=None, history=None, file_name="gmm_data_synth")

    
    contexts = x
    features_1 = th.rand(N_CONTEXTS*EXAMPLES_PER_CONTEXT, N_FEATURES).uniform_(-10, 10)
    features_2 = th.rand(N_CONTEXTS*EXAMPLES_PER_CONTEXT, N_FEATURES).uniform_(-10, 10)

    grounding_weights = th.rand(N_FEATURES, N_VALUES).uniform_(-2, 2)

    value_systems = th.tensor(value_systems, dtype=th.float32)

    rows = []
    ctx_counter = -1
    for iex, (example_1, example_2) in enumerate(zip(features_1, features_2)):
        row = {}
        if iex % EXAMPLES_PER_CONTEXT == 0:
            ctx_counter += 1
        row["context_features"] = contexts[ctx_counter].numpy()
        row["context"] = ctx_counter
        row["grounding_features_1"] = example_1.numpy()
        row["grounding_features_2"] = example_2.numpy()
        row["state"] = iex
        row["user_id"] = ctx_counter
        row["action1"] = 1
        row["action2"] = 2

        row["grounding_1"] = example_1.numpy()
        row["grounding_2"] = example_2.numpy()

        value_ratings_1 = example_1.numpy().dot(grounding_weights.numpy())
        value_ratings_2 = example_2.numpy().dot(grounding_weights.numpy())
        for value_name in range(N_VALUES):
            row[f"value_syn{value_name}_1"] = float(value_ratings_1[value_name])
            row[f"value_syn{value_name}_2"] = float(value_ratings_2[value_name])

        row["vs_id"] = c[ctx_counter].item()
        row["ctx_id"] = ctx_counter
        row["vs_real"] = value_systems[c[ctx_counter]].numpy()
        row["score1"] = float(value_ratings_1.dot(value_systems[c[ctx_counter]].numpy()))
        row["score2"] = float(value_ratings_2.dot(value_systems[c[ctx_counter]].numpy()))
        row["choice"] = 1 if row["score1"] > row["score2"] else 2


        rows.append(row)
    
    pd_dataset = pd.DataFrame(rows)
    # Print the distribution of choices in each value system:
    choice_distribution = pd_dataset.groupby('vs_id')['choice'].value_counts(normalize=True).unstack(fill_value=0)
    print("Choice distribution by value system:")
    print(choice_distribution)

    val_indices = pd_dataset.sample(frac=0.1, random_state=42).index.tolist()
    test_indices = np.random.choice(pd_dataset.index.difference(val_indices), len(val_indices)).tolist()

    hf_dataset = HFDataset.from_pandas(pd_dataset)

    save_processeddataset(SYNTH_PROCESSED_PATH, hf_dataset, val_indices, test_indices)
    total_rows = len(hf_dataset)
    print(f"Total merged rows: {total_rows}")
    print(f"Validation rows: {len(val_indices)}")
    print(f"Test rows: {len(test_indices)}")