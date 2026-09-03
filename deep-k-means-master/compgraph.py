#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import os

import torch as th
import torch.nn as nn

import numpy as np

from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
                
# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_activation(activation):
    """
    Convert an activation specification into a PyTorch activation module.

    Supports:
        None
        strings: "relu", "sigmoid", "tanh", "linear", "identity"
        PyTorch nn.Module instances
        PyTorch callable functions
    """
    if activation is None:
        return nn.Identity()

    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, str):
        name = activation.lower()

        if name in ("none", "linear", "identity"):
            return nn.Identity()
        elif name == "relu":
            return nn.ReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "softmax":
            return nn.Softmax(dim=-1)
        elif name == "log_softmax":
            return nn.LogSoftmax(dim=-1)

        raise ValueError(f"Unknown activation: {activation}")

    raise TypeError(
        f"Unsupported activation type: {type(activation)}"
    )


def fc_layers(input, specs):
    """
    Build a fully-connected network.

    specs:
        [dimensions, activations, names]

    dimensions:
        List of layer output dimensions.

    activations:
        List of activation specifications.

    names:
        List of layer names.
    """
    dimensions, activations, names = specs

    layers = []

    input_size = None

    # We need the input dimension. In the original TensorFlow code this was
    # inferred dynamically by tf.layers.dense. Here it is supplied by the
    # first dimension in the complete autoencoder architecture.
    #
    # This function is therefore mainly kept as a compatibility helper.
    for i, (dimension, activation, name) in enumerate(
        zip(dimensions, activations, names)
    ):
        if input_size is None:
            raise ValueError(
                "fc_layers requires an input size. "
                "Use AutoEncoder instead."
            )

        layers.append(
            nn.Linear(input_size, dimension)
        )

        layers.append(_get_activation(activation))

        input_size = dimension

    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    """
    PyTorch implementation of the TensorFlow autoencoder.
    """

    def __init__(self, specs):
        super().__init__()

        dimensions, activations, names = specs

        self.dimensions = dimensions
        self.activations = activations
        self.names = names

        self.mid_ind = len(dimensions) // 2

        layers = []

        # Input dimension comes from the last dimension of the original
        # TensorFlow specification.
        input_size = dimensions[-1]

        for dimension, activation, name in zip(
            dimensions,
            activations,
            names,
        ):
            layers.append(
                (
                    name,
                    nn.Linear(input_size, dimension)
                )
            )

            layers.append(
                (
                    f"{name}_activation",
                    _get_activation(activation)
                )
            )

            input_size = dimension

        self.network = nn.Sequential(
            nn.ModuleDict(layers)
        )

        # The above ModuleDict is inconvenient for sequential execution,
        # so build an ordinary Sequential as well.
        modules = []

        input_size = dimensions[-1]

        for dimension, activation in zip(
            dimensions,
            activations,
        ):
            modules.append(
                nn.Linear(input_size, dimension)
            )
            modules.append(
                _get_activation(activation)
            )

            input_size = dimension

        self.network = nn.Sequential(*modules)

        # Index of the embedding layer.
        self.embedding_layer_index = 2 * (self.mid_ind - 1)

    def encode(self, x):
        """
        Run the encoder only.
        """
        for i in range(self.mid_ind):
            linear = self.network[2 * i]
            activation = self.network[2 * i + 1]

            x = activation(linear(x))

        return x

    def decode(self, embedding):
        """
        Run the decoder.
        """
        x = embedding

        for i in range(self.mid_ind, len(self.dimensions)):
            network_index = 2 * i

            linear = self.network[network_index]
            activation = self.network[network_index + 1]

            x = activation(linear(x))

        return x

    def forward(self, x):
        embedding = self.encode(x)
        output = self.decode(embedding)

        return embedding, output


def autoencoder(input, specs):
    """
    Compatibility helper.

    Returns a PyTorch AutoEncoder and its forward result.
    """
    model = AutoEncoder(specs)
    return model(input)


def f_func(x, y):
    """
    Squared Euclidean distance.

    x: [B, D]
    y: [K, D] or [1, D]

    Returns:
        [B, K] when y is [K, D]
        [B] when y is [1, D]
    """
    return th.sum((x - y) ** 2, dim=-1)


def g_func(x, y):
    """
    Squared reconstruction error.
    """
    return th.sum((x - y) ** 2, dim=1)


class DkmCompGraph(nn.Module):
    """
    PyTorch implementation of Deep K-Means.

    This replaces the TensorFlow computation graph, placeholders,
    Session.run(), tf.assign(), and AdamOptimizer.
    """

    def __init__(
        self,
        ae_specs,
        num_contexts: int,
        val_lambda,
        lr=1e-3,
        device="cpu",
    ):
        super().__init__()

        dimensions, activations, names = ae_specs

        input_size = dimensions[-1]
        latent_dim = dimensions[
            int((len(dimensions) - 1) / 2)
        ]

        self.num_contexts = num_contexts
        self.val_lambda = val_lambda
        self.latent_dim = latent_dim

        self.device = th.device(device)

        # ---------------------------------------------------------------
        # Autoencoder
        # ---------------------------------------------------------------

        self.autoencoder = AutoEncoder(ae_specs)

        # ---------------------------------------------------------------
        # Cluster representatives
        # ---------------------------------------------------------------

        self.latent_centroids = nn.Parameter(
            th.empty(
                num_contexts,
                latent_dim,
                dtype=th.float32,
            )
        )

        nn.init.uniform_(
            self.latent_centroids,
            -1.0,
            1.0,
        )

        # ---------------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------------

        self.optimizer = th.optim.Adam(
            self.parameters(),
            lr=lr,
        )

        self.to(self.device)
        self.vae_type = "ae"

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------
    def plot_embedding_space(self, sample_data: th.Tensor, sample_labels: th.Tensor=None, original_space_centroids: th.Tensor=None, save_path: str = None):
            
            with th.no_grad(): 
                encoder_output, output = self.autoencoder(sample_data)
                if original_space_centroids is not None:
                    # Project original space centroids to latent space
                    original_space_centroids = original_space_centroids.to(sample_data.device, sample_data.dtype)
                    latent_original_centroids, _ = self.autoencoder(original_space_centroids)
                    
                if self.vae_type == "vae":
                    mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                        
                    std = th.exp(0.5 * log_var)
                    std_m = th.mean(std).item()
    
                    z_all = []
                    for r in range(50):
                        z, eps = self._sample_gauss(mu, std)
                        z = z.cpu().numpy()
                        z_all.append(z)
                    z = np.concatenate(z_all, axis=0)
                else:
                    z = encoder_output.cpu().numpy()
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
                    if self.latent_dim > 2:
                        umap_original_centroids = umap.transform(latent_original_centroids)
                    else:
                        umap_original_centroids = latent_original_centroids.detach().cpu().numpy()
                plt.figure(figsize=(8, 6))
                plt.scatter(z_umap[:, 0], z_umap[:, 1], c=sample_labels, cmap='viridis', s=5, alpha=0.8)
                # plot centroids the same color as the closest points in the latent space:
                plt.scatter(z_umap_centroids[:, 0], z_umap_centroids[:, 1], c=list(range(self.num_contexts)), cmap='viridis', marker='X', s=100, label='Centroids')
                if original_space_centroids is not None:
                    plt.scatter(umap_original_centroids[:, 0], umap_original_centroids[:, 1], c='blue', marker='o', s=100, label='Original Centroids')
                #plt.scatter(z_umap_centroids[:, 0], z_umap_centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
                plt.title('Latent Space UMAP Projection')
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')
                
                plt.legend()
                if save_path is not None:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path + ".png", dpi=50)
                plt.show() 
    def forward(self, x, alpha):
        """
        Compute the DKM forward pass.

        Returns:
            loss
            stack_dist
            embedding
            output
            ae_loss
            kmeans_loss
        """

        embedding, output = self.autoencoder(x)

        # Reconstruction error
        rec_error = g_func(
            x,
            output,
        )

        ae_loss = rec_error.mean()

        # ---------------------------------------------------------------
        # Distances from every embedding to every cluster representative
        #
        # embedding:    [B, D]
        # latent_centroids:  [K, D]
        #
        # stack_dist:   [K, B]
        # ---------------------------------------------------------------

        stack_dist = (
            (embedding[:, None, :] - self.latent_centroids[None, :, :])
            .pow(2)
            .sum(dim=-1)
            .transpose(0, 1)
        )

        # [B]
        min_dist = stack_dist.min(dim=0).values
        
        # ---------------------------------------------------------------
        # Soft assignment
        #
        # Original:
        #
        # exp(-alpha * (distance - min_distance))
        #
        # The subtraction prevents numerical underflow.
        # ---------------------------------------------------------------

        stack_exp = th.exp(
            -alpha * (
                stack_dist - min_dist.unsqueeze(0)
            )
        )

        sum_exponentials = stack_exp.sum(dim=0)

        softmax = (
            stack_exp /
            sum_exponentials.unsqueeze(0)
        )

        weighted_dist = stack_dist * softmax

        kmeans_loss = weighted_dist.sum(dim=0).mean()

        loss = (
            ae_loss +
            self.val_lambda * kmeans_loss
        )

        return (
            loss,
            stack_dist,
            embedding,
            output,
            ae_loss,
            kmeans_loss,
        )

    # -------------------------------------------------------------------
    # Pretraining
    # -------------------------------------------------------------------

    def pretrain_step(self, x):
        """
        One autoencoder optimization step.

        Equivalent to:

            sess.run(
                (cg.pretrain_op, cg.embedding, cg.ae_loss),
                feed_dict={cg.input: data_batch}
            )
        """

        self.train()

        self.optimizer.zero_grad(set_to_none=True)

        embedding, output = self.autoencoder(x)

        rec_error = g_func(
            x,
            output,
        )

        ae_loss = rec_error.mean()

        ae_loss.backward()

        # Only autoencoder parameters should be optimized during
        # pretraining. The cluster representatives have no gradient
        # because they aren't involved in this computation.
        self.optimizer.step()

        return (
            embedding.detach(),
            ae_loss.detach(),
        )

    # -------------------------------------------------------------------
    # Full DKM training
    # -------------------------------------------------------------------

    def train_step(self, x, alpha):
        """
        One DKM optimization step.

        Equivalent to the TensorFlow train_op + sess.run().
        """

        self.train()

        self.optimizer.zero_grad(set_to_none=True)

        (
            loss,
            stack_dist,
            embedding,
            output,
            ae_loss,
            kmeans_loss,
        ) = self.forward(
            x,
            alpha,
        )

        loss.backward()

        self.optimizer.step()

        return (
            loss.detach(),
            stack_dist.detach(),
            self.latent_centroids.detach(),
            ae_loss.detach(),
            kmeans_loss.detach(),
        )

    # -------------------------------------------------------------------
    # Initialize cluster centers from sklearn KMeans
    # -------------------------------------------------------------------

    @th.no_grad()
    def set_cluster_centers(self, centers):
        """
        Replace TensorFlow:

            sess.run(
                tf.assign(
                    cg.latent_centroids,
                    kmeans_model.cluster_centers_
                )
            )
        """

        centers = th.as_tensor(
            centers,
            dtype=self.latent_centroids.dtype,
            device=self.latent_centroids.device,
        )

        self.latent_centroids.copy_(centers)


class AeCompGraph(nn.Module):
    """
    PyTorch equivalent of AeCompGraph.
    """

    def __init__(
        self,
        ae_specs,
        lr=1e-3,
        device="cpu",
    ):
        super().__init__()

        self.autoencoder = AutoEncoder(ae_specs)

        self.optimizer = th.optim.Adam(
            self.autoencoder.parameters(),
            lr=lr,
        )

        self.device = th.device(device)

        self.to(self.device)

    def forward(self, x):
        return self.autoencoder(x)

    def train_step(self, x):
        self.train()

        self.optimizer.zero_grad(set_to_none=True)

        embedding, output = self.autoencoder(x)

        rec_error = g_func(
            x,
            output,
        )

        loss = rec_error.mean()

        loss.backward()

        self.optimizer.step()

        return (
            loss.detach(),
            embedding.detach(),
        )


class DcnCompGraph(nn.Module):
    """
    PyTorch implementation of the DCN model.

    This class is included for completeness, although your supplied
    main script uses DkmCompGraph rather than DcnCompGraph.
    """

    def __init__(
        self,
        ae_specs,
        n_clusters,
        batch_size,
        n_samples,
        val_lambda,
        lr=1e-3,
        device="cpu",
    ):
        super().__init__()

        dimensions, activations, names = ae_specs

        embedding_size = dimensions[
            int((len(dimensions) - 1) / 2)
        ]

        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.val_lambda = val_lambda

        self.autoencoder = AutoEncoder(ae_specs)

        self.latent_centroids = nn.Parameter(
            th.empty(
                n_clusters,
                embedding_size,
                dtype=th.float32,
            )
        )

        nn.init.uniform_(
            self.latent_centroids,
            -1.0,
            1.0,
        )

        # Original TensorFlow code initializes all counts to 100.
        self.register_buffer(
            "count",
            th.full(
                (n_clusters,),
                100.0,
                dtype=th.float32,
            ),
        )

        self.cluster_assign = th.randint(
            0,
            n_clusters,
            (n_samples,),
            dtype=th.long,
        )

        self.optimizer = th.optim.Adam(
            self.parameters(),
            lr=lr,
        )

        self.to(device)

    def forward(self, x, indices):
        embedding, output = self.autoencoder(x)

        rec_error = g_func(
            x,
            output,
        )

        ae_loss = rec_error.mean()

        # Cluster assignment for each element in the batch.
        assignments = self.cluster_assign[
            indices
        ]

        batch_latent_centroids = self.latent_centroids[
            assignments
        ]

        clustering_error = f_func(
            embedding,
            batch_latent_centroids,
        )

        kmeans_loss = clustering_error.mean()

        loss = (
            ae_loss +
            self.val_lambda * kmeans_loss
        )

        return (
            loss,
            embedding,
            output,
            ae_loss,
            kmeans_loss,
        )

    def pretrain_step(self, x):
        self.train()

        self.optimizer.zero_grad(set_to_none=True)

        embedding, output = self.autoencoder(x)

        loss = g_func(
            x,
            output,
        ).mean()

        loss.backward()
        self.optimizer.step()

        return (
            embedding.detach(),
            loss.detach(),
        )

    @th.no_grad()
    def update_assignments(self, embedding, indices):
        """
        Equivalent to the TensorFlow cluster assignment update.

        All assignments are updated vectorially rather than in a
        Python loop over the batch.
        """

        distances = (
            (
                embedding[:, None, :]
                - self.latent_centroids[None, :, :]
            )
            .pow(2)
            .sum(dim=-1)
        )

        new_assignments = distances.argmin(dim=1)

        self.cluster_assign[indices] = new_assignments

    @th.no_grad()
    def update_latent_centroidsresentatives(
        self,
        embedding,
        indices,
    ):
        """
        Vectorized online update of cluster representatives.
        """

        assignments = self.cluster_assign[
            indices
        ]

        for k in range(self.n_clusters):

            mask = assignments == k

            if not mask.any():
                continue

            points = embedding[mask]

            n = points.shape[0]

            old_count = self.count[k]

            new_count = old_count + n

            # Equivalent to applying the original online updates
            # to all points in the batch, expressed as a batch mean.
            batch_mean = points.mean(dim=0)

            self.latent_centroids[k].mul_(
                old_count / new_count
            )

            self.latent_centroids[k].add_(
                batch_mean * (n / new_count)
            )

            self.count[k] = new_count