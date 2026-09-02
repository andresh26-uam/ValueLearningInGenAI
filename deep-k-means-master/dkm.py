#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import math
import argparse

import numpy as np
import torch as th

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans

from utils import cluster_acc
from utils import next_batch

from compgraph import DkmCompGraph


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Deep k-means algorithm"
)

parser.add_argument(
    "-d",
    "--dataset",
    type=str.upper,
    help="Dataset on which DKM will be run "
         "(one of USPS, MNIST, 20NEWS, RCV1)",
    required=True,
)

parser.add_argument(
    "-v",
    "--validation",
    help="Split data into validation and test sets",
    action="store_true",
)

parser.add_argument(
    "-p",
    "--pretrain",
    help="Pretrain the autoencoder and cluster representatives",
    action="store_true",
)

parser.add_argument(
    "-a",
    "--annealing",
    help="Use an annealing scheme for the values of alpha",
    action="store_true",
)

parser.add_argument(
    "-s",
    "--seeded",
    help="Use a fixed seed, different for each run",
    action="store_true",
)

parser.add_argument(
    "-c",
    "--cpu",
    help="Force the program to run on CPU",
    action="store_true",
)

parser.add_argument(
    "-l",
    "--lambda",
    type=float,
    default=1.0,
    dest="lambda_",
    help="Value of the hyperparameter weighing the clustering "
         "loss against the reconstruction loss",
)

parser.add_argument(
    "-e",
    "--p_epochs",
    type=int,
    default=50,
    help="Number of pretraining epochs",
)

parser.add_argument(
    "-f",
    "--f_epochs",
    type=int,
    default=5,
    help="Number of fine-tuning epochs per alpha value",
)

parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=256,
    help="Size of the minibatches used by the optimizer",
)

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

if args.dataset == "USPS":
    import usps_specs as specs
elif args.dataset == "MNIST":
    import mnist_specs as specs
elif args.dataset == "20NEWS":
    import _20news_specs as specs
elif args.dataset == "RCV1":
    import rcv1_specs as specs
else:
    parser.error("Unknown dataset!")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

n_pretrain_epochs = args.p_epochs
n_finetuning_epochs = args.f_epochs
lambda_ = args.lambda_
batch_size = args.batch_size

n_batches = int(
    math.ceil(specs.n_samples / batch_size)
)

validation = args.validation
pretrain = args.pretrain
annealing = args.annealing
seeded = args.seeded

print("Hyperparameters...")
print("lambda =", lambda_)


# ---------------------------------------------------------------------------
# Alpha schedule
# ---------------------------------------------------------------------------

if annealing and not pretrain:

    constant_value = 1
    max_n = 40

    alphas = np.zeros(
        max_n,
        dtype=np.float32,
    )

    alphas[0] = 0.1

    for i in range(1, max_n):
        alphas[i] = (
            2 ** (
                1 / (np.log(i + 1) ** 2)
            )
        ) * alphas[i - 1]

    alphas /= constant_value

elif not annealing and pretrain:

    constant_value = 1
    max_n = 20

    alphas = (
        1000
        * np.ones(
            max_n,
            dtype=np.float32,
        )
    )

    alphas /= constant_value

else:

    parser.error(
        "Run with either annealing (-a) or pretraining (-p), "
        "but not both."
    )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

if validation:

    validation_target = np.asarray([
        specs.target[i]
        for i in specs.validation_indices
    ], dtype=np.int64)

    test_target = np.asarray([
        specs.target[i]
        for i in specs.test_indices
    ], dtype=np.int64)

else:

    target = np.asarray(specs.target, dtype=np.int64)

# You said you don't need GPU, so simply use CPU.
device = th.device("cpu")

data = specs.data
data_th = th.tensor(
                    data,
                    dtype=th.float32,
                    device=device,requires_grad=False
                )
validation_data = np.asarray([
    data[i]
    for i in specs.validation_indices
], dtype=np.float32)
validation_data_th = th.tensor(
                    validation_data,
                    dtype=th.float32,
                    device=device, requires_grad=False
                )
test_data = np.asarray([
    data[i]
    for i in specs.test_indices
], dtype=np.float32)
test_data_th = th.tensor(
                    test_data,
                    dtype=th.float32,
                    device=device, requires_grad=False
                )
# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


print("Using device:", device)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

seeds = [
    8905,
    9129,
    291,
    4012,
    1256,
    6819,
    4678,
    6971,
    1362,
    575,
]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if validation:

    list_validation_acc = []
    list_validation_ari = []
    list_validation_nmi = []

    list_test_acc = []
    list_test_ari = []
    list_test_nmi = []

else:

    list_acc = []
    list_ari = []
    list_nmi = []


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

n_runs = 10

for run in range(n_runs):

    print("Run", run)

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------

    if seeded:

        seed = seeds[run]

        np.random.seed(seed)
        th.manual_seed(seed)

    # ---------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------

    cg = DkmCompGraph(
        [
            specs.dimensions,
            specs.activations,
            specs.names,
        ],
        specs.n_clusters,
        lambda_,
        device=device,
    )

    # ---------------------------------------------------------------
    # Distances
    # ---------------------------------------------------------------

    distances = np.zeros(
        (
            specs.n_clusters,
            specs.n_samples,
        ),
        dtype=np.float32,
    )


    # ===============================================================
    # PRETRAINING
    # ===============================================================

    if pretrain:

        print(
            "Starting autoencoder pretraining..."
        )

        embeddings = np.zeros(
            (
                specs.n_samples,
                specs.embedding_size,
            ),
            dtype=np.float32,
        )

        cg.train()
        ae_loss_ = 10000.0
        for epoch in range(n_pretrain_epochs):

            print(
                "Pretraining step: epoch {}. Loss: {}".format(epoch, ae_loss_)
            )

            for _ in range(n_batches):

                indices, data_batch = next_batch(
                    batch_size,
                    data,
                )

                # ---------------------------------------------------
                # NumPy -> Torch
                # ---------------------------------------------------

                data_batch = th.as_tensor(
                    data_batch,
                    dtype=th.float32,
                    device=device,
                )

                # ---------------------------------------------------
                # Optimization step
                # ---------------------------------------------------

                embedding_, ae_loss_ = (
                    cg.pretrain_step(
                        data_batch
                    )
                )

                embedding_ = (
                    embedding_
                    .cpu()
                    .numpy()
                )

                # ---------------------------------------------------
                # Save embeddings
                # ---------------------------------------------------

                embeddings[
                    indices,
                    :
                ] = embedding_

            
        # -----------------------------------------------------------
        # K-means initialization
        # -----------------------------------------------------------

        print(
            "Running k-means on the learned embeddings..."
        )

        kmeans_model = KMeans(
            n_clusters=specs.n_clusters,
            init="k-means++",
        ).fit(embeddings)

        cg.plot_embedding_space(
            sample_data=validation_data_th,
            sample_labels=validation_target,
            #original_space_centroids=th.as_tensor(kmeans_model.cluster_centers_, device=device, dtype=th.float32),
            save_path=f"plots/embedding_space_run_{run}_epoch_0",
        )

        cg.plot_embedding_space(
            sample_data=validation_data_th,
            sample_labels=kmeans_model.predict(cg.autoencoder(validation_data_th)[0].detach().numpy().astype(np.float32)),
            #original_space_centroids=th.as_tensor(kmeans_model.cluster_centers_, device=device, dtype=th.float32),
            save_path=f"plots/embedding_space_run_{run}_epoch_0_kmeans",
        )
        
        # -----------------------------------------------------------
        # Evaluate initial K-means clustering
        # -----------------------------------------------------------

        if validation:

            validation_cluster_assign = np.asarray([
                kmeans_model.labels_[i]
                for i in specs.validation_indices
            ])

            test_cluster_assign = np.asarray([
                kmeans_model.labels_[i]
                for i in specs.test_indices
            ])

            validation_acc = cluster_acc(
                validation_cluster_assign,
                validation_target,
            )

            print(
                "Validation ACC",
                validation_acc,
            )

            validation_ari = adjusted_rand_score(
                validation_target,
                validation_cluster_assign,
            )

            print(
                "Validation ARI",
                validation_ari,
            )

            validation_nmi = normalized_mutual_info_score(
                validation_target,
                validation_cluster_assign,
            )

            print(
                "Validation NMI",
                validation_nmi,
            )

            test_acc = cluster_acc(
                test_cluster_assign,
                test_target,
            )

            print(
                "Test ACC",
                test_acc,
            )

            test_ari = adjusted_rand_score(
                test_target,
                test_cluster_assign,
            )

            print(
                "Test ARI",
                test_ari,
            )

            test_nmi = normalized_mutual_info_score(
                test_target,
                test_cluster_assign,
            )

            print(
                "Test NMI",
                test_nmi,
            )

        else:

            acc = cluster_acc(
                kmeans_model.labels_,
                target,
            )

            print("ACC", acc)

            ari = adjusted_rand_score(
                target,
                kmeans_model.labels_,
            )

            print("ARI", ari)

            nmi = normalized_mutual_info_score(
                target,
                kmeans_model.labels_,
            )

            print("NMI", nmi)


        # -----------------------------------------------------------
        # Initialize DKM cluster representatives
        # -----------------------------------------------------------

        cg.set_cluster_centers(
            kmeans_model.cluster_centers_
        )


    # ===============================================================
    # DKM TRAINING
    # ===============================================================

    if len(alphas) > 0:

        print(
            "Starting DKM training..."
        )


    cg.train()

    for k, alpha in enumerate(alphas):

        print(
            "Training step: alpha[{}]: {}".format(
                k,
                alpha,
            )
        )

        # -----------------------------------------------------------
        # Loop over epochs
        # -----------------------------------------------------------

        for epoch_ in range(n_finetuning_epochs):

            for _ in range(n_batches):

                indices, data_batch = next_batch(
                    batch_size,
                    data,
                )

                # ---------------------------------------------------
                # NumPy -> Torch
                # ---------------------------------------------------

                data_batch = th.as_tensor(
                    data_batch,
                    dtype=th.float32,
                    device=device,
                )

                alpha_tensor = th.as_tensor(
                    alpha,
                    dtype=data_batch.dtype,
                    device=device,
                )

                # ---------------------------------------------------
                # Train
                # ---------------------------------------------------

                (
                    loss_,
                    stack_dist_,
                    cluster_rep_,
                    ae_loss_,
                    kmeans_loss_,
                ) = cg.train_step(
                    data_batch,
                    alpha_tensor,
                )

                # ---------------------------------------------------
                # Torch -> NumPy
                # ---------------------------------------------------

                stack_dist_ = (
                    stack_dist_
                    .cpu()
                    .numpy()
                )

                # ---------------------------------------------------
                # Save distances
                #
                # stack_dist_: [K, B]
                # ---------------------------------------------------

                distances[
                    :,
                    indices,
                ] = stack_dist_


        # ===========================================================
        # Evaluation
        # ===========================================================

        print_val = 1

        if (
            k % print_val == 0
            or k == len(alphas) - 1
        ):

            loss_value = float(loss_.cpu())
            ae_loss_value = float(ae_loss_.cpu())
            kmeans_loss_value = float(
                kmeans_loss_.cpu()
            )

            print(
                "loss:",
                loss_value,
            )

            print(
                "ae loss:",
                ae_loss_value,
            )

            print(
                "kmeans loss:",
                kmeans_loss_value,
            )

            # -------------------------------------------------------
            # Infer cluster assignments
            #
            # Vectorized replacement for:
            #
            # for i in range(specs.n_samples):
            #     np.argmin(distances[:, i])
            # -------------------------------------------------------

            cluster_assign = np.argmin(
                distances,
                axis=0,
            ).astype(np.int64)


            if validation:

                cg.plot_embedding_space(
                            sample_data=validation_data_th,
                            sample_labels=validation_target,
                            #original_space_centroids=th.as_tensor(kmeans_model.cluster_centers_, device=device, dtype=th.float32),
                            save_path=f"plots/embedding_space_run_{run}_alpha_{k}_{alpha}",
                        )
                cg.plot_embedding_space(
                        sample_data=validation_data_th,
                        sample_labels=cluster_assign,
                        #original_space_centroids=th.as_tensor(kmeans_model.cluster_centers_, device=device, dtype=th.float32),
                        save_path=f"plots/embedding_space_run_{run}_alpha_{k}_{alpha}_cluster_assign",
                    )

                validation_cluster_assign = (
                    cluster_assign[
                        specs.validation_indices
                    ]
                )

                test_cluster_assign = (
                    cluster_assign[
                        specs.test_indices
                    ]
                )

                # ---------------------------------------------------
                # Validation
                # ---------------------------------------------------

                validation_acc = cluster_acc(
                    validation_cluster_assign,
                    validation_target,
                )

                print(
                    "Validation ACC",
                    validation_acc,
                )

                validation_ari = adjusted_rand_score(
                    validation_target,
                    validation_cluster_assign,
                )

                print(
                    "Validation ARI",
                    validation_ari,
                )

                validation_nmi = normalized_mutual_info_score(
                    validation_target,
                    validation_cluster_assign,
                )

                print(
                    "Validation NMI",
                    validation_nmi,
                )

                # ---------------------------------------------------
                # Test
                # ---------------------------------------------------

                test_acc = cluster_acc(
                    test_cluster_assign,
                    test_target,
                )

                print(
                    "Test ACC",
                    test_acc,
                )

                test_ari = adjusted_rand_score(
                    test_target,
                    test_cluster_assign,
                )

                print(
                    "Test ARI",
                    test_ari,
                )

                test_nmi = normalized_mutual_info_score(
                    test_target,
                    test_cluster_assign,
                )

                print(
                    "Test NMI",
                    test_nmi,
                )

            else:

                acc = cluster_acc(
                    cluster_assign,
                    target,
                )

                print(
                    "ACC",
                    acc,
                )

                ari = adjusted_rand_score(
                    target,
                    cluster_assign,
                )

                print(
                    "ARI",
                    ari,
                )

                nmi = normalized_mutual_info_score(
                    target,
                    cluster_assign,
                )

                print(
                    "NMI",
                    nmi,
                )


    # ===============================================================
    # Record results
    # ===============================================================

    if validation:

        list_validation_acc.append(
            validation_acc
        )

        list_validation_ari.append(
            validation_ari
        )

        list_validation_nmi.append(
            validation_nmi
        )

        list_test_acc.append(
            test_acc
        )

        list_test_ari.append(
            test_ari
        )

        list_test_nmi.append(
            test_nmi
        )

    else:

        list_acc.append(acc)
        list_ari.append(ari)
        list_nmi.append(nmi)


# ---------------------------------------------------------------------------
# Final statistics
# ---------------------------------------------------------------------------

if validation:

    list_validation_acc = np.asarray(
        list_validation_acc
    )

    print(
        "Average validation ACC: "
        "{:.3f} +/- {:.3f}".format(
            np.mean(list_validation_acc),
            np.std(list_validation_acc),
        )
    )

    list_validation_ari = np.asarray(
        list_validation_ari
    )

    print(
        "Average validation ARI: "
        "{:.3f} +/- {:.3f}".format(
            np.mean(list_validation_ari),
            np.std(list_validation_ari),
        )
    )

    list_validation_nmi = np.asarray(
        list_validation_nmi
    )

    print(
        "Average validation NMI: "
        "{:.3f} +/- {:.3f}".format(
            np.mean(list_validation_nmi),
            np.std(list_validation_nmi),
        )
    )

    list_test_acc = np.asarray(
        list_test_acc
    )

    print(
        "Average test ACC: "
        "{:.3f} +/- {:.3f}".format(
            np.mean(list_test_acc),
            np.std(list_test_acc),
        )
    )

    list_test_ari = np.asarray(
        list_test_ari
    )

    print(
        "Average test ARI: "
        "{:.3f} +/- {:.3f}".format(
            np.mean(list_test_ari),
            np.std(list_test_ari),
        )
    )

    list_test_nmi = np.asarray(
        list_test_nmi
    )

    print(
        "Average test NMI: "
        "{:.3f} +/- {:.3f}".format(
            np.mean(list_test_nmi),
            np.std(list_test_nmi),
        )
    )

else:

    list_acc = np.asarray(list_acc)

    print(
        "Average ACC: "
        "{:.3f} +/- {:.3f}".format(
            np.mean(list_acc),
            np.std(list_acc),
        )
    )

    list_ari = np.asarray(list_ari)

    print(
        "Average ARI: "
        "{:.3f} +/- {:.3f}".format(
            np.mean(list_ari),
            np.std(list_ari),
        )
    )

    list_nmi = np.asarray(list_nmi)

    print(
        "Average NMI: "
        "{:.3f} +/- {:.3f}".format(
            np.mean(list_nmi),
            np.std(list_nmi),
        )
    )