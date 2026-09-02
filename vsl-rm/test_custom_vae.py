#print current folder:
import os

import numpy as np

from vsllib.model_utils import CustomDecoder, CustomEncoder, CustomVAE, CustomVAEConfig, MORMForClassificationConfig, TrainingArguments

from vsllib.utils import seed_everything
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans

import torch as th


def read_list(file_name, type='int'):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    if type == 'str':
        array = np.asarray([l.strip() for l in lines])
        return array
    elif type == 'int':
        array = np.asarray([int(l.strip()) for l in lines])
        return array
    else:
        print("Unknown type")
        return None
    
# Fetch the dataset
dataset = fetch_openml("mnist_784", version=1, as_frame=False)
print("Dataset MNIST loaded...")
data = dataset.data
target = np.array(dataset.target, dtype=np.long)
n_samples = data.shape[0] # Number of samples in the dataset
n_clusters = 10 # Number of clusters to obtain

# Pre-process the dataset
data = data / 255.0 # Normalize the levels of grey between 0 and 1

# Get the split between training/test set and validation set
validation_indices = read_list("deep-k-means-master/split/mnist/validation")
# Pick a small random subset of the validation set for testing purposes
validation_indices = np.random.choice(validation_indices, size=int(len(validation_indices)/10), replace=False)
test_indices = read_list("deep-k-means-master/split/mnist/test")[0:2*len(validation_indices)]

# Auto-encoder architecture
input_size = data.shape[1]
hidden_size = 1000
n_hidden_layers = 3

embedding_size = n_clusters
"""activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, # Encoder layer activations
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None] # Decoder layer activations"""
activation = "ReLU"
"""= ["relu", "relu", "relu", None, # Encoder layer activations
               "relu", "relu", "relu", None] # Decoder layer activations"""
names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
         'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names

vae_config = CustomVAEConfig(
    input_dim=(input_size,),
    vae_latent_dim=embedding_size,
    vae_n_hidden_layers=n_hidden_layers,
    vae_hidden_dim=hidden_size,
    vae_similarity= "euclidean",
    vae_final_encoder_layer_activation="none",
    vae_resampling_iterations=1,
    vae_reconstruction_loss="mse",
    vae_dropout=0.0,
    vae_layer_activation=activation,   
    vae_lambda_clustering=1.0,
    vae_type="ae",
    vae_initial_temperature=1.0,
)

num_contexts = n_clusters

detach_context_selection_for_value_system_selection = False
direct_context_to_vs_relation = True

config = MORMForClassificationConfig(lr_context=1e-3, lr_lambda=0.01, direct_context_to_vs_relation=direct_context_to_vs_relation, max_contexts=num_contexts, max_value_systems=num_contexts, lr_grounding=0.001)
args = TrainingArguments(
    output_dir="my_model",
    learning_rate=1e-3,
    per_device_train_batch_size=256,
    num_train_epochs=20,
    logging_dir="logs",
    logging_steps=10,
    seed=42,
)

device = "cuda" if th.cuda.is_available() else "cpu"
dtype = th.float32
encoder = CustomEncoder(args=vae_config, device=device, dtype=dtype)
decoder = CustomDecoder(args=vae_config, device=device, dtype=dtype)


vae_model = CustomVAE(vae_config=vae_config, encoder=encoder, decoder=decoder, device=device, dtype=dtype, 
                         num_contexts=num_contexts, detach_context_selection_for_value_system_selection=detach_context_selection_for_value_system_selection)
vae_model.train()

data_th = th.tensor(data, dtype=dtype, device=device)
train_indices = list(set(range(n_samples)) - set(test_indices) - set(validation_indices))
train_data_th = data_th[train_indices]
train_data = data[train_indices]
eval_data = data[validation_indices]
eval_data_th = data_th[validation_indices]

seed_everything(int(args.seed))
km = KMeans(n_clusters=num_contexts, init="k-means++")

y_pred_kmeans = km.fit_predict(train_data)
centroids_initial = th.tensor(km.cluster_centers_, dtype=dtype, device=device)
assignments = th.tensor(y_pred_kmeans, dtype=th.long, device=device)
eval_assignments = km.predict(eval_data)
eval_ground_truth = th.tensor(target[validation_indices], dtype=th.long, device=device)

vae_model.initialize_from_data(data=train_data_th, 
                               eval_data=eval_data_th,
                               assignments=assignments, 
                               eval_assignments=eval_assignments, 
                               eval_ground_truth=eval_ground_truth,
                               centroids=centroids_initial, config=config, args=args)
exit()