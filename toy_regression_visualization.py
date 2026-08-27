"""Run a small DAC toy experiment and export every round for web replay."""

import copy
import json
import os
import random

import numpy as np
import torch

from utils.classes import Client
from utils.toy_regression_utils import LinearRegression, generate_regression_multi
from utils.training_utils import client_information_exchange_DAC, train_clients_locally


# These values follow the existing toy experiment settings where possible.
SEED = 1
N_CLIENTS = 30
N_CLUSTERS = 3
CLIENTS_PER_CLUSTER = N_CLIENTS // N_CLUSTERS
N_ROUNDS = 50
N_DATA_TRAIN = 50
N_DATA_VAL = 100
BATCH_SIZE = 8
N_LOCAL_EPOCHS = 1
LEARNING_RATE = 0.003
N_NEIGHBORS = 5
SIMILARITY_METRIC = "inverse_training_loss"
PRIOR_UPDATE_RULE = "softmax"
TAU = 30
OUTPUT_PATH = os.path.join("save", "toy_visualization_30_clients.json")


def set_seed(seed):
    """Make the generated data, training, and communication order reproducible."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(theta, size, sigma):
    """Generate one tensor dataset using the repository's toy data generator."""
    features, targets = generate_regression_multi(theta, size, sigma)
    features = torch.from_numpy(features).float()
    targets = torch.from_numpy(targets).float().reshape(-1, 1)
    return torch.utils.data.TensorDataset(features, targets)


def model_values(client):
    """Flatten a client's model so it can be stored in JSON."""
    values = []
    for parameter in client.local_model.state_dict().values():
        values.extend(parameter.detach().cpu().reshape(-1).tolist())
    return [round(value, 6) for value in values]


def numeric(value):
    """Convert NumPy and Torch scalar values into JSON-compatible floats."""
    if hasattr(value, "item"):
        value = value.item()
    return round(float(value), 6)


def cluster_metrics(clients):
    """Average the current validation loss separately for each cluster."""
    return [
        round(float(np.mean([
            client.val_loss_list[-1]
            for client in clients
            if client.group == cluster
        ])), 6)
        for cluster in range(N_CLUSTERS)
    ]


def make_state(clients, round_number, communications):
    """Create one complete, browser-friendly snapshot of the experiment."""
    nodes = []
    for client in clients:
        similarities = {
            str(peer): numeric(score)
            for peer, score in enumerate(client.similarity_scores)
            if score > 0
        }
        priors = {
            str(peer): numeric(probability)
            for peer, probability in enumerate(client.priors)
            if probability > 0
        }
        nodes.append({
            "id": str(client.idx),
            "cluster": client.group,
            "validationLoss": round(float(client.val_loss_list[-1]), 6),
            "model": model_values(client),
            "similarities": similarities,
            "neighborProbabilities": priors,
        })

    return {
        "round": round_number,
        "clusterAverageValidationLoss": cluster_metrics(clients),
        "nodes": nodes,
        "communications": communications,
    }


def communication_records(clients):
    """Convert the latest sampled neighbors into directed communication events."""
    communications = []
    for client in clients:
        if not client.exchanges_every_round:
            continue
        neighbors = client.exchanges_every_round[-1]
        for neighbor in neighbors:
            communications.append({
                "source": str(client.idx),
                "target": str(int(neighbor)),
                "similarity": numeric(client.similarity_scores[neighbor]),
                "sameCluster": client.group == clients[neighbor].group,
            })
    return communications


def main():
    set_seed(SEED)
    device = torch.device("cpu")
    sigma = 3
    thetas = [np.random.uniform(-10, 10, 10) for _ in range(N_CLUSTERS)]

    trainsets = []
    for cluster in range(N_CLUSTERS):
        for _ in range(CLIENTS_PER_CLUSTER):
            trainsets.append(make_dataset(thetas[cluster], N_DATA_TRAIN, sigma))

    valsets = [make_dataset(theta, N_DATA_VAL, sigma) for theta in thetas]
    initial_model = LinearRegression(10, 1)
    clients = []
    for client_index in range(N_CLIENTS):
        cluster = client_index // CLIENTS_PER_CLUSTER
        clients.append(Client(
            train_set=trainsets[client_index],
            val_set=valsets[cluster],
            idxs_train=None,
            idxs_val=None,
            criterion=torch.nn.MSELoss(),
            lr=LEARNING_RATE,
            device=device,
            batch_size=BATCH_SIZE,
            num_users=N_CLIENTS,
            model=copy.deepcopy(initial_model),
            idx=client_index,
            stopping_rounds=N_ROUNDS + 1,
            ratio=1 / N_CLUSTERS,
            dataset="toy_problem",
            shift=None,
            theta=thetas[cluster],
        ))

    # Match the existing experiment: each client trains once before round one.
    clients = train_clients_locally(clients, N_LOCAL_EPOCHS, verbose=False)
    states = [make_state(clients, 0, [])]

    parameters = {
        "nbr_neighbors_sampled": N_NEIGHBORS,
        "prior_update_rule": PRIOR_UPDATE_RULE,
        "similarity_metric": SIMILARITY_METRIC,
        "tau": TAU,
        "cosine_alpha": 0.0,
        "mergatron": "chill",
        "aggregation_weighting": "trainset_size",
        "dataset": "toy_problem",
        "minmax": False,
    }

    for round_number in range(N_ROUNDS):
        clients = client_information_exchange_DAC(
            clients,
            parameters=parameters,
            verbose=False,
            round=round_number,
        )
        communications = communication_records(clients)
        clients = train_clients_locally(clients, N_LOCAL_EPOCHS, verbose=False)
        states.append(make_state(clients, round_number + 1, communications))
        print("Saved round {}/{}".format(round_number + 1, N_ROUNDS))

    output = {
        "experiment": "toy_regression_dac_30_clients",
        "description": "Three-cluster synthetic regression replay for decentralized communication.",
        "parameters": {
            "seed": SEED,
            "clients": N_CLIENTS,
            "clusters": N_CLUSTERS,
            "clientsPerCluster": CLIENTS_PER_CLUSTER,
            "rounds": N_ROUNDS,
            "trainSamplesPerClient": N_DATA_TRAIN,
            "validationSamplesPerCluster": N_DATA_VAL,
            "batchSize": BATCH_SIZE,
            "localEpochs": N_LOCAL_EPOCHS,
            "learningRate": LEARNING_RATE,
            "neighborsSampled": N_NEIGHBORS,
            "similarityMetric": SIMILARITY_METRIC,
            "priorUpdateRule": PRIOR_UPDATE_RULE,
            "tau": TAU,
        },
        "states": states,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, separators=(",", ":"))
    print("Wrote {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()
