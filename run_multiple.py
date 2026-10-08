
"""Run a parameter sweep by creating temporary config-file variants."""

import itertools
import json
import math
import secrets
import subprocess
import tempfile
from pathlib import Path

# PKU TRAIN EPOCHS = 100
DATASET = "oasst"
BASE_CONFIG = "run_configs/smol_ctx_linear_ae_kmeans.json"
MAX_SIMULTANEOUS_RUNS = 16 # 24 no.
MODEL_ALIAS = "smol"
NUM_TRAIN_EPOCHS = [75]

if DATASET == "pku":
    NUM_VALUES = 5 
elif DATASET == "ultra":
    NUM_VALUES = 4
elif DATASET == "oasst":
    NUM_VALUES = 7
else:
    raise ValueError(f"Unknown dataset {DATASET}")
ARGS_TO_CHANGE = {
    "loss_func_type":["DEFAULT"],
    "C1_learning_rate": [0.0001, 0.0003],
    "C1_grounding_learning_rate": [0.0001, 0.0003],
    "C1_context_learning_rate": [0.0001, 0.0003],
    "weight_decay": [0.0,0.001],
    "vae_type": ["ae"],
    "vae_dropout": [0.0],
    "max_grad_norm": [10.0,],
    "ctx_coefficient": [1.0,50.0],
    "ctx_chr_coefficient": [0.0],
    "vae_orthogonality_coefficient": [0.0, 1.0],
    "layer_normalization": ["BatchNorm", "none"],
    "final_layer_activation": ["none"],
    "vs_weight_initialization": ["equal"],
    "do_initialization": [False, True],
    "vae_latent_dim": [16, 128],
}
# If an argument starts with C<number>_, arguments with the same number are
# coupled and vary together by position instead of by Cartesian product.
#DISCORDANCE_EPSILON = ["use_default", 0.0, 0.01]
DISCORDANCE_EPSILON = [0.1, 0.003]

AVOID_COMBINATIONS = [
    {"loss_func_type": "DEFAULT", "vae_latent_dim": 2},
    {"ctx_coefficient": 50.0, "do_initialization": False},

    {"final_layer_activation": "Tanh", "discordance_epsilon": "use_default"},
]
def _split_coupled_arguments(arguments):
    groups = {}
    independent = {}
    for name, values in arguments.items():
        if name.startswith("C") and "_" in name:
            prefix, parameter = name.split("_", 1)
            if prefix[1:].isdigit():
                groups.setdefault(prefix, {})[parameter] = values
                continue
        independent[name] = values

    coupled = []
    for prefix, parameters in groups.items():
        lengths = {len(values) for values in parameters.values()}
        if len(lengths) != 1:
            raise ValueError(f"Coupled group {prefix} must contain equally sized lists")
        coupled.append([
            dict(zip(parameters, values))
            for values in zip(*parameters.values())
        ])
    return independent, coupled


def _is_avoided(config):
    return any(
        all(config.get(parameter) == value for parameter, value in combination.items())
        for combination in AVOID_COMBINATIONS
    )


def _config_variants(base_config):
    independent, coupled = _split_coupled_arguments(ARGS_TO_CHANGE)
    independent_variants = itertools.product(*independent.values())
    independent_names = tuple(independent)
    coupled_variants = itertools.product(*coupled) if coupled else [()]

    for independent_values, coupled_values in itertools.product(
        independent_variants, coupled_variants
    ):
        variant = dict(zip(independent_names, independent_values))
        for group in coupled_values:
            variant.update(group)
        if "vae_latent_dim" in variant and variant["vae_latent_dim"] == "num_values":
            variant["vae_latent_dim"] = NUM_VALUES
        config = dict(base_config)
        config.update(variant)
        if _is_avoided(config):
            continue
        yield config


def main():
    base_path = Path(BASE_CONFIG)
    with base_path.open(encoding="utf-8") as config_file:
        base_config = json.load(config_file)

    with tempfile.TemporaryDirectory(prefix="run_multiple_") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        config_stem = base_path.stem
        commands = []
        for index, config in enumerate(_config_variants(base_config), start=1):
            config_path = temporary_directory / f"config_{index}_{config_stem}.json"
            with config_path.open("w", encoding="utf-8") as config_file:
                json.dump(config, config_file, indent=2)
                config_file.write("\n")

            for epsilon in DISCORDANCE_EPSILON:
                for num_train_epochs in NUM_TRAIN_EPOCHS:
                    run_config = {**config, "discordance_epsilon": epsilon}
                    if _is_avoided(run_config):
                        continue
                    run_number = len(commands) + 1
                    seeds = [secrets.randbelow(2000)]
                    command = [
                        "bash",
                        "run_seeds.sh",
                        "--config",
                        str(config_path),
                        "--dataset",
                        DATASET,
                        "--model",
                        MODEL_ALIAS,
                        "--num_train_epochs",
                        str(num_train_epochs),
                        "--seed",
                        *map(str, seeds),
                        "--run_index",
                        str(run_number),
                        "--skip_compare_checkpoints",
                    ]
                    if epsilon != "use_default":
                        command.extend(["--discordance_epsilon", str(epsilon)])
                    commands.append((index, epsilon, num_train_epochs, command))

        total_runs = len(commands)
        cycles = math.ceil(total_runs / MAX_SIMULTANEOUS_RUNS)
        print(
            f"This sweep will execute {total_runs} runs in {cycles} cycles "
            f"of up to {MAX_SIMULTANEOUS_RUNS} simultaneous runs."
        )
        approval = input("Proceed with the sweep? [y/N]: ").strip().lower()
        if approval not in {"y", "yes"}:
            print("Sweep cancelled.")
            return

        for cycle_start in range(0, total_runs, MAX_SIMULTANEOUS_RUNS):
            cycle = commands[cycle_start : cycle_start + MAX_SIMULTANEOUS_RUNS]
            print(
                f"Starting cycle {cycle_start // MAX_SIMULTANEOUS_RUNS + 1}/{cycles} "
                f"({len(cycle)} runs)",
                flush=True,
            )
            processes = []
            for index, epsilon, num_train_epochs, command in cycle:
                print(
                    f"Running variant {index} with epsilon={epsilon}, "
                    f"epochs={num_train_epochs}",
                    flush=True,
                )
                processes.append(subprocess.Popen(command))

            return_codes = [process.wait() for process in processes]
            failures = [return_code for return_code in return_codes if return_code != 0]
            if failures:
                raise subprocess.CalledProcessError(failures[0], "run_seeds.sh")



if __name__ == "__main__":
    main()

