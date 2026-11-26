
"""Run a parameter sweep by creating temporary config-file variants."""

import itertools
import json
import math
import secrets
import shlex
import subprocess
import tempfile
import traceback
from pathlib import Path

# PKU TRAIN EPOCHS = 100

GPUS=""
USE_CPU = True
USE_SBATCH = True
assert GPUS in {"L40S:1", "H100:1", ""}, f"GPUS must be one of 'L40S:1' or 'H100:1', or nothing (\"\"), but got {GPUS}"
DATASET = "oasst"
BASE_CONFIG = "run_configs/llama_ctx_linear_ae_kmeans.json"
MAX_SIMULTANEOUS_RUNS = 8
N_SEEDS_PER_RUN = 3
MODEL_ALIAS = "llama"
DISCORDANCE_EPSILON = "use_default"
CPU_PROCESSES_PER_RUN = 2
MEMORY_GB_PER_RUN = 20

LAST_TRACEBACK_PATH = Path("___last_rumultiple_traceback.txt")


def _write_parallel_script(path, commands, use_current_allocation):
    dataset_ready_file = path.with_suffix(".dataset-ready")
    leader_finished_file = path.with_suffix(".leader-finished")
    quoted_ready_file = shlex.quote(str(dataset_ready_file))
    quoted_finished_file = shlex.quote(str(leader_finished_file))
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "pids=()",
    ]
    if use_current_allocation:
        lines.append("export RUN_IN_CURRENT_ALLOCATION=1")
    lines.extend(
        [
            "(",
            f"  RUN_MULTIPLE_DATASET_READY_FILE={quoted_ready_file} "
            f"{shlex.join(commands[0][3])}",
            "  status=$?",
            f"  touch {quoted_finished_file}",
            '  exit "$status"',
            ") &",
            'leader_pid=$!',
            'pids+=("$leader_pid")',
            f"while [[ ! -e {quoted_ready_file} && ! -e {quoted_finished_file} ]]; do",
            "  sleep 1",
            "done",
            f"if [[ ! -e {quoted_ready_file} ]]; then",
            '  if wait "$leader_pid"; then',
            '    echo "First run exited before shared dataset initialization completed" >&2',
            "    exit 1",
            "  else",
            '    exit "$?"',
            "  fi",
            "fi",
        ]
    )
    for _, _, _, command in commands[1:]:
        lines.extend([f"{shlex.join(command)} &", 'pids+=("$!")'])
    lines.extend(
        [
            "status=0",
            'for pid in "${pids[@]}"; do',
            '  if ! wait "$pid"; then',
            "    status=1",
            "  fi",
            "done",
            "exit \"$status\"",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


print(f"Running sweep for dataset:\n {DATASET}\nbase config:\n {BASE_CONFIG}\nmodel alias: \n {MODEL_ALIAS}")
assert MODEL_ALIAS in {"llama", "smol"}
assert MODEL_ALIAS in BASE_CONFIG, f"Model alias {MODEL_ALIAS} not in base config name path {BASE_CONFIG}, which may indicate a mismatch between the base config and the model alias."
if not USE_CPU and not GPUS:
    raise ValueError("A non-empty GPUS value is required when USE_CPU is False.")
if DATASET == "pku":
    NUM_VALUES = 5 
    NUM_TRAIN_EPOCHS = [100]
elif DATASET == "ultra":
    NUM_VALUES = 4
    NUM_TRAIN_EPOCHS = [10]
elif DATASET == "oasst":
    NUM_VALUES = 7
    NUM_TRAIN_EPOCHS = [75]
elif DATASET == "prism":
    NUM_VALUES = 7
    NUM_TRAIN_EPOCHS = [50]
else:
    raise ValueError(f"Unknown dataset {DATASET}")
ARGS_TO_CHANGE = {
    "loss_func_type":["DEFAULT"],
    "C1_learning_rate": [0.0003],
    "C1_grounding_learning_rate": [0.0003],
    "C1_context_learning_rate": [0.0003],
    "weight_decay": [0.001,0.0],
    "vae_type": ["ae"],
    "vae_dropout": [0.0],
    "max_grad_norm": [10.0,],
    "ctx_coefficient": [1.0],
    "ctx_chr_coefficient": [0.0],
    "do_initialization": [False, True],
    "vae_orthogonality_coefficient": [1.0],
    "detach_context_selection_for_value_system_selection": [False, True],
    "layer_normalization": ["BatchNorm"],
    "final_layer_activation": ["none"],
    "vs_weight_initialization": ["equal"],
    "grounding_loss_tendency_update_ratio": [0.1,0.01],
    "vae_latent_dim": [128],
    "C2_max_contexts": [20],
    "C2_max_value_systems": [20]
}
# If an argument starts with C<number>_, arguments with the same number are
# coupled and vary together by position instead of by Cartesian product.
#DISCORDANCE_EPSILON = ["use_default", 0.0, 0.01]
#DISCORDANCE_EPSILON = [0.1,0.003]



AVOID_COMBINATIONS = [
    {"ctx_coefficient": 50.0, "do_initialization": False},
    {"ctx_coefficient": 1.0, "vae_orthogonality_coefficient": 1.0, "max_contexts": 10},

    {"final_layer_activation": "Tanh",},
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

    with tempfile.TemporaryDirectory(
        prefix="run_multiple_", dir=Path(__file__).resolve().parent
    ) as temporary_directory:
        temporary_directory = Path(temporary_directory)
        config_stem = base_path.stem
        commands = []
        for index, config in enumerate(_config_variants(base_config), start=1):
            config_path = temporary_directory / f"config_{index}_{config_stem}.json"
            with config_path.open("w", encoding="utf-8") as config_file:
                json.dump(config, config_file, indent=2)
                config_file.write("\n")

            epsilon = DISCORDANCE_EPSILON
            for num_train_epochs in NUM_TRAIN_EPOCHS:
                run_config = {**config, "discordance_epsilon": epsilon}
                if _is_avoided(run_config):
                    continue
                run_number = len(commands) + 1
                seeds = [secrets.randbelow(2000) for s in range(N_SEEDS_PER_RUN)]
                command = [
                    "bash",
                    "run_seeds.sh",
                    "--config",
                    str(config_path),
                    "--dataset",
                    DATASET,
                    "--gpus",
                    GPUS,
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
                if not USE_CPU:
                    command.append("--use_gpu")
                if epsilon != "use_default":
                    command.extend(["--discordance_epsilon", str(epsilon)])
                commands.append((index, epsilon, num_train_epochs, command))

        total_configurations = len(commands)
        total_runs = total_configurations * N_SEEDS_PER_RUN
        cycles = math.ceil(total_configurations / MAX_SIMULTANEOUS_RUNS)
        print(
            f"This sweep will execute {total_runs} runs "
            f"({total_configurations} configurations x {N_SEEDS_PER_RUN} seeds) "
            f"in {cycles} cycles of up to {MAX_SIMULTANEOUS_RUNS} x {N_SEEDS_PER_RUN} simultaneous runs."
        )
        approval = input("Proceed with the sweep? [y/N]: ").strip().lower()
        if approval not in {"y", "yes"}:
            print("Sweep cancelled.")
            return

        for cycle_start in range(0, total_configurations, MAX_SIMULTANEOUS_RUNS):
            cycle = commands[cycle_start : cycle_start + MAX_SIMULTANEOUS_RUNS]
            print(
                f"Starting cycle {cycle_start // MAX_SIMULTANEOUS_RUNS + 1}/{cycles} "
                f"({len(cycle)} runs)",
                flush=True,
            )
            for index, epsilon, num_train_epochs, _ in cycle:
                print(
                    f"Running variant {index} with epsilon={epsilon}, "
                    f"epochs={num_train_epochs}",
                    flush=True,
                )

            cycle_number = cycle_start // MAX_SIMULTANEOUS_RUNS + 1
            batch_script = temporary_directory / f"cycle_{cycle_number}.sh"
            _write_parallel_script(batch_script, cycle, USE_SBATCH)
            if USE_SBATCH:
                sbatch_command = ["sbatch", "--wait"]
                if USE_CPU:
                    sbatch_command.extend(
                        [
                            f"--cpus-per-task={CPU_PROCESSES_PER_RUN * len(cycle)}",
                            f"--mem={MEMORY_GB_PER_RUN * len(cycle)}G",
                        ]
                    )
                else:
                    sbatch_command.extend(
                        [f"--gpus={GPUS}", "--mem-per-gpu=40G"]
                    )
                sbatch_command.extend(
                    ["sbatch_script.sh", "--parallel-script", str(batch_script)]
                )
                resource_description = (
                    f"{CPU_PROCESSES_PER_RUN * len(cycle)} CPUs, no GPUs"
                    if USE_CPU
                    else f"{GPUS}"
                )
                print(
                    f"Submitting one Slurm allocation for {len(cycle)} parallel runs "
                    f"using {resource_description}",
                    flush=True,
                )
                return_codes = [subprocess.run(sbatch_command).returncode]
            else:
                return_codes = [
                    subprocess.run(["bash", str(batch_script)]).returncode
                ]

            failures = [return_code for return_code in return_codes if return_code != 0]
            if failures:
                try:
                    command = sbatch_command if USE_SBATCH else "run_seeds.sh"
                    raise subprocess.CalledProcessError(failures[0], command)
                except subprocess.CalledProcessError:
                    with LAST_TRACEBACK_PATH.open("w", encoding="utf-8") as traceback_file:
                        traceback.print_exc(file=traceback_file)
                print(
                    f"Cycle failed; traceback written to "
                    f"{LAST_TRACEBACK_PATH}",
                    flush=True,
                )



if __name__ == "__main__":
    main()
