#!/usr/bin/env bash
#SBATCH --job-name=ValueLearningInGenAI
#SBATCH --chdir=/home/aholg/ValueLearningInGenAI

# GPU allocations should pass --gpus and --mem-per-gpu at submission time.
# CPU allocations should pass --mem and --cpus-per-task instead.
#SBATCH --time=100:00:00
set -euo pipefail

if [[ "${1:-}" == "--parallel-script" ]]; then
	if [[ $# -ne 2 || ! -f "$2" ]]; then
		echo "Usage: sbatch_script.sh --parallel-script <script>" >&2
		exit 2
	fi
	echo "Running parallel jobs in Slurm allocation: $2"
	bash "$2"
	exit $?
fi

if [[ $# -lt 1 ]]; then
	echo "Usage: sbatch_script.sh <train_script.py> [script args ...]" >&2
	exit 2
fi

entrypoint="$1"
shift

debug_mode=0
use_cpu=0
entrypoint_args=()
while [[ $# -gt 0 ]]; do
	case "$1" in
		--debug)
			debug_mode=1
			;;
        --use_cpu)
            use_cpu=1
            entrypoint_args+=("$1")
            ;;
		*)
			entrypoint_args+=("$1")
			;;
	esac
	shift
done

num_processes="${SLURM_GPUS_ON_NODE:-${NUM_PROCESSES:-1}}"
if ! [[ "$num_processes" =~ ^[0-9]+$ ]] || [[ "$num_processes" -lt 1 ]]; then
	num_processes=1
fi

num_machines="${NUM_MACHINES:-${SLURM_NNODES:-1}}"

config_file="accelerate_config/default_config.yaml"
if [[ "$num_processes" -eq 1 ]]; then
    if [[ "$use_cpu" -eq 1 ]]; then
        config_file="accelerate_config/cpu_config.yaml"
    else
        config_file="accelerate_config/single_config.yaml"
    fi
	#config_file="accelerate_config/single_config.yaml"
fi

gpu_ids=""
if [[ "$use_cpu" -eq 0 ]]; then
	gpu_ids="${CUDA_VISIBLE_DEVICES:-}"
	if [[ -z "$gpu_ids" || "$gpu_ids" == "NoDevFiles" ]]; then
		gpu_ids="$(seq -s, 0 $((num_processes - 1)))"
	fi
fi

echo "Running with GPU IDs: $gpu_ids"
echo "Number of processes: $num_processes"
echo "Number of machines: $num_machines"
echo "Debug mode: $debug_mode"

if [[ "$debug_mode" -eq 1 ]]; then
	python_optimize=0
else
	python_optimize=1
fi

launch_args=(
	--config_file="$config_file"
	--num_processes="$num_processes"
	--num_machines="$num_machines"
)
if [[ "$use_cpu" -eq 0 ]]; then
	launch_args+=(--gpu_ids="$gpu_ids")
fi

PYTHONOPTIMIZE="$python_optimize" python -m accelerate.commands.launch \
	"${launch_args[@]}" \
	"$entrypoint" "${entrypoint_args[@]}"
