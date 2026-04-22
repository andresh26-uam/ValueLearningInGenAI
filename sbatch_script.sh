#!/usr/bin/env bash
#SBATCH --job-name=ValueLearningInGenAI
#SBATCH --chdir=/home/aholg/ValueLearningInGenAI

#SBATCH --gpus=L40S:4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=24:00:00
set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: sbatch srun_test.sh <train_script.py> [script args ...]" >&2
	exit 2
fi

entrypoint="$1"
shift

num_processes="${SLURM_GPUS_ON_NODE:-${NUM_PROCESSES:-1}}"
if ! [[ "$num_processes" =~ ^[0-9]+$ ]] || [[ "$num_processes" -lt 1 ]]; then
	num_processes=1
fi

num_machines="${NUM_MACHINES:-${SLURM_NNODES:-1}}"

config_file="accelerate_config/default_config.yaml"
if [[ "$num_processes" -eq 1 ]]; then
	config_file="accelerate_config/single_config.yaml"
fi

gpu_ids="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$gpu_ids" || "$gpu_ids" == "NoDevFiles" ]]; then
	gpu_ids="$(seq -s, 0 $((num_processes - 1)))"
fi

echo "Running with GPU IDs: $gpu_ids"
echo "Number of processes: $num_processes"
echo "Number of machines: $num_machines"

python -m accelerate.commands.launch \
	--config_file="$config_file" \
	--debug \
	--num_processes="$num_processes" \
	--num_machines="$num_machines" \
	--gpu_ids="$gpu_ids" \
	"$entrypoint" "$@"

