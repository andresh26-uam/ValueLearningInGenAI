#!/usr/bin/env bash



set -eu
# Enable pipefail only when supported (e.g. bash, zsh).
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

# Run inside the Slurm context so we can read CUDA_VISIBLE_DEVICES assigned by srun.
srun --gpus=L40S:2 --mem-per-gpu=20G --pty bash -lc '
set -euo pipefail
#SBATCH --job-name=ValueLearningInGenAI
#SBATCH --chdir=/home/aholg/ValueLearningInGenAI
#SBATCH --mem-per-gpu=8G
#SBATCH --cpus-per-gpu=1
#SBATCH --mincpus=1

gpu_ids="${CUDA_VISIBLE_DEVICES:-all}"
if [[ -z "$gpu_ids" || "$gpu_ids" == "NoDevFiles" ]]; then
	gpu_ids="all"
fi

IFS=',' read -r -a gpu_id_arr <<< "$gpu_ids"
num_processes="${#gpu_id_arr[@]}"

num_machines="${NUM_MACHINES:-1}"

config_file="accelerate_config/default_config.yaml"
if [[ "$num_processes" -eq 1 ]]; then
	config_file="accelerate_config/single_config.yaml"
fi

echo "Running with GPU IDs: $gpu_ids"
echo "Number of processes: $num_processes"
echo "Number of machines: $num_machines"
exec python -m accelerate.commands.launch \
	--config_file="$config_file" \
	--debug \
	--num_processes="$num_processes" \
	--num_machines="$num_machines" \
	--gpu_ids="$gpu_ids" \
	"$@"
' _ "$@"

