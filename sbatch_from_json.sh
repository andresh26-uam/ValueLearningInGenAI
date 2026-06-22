#!/usr/bin/env bash
set -euo pipefail

scriptvsl="${1}"
config_file="${2}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -f "$config_file" ]]; then
    echo "Config file not found: $config_file" >&2
    exit 1
fi

gpu_request="L40S:2"
extra_args=()
scriptvsl="vsl-rm/no_context_vsl.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus=*)
            gpu_request="${1#*=}"
            ;;
        --gpus)
            gpu_request="${2:-}"
            shift
            ;;
        --debug)
            extra_args+=("$1")
            ;;
        *)
            extra_args+=("$1")
            ;;
    esac
    shift
done

sbatch --gpus="$gpu_request" sbatch_script.sh $script_vsl --config_file="$config_file" "${extra_args[@]}"
