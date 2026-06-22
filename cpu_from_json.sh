#!/usr/bin/env bash
set -euo pipefail

scriptvsl="${1}"
config_file="${2}"

if [[ $# -gt 0 ]]; then
    shift 2
fi

if [[ ! -f "$config_file" ]]; then
    echo "Config file not found: $config_file" >&2
    exit 1
fi

debug_mode=0
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            debug_mode=1
            ;;
        *)
            extra_args+=("$1")
            ;;
    esac
    shift
done

if [[ "$debug_mode" -eq 1 ]]; then
    python_optimize=0
else
    python_optimize=1
fi

PYTHONOPTIMIZE="$python_optimize" accelerate launch --config_file="accelerate_config/cpu_config.yaml" $scriptvsl --use_cpu --config_file="$config_file" "${extra_args[@]}"