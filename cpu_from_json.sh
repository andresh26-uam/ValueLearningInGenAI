#!/usr/bin/env bash

config_file="${1}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -f "$config_file" ]]; then
    echo "Config file not found: $config_file" >&2
    exit 1
fi

PYTHONOPTIMIZE=1 accelerate launch --config_file="accelerate_config/cpu_config.yaml" vsl-rm/no_context_vsl.py --use_cpu --config_file="$config_file" "$@"