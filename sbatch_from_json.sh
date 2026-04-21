#!/usr/bin/env bash

config_file="${1:-eval_armo.json}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -f "$config_file" ]]; then
    echo "Config file not found: $config_file" >&2
    exit 1
fi

sbatch sbatch_script.sh vsl-rm/no_context_vsl.py --config_file="$config_file" "$@"
