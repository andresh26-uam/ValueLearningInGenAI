#!/usr/bin/env bash
set -euo pipefail

MODEL_ALIAS="armo"
DATASET_ALIAS="ultra"
EXTRA_ARGS=()

resolve_model_name() {
    case "$1" in
        armo)
            printf '%s' 'ArmoRM-Llama3-8B-v0.1'
            ;;
        smol)
            printf '%s' 'SmolLM-135M-Instruct'
            ;;
        llama)
            printf '%s' 'ArmoRM-Llama3-8B-v0.1'
            ;;
        *)
            printf '%s' "$1"
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model=*)
            MODEL_ALIAS="${1#*=}"
            ;;
        --model)
            MODEL_ALIAS="$2"
            shift
            ;;
        --dataset=*)
            DATASET_ALIAS="${1#*=}"
            ;;
        --dataset)
            DATASET_ALIAS="$2"
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Usage: bash srun_eval.sh [--model=armo|smol|<hf-model>] [--dataset=ultra|pku] [extra eval args...]

Examples:
  bash srun_eval.sh --model=armo --dataset=ultra
  bash srun_eval.sh --model=smol --dataset=pku --checkpoint_path=run_name
EOF
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done

MODEL_NAME="$(resolve_model_name "$MODEL_ALIAS")"

exec srun --gpus=L40S:1 python vsl-rm/eval_no_context.py \
    --model_name="$MODEL_NAME" \
    --dataset="$DATASET_ALIAS" "${EXTRA_ARGS[@]}"