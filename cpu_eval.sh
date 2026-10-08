#!/usr/bin/env bash
set -euo pipefail

# DEFAULTS
MODEL_ALIAS="smol"
DATASET_ALIAS="pku"
#TASK_TYPE="nlp_based"
TASK_TYPE="feature_based" # This one for apollo

SKIP_COMPARE_CHECKPOINTS=False
KNLPMEANS_BASELINE="pku_test_ST_True_KMEANS_THEN_VS_INITVS_equal_INITCTX_True_INITVSGR_True_VS_0.0_CTX_1.0_TM_1.0_LC_1.0_NORM_BatchNorm_DIRCVS_True_0917_155257_epo100_s42"
DIRECT_VS_BASELINE="pku_test_ST_True_DIRECT_VS_INITVS_equal_INITCTX_True_INITVSGR_True_VS_0.0_CTX_0.0_TM_0.0_LC_0.0_NORM_BatchNorm_DIRCVS_False_0916_165014_epo100_s42"
KNLPMEANS_BASELINE="oasst_test_ST_True_KMEANS_THEN_VS_INITVS_equal_INITCTX_True_INITVSGR_True_VS_0.0_CTX_1.0_TM_1.0_LC_1.0_NORM_BatchNorm_DIRCVS_True_1002_170032_epo25_s42"
DIRECT_VS_BASELINE="oasst_test_ST_True_DIRECT_VS_INITVS_equal_INITCTX_True_INITVSGR_True_VS_0.0_CTX_0.0_TM_0.0_LC_0.0_NORM_BatchNorm_DIRCVS_False_1001_154922_epo10_s42"

EXTRA_ARGS=("--use_cpu")

resolve_model_name() {
    case "$1" in
        armo)
            printf '%s' 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
            ;;
        smol)
            printf '%s' 'HuggingFaceTB/SmolLM-135M-Instruct'
            ;;
        llama)
            printf '%s' 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
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
if [[ $MODEL_ALIAS == "armo" ]]; then
    TASK_TYPE="nlp_based"
fi
if [[ $MODEL_ALIAS == "smol" ]]; then
    TASK_TYPE="nlp_based"
fi
if [[ $MODEL_ALIAS == "llama" ]]; then
    TASK_TYPE="nlp_based"
fi
if [[ $DATASET_ALIAS == "pku" ]]; then
    TASK_TYPE="nlp_based"
fi
EXTRA_ARGS+=("--task_type=$TASK_TYPE")

# Add defaults for checkpoint comparison only if not already in EXTRA_ARGS
if ! printf '%s\n' "${EXTRA_ARGS[@]}" | grep -q "^--skip_compare_checkpoints"; then
    EXTRA_ARGS+=("--skip_compare_checkpoints=${SKIP_COMPARE_CHECKPOINTS}")
fi
if ! printf '%s\n' "${EXTRA_ARGS[@]}" | grep -q "^--compare_checkpoint_paths"; then
    EXTRA_ARGS+=("--compare_checkpoint_paths" "${KNLPMEANS_BASELINE}" "${DIRECT_VS_BASELINE}")
fi

MODEL_NAME="$(resolve_model_name "$MODEL_ALIAS")"

echo "Evaluating model: $MODEL_NAME on dataset: $DATASET_ALIAS"
echo "with extra args: ${EXTRA_ARGS[*]}"

bash sbatch_script.sh  vsl-rm/eval_no_context.py \
    --model_name="$MODEL_NAME" \
    --dataset="$DATASET_ALIAS" --use_cpu "${EXTRA_ARGS[@]}" \
    --debug 