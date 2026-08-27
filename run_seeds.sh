#!/bin/bash


GPUS="L40S:1"
CPU=True

DEBUG=()
SAVE=(--do_save=False)
REPORT_TO=wandb

if [[ "$CPU" == "True" ]]; then
    GPUS=""
    GPUDIRECTIVE=()
    SCRIPT=cpu_from_json.sh
else
    GPUDIRECTIVE=(--gpus="$GPUS")
    SCRIPT=sbatch_from_json.sh
fi

#DATASET=ultra
#CONFIG=run_configs/smol_ctx_linear.json

#DATASET=pku
#CONFIG=run_configs/smol_ctx_linear.json
DATASET=synth
CONFIG=run_configs/features_synth_ctx.json

DATASET=pku
CONFIG=run_configs/smol_ctx_linear.json

CTX_IMPLEMENTATION=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["context_implementation"])' "$CONFIG")
DOINIT=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("do_initialization", False))' "$CONFIG")
CTX_COEFF=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("ctx_coefficient", 0.0))' "$CONFIG")
VS_COEFF=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("vs_selection_coefficient", 0.0))' "$CONFIG")
DIRECT_GMM=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("direct_gmm", False))' "$CONFIG")

NORM=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("layer_normalization", "none"))' "$CONFIG")
VS_WEIGHT_INIT=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("vs_weight_initialization", "dirichlet"))' "$CONFIG")

SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"

if [[ "$DATASET" == "ultra" ]]; then

  DISCORDANCE_EPSILON=0.25
  training_initialization_data_size=5000
  NUM_TRAIN_EPOCHS=10
  if [[ "$CONFIG" == "run_configs/llama_linear_firstgr_thenvs.json" ]]; then
    NUM_TRAIN_EPOCHS=15
    fi
elif [[ "$DATASET" == "pku" ]]; then
  DISCORDANCE_EPSILON=0.5
  training_initialization_data_size=10000
  NUM_TRAIN_EPOCHS=100
  if [[ "$CONFIG" == "run_configs/llama_linear_firstgr_thenvs.json" ]]; then
    NUM_TRAIN_EPOCHS=150
  fi
elif [[ "$DATASET" == "apollo" ]]; then
  DISCORDANCE_EPSILON=0.04
  training_initialization_data_size=2000
  NUM_TRAIN_EPOCHS=5000
elif [[ "$DATASET" == "synth" ]]; then
  DISCORDANCE_EPSILON=-1
  training_initialization_data_size=1000
  NUM_TRAIN_EPOCHS=500
else
  echo "Unknown dataset: $DATASET" >&2
  exit 1
fi


echo $CONFIG
if [[ "$CONFIG" == "run_configs/features_apollo_ctx.json" ]]; then
    NAME=LinearGR_NOTNORMED_BASICSMOOTH_10_Sharp_slow
elif [[ "$CONFIG" == "run_configs/features_synth_ctx.json" ]]; then
    NAME=Synth
elif [[ "$CONFIG" == "run_configs/llama_ctx_linear.json" ]]; then
    NAME=LlamaCtxVSLRMv5
elif [[ "$CONFIG" == "run_configs/llama_rlhf_linear.json" ]]; then
  NAME=LlamaBTRMv2
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
elif [[ "$CONFIG" == "run_configs/llama_linear_firstgr_thenvs.json" ]]; then
  NAME=LlamaSEQ-RMv2
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
elif [[ "$CONFIG" == "run_configs/llama_linear.json" ]]; then
  NAME=LlamaVSLRMv2
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
elif [[ "$CONFIG" == "run_configs/llama_grounding_linear.json" ]]; then
  NAME=LlamaGRRMv2
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
  elif [[ "$CONFIG" == "run_configs/llama_nolag_linear.json" ]]; then
  NAME=LlamaVSL-NL-RMv2
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
elif [[ "$CONFIG" == "run_configs/smol_rlhf_linear.json" ]]; then
  NAME=SmolBTRMv2
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
elif [[ "$CONFIG" == "run_configs/smol_linear.json" ]]; then
  NAME=SmolVSLRMv2
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
elif [[ "$CONFIG" == "run_configs/smol_ctx_linear.json" ]]; then
  NAME=SmolCtxVSLRMv12_VAE_TH1_SMOOTH
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
elif [[ "$CONFIG" == "run_configs/smol_ctx_linear_baseline.json" ]]; then
  NAME=SmolCtxVSLRMv11_TH50VS
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
elif [[ "$CONFIG" == "run_configs/smol_ctx_linear_baseline_smooth.json" ]]; then
  NAME=SmolCtxVSLRMv11
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
else
  NAME=test
fi

NAME="${NAME}_${CTX_IMPLEMENTATION}_INITVS_${VS_WEIGHT_INIT}_INITCTX_${DOINIT}_VS_${VS_COEFF}_CTX_${CTX_COEFF}_NORM_${NORM}_DIRGMM_${DIRECT_GMM}"
echo "GPUs: $GPUS"
echo "Training with config $CONFIG on dataset: $DATASET with discordance_epsilon: $DISCORDANCE_EPSILON and num_train_epochs: $NUM_TRAIN_EPOCHS"
echo "Run name: ${GPUS}${NAME}"



for seed in 42; do
    args=(
        "$SCRIPT_PYTHON"
        "$CONFIG"
        "--dataset=$DATASET"
        "--run_name=${GPUS}${NAME}"
        "--seed=$seed"
        "--num_train_epochs=$NUM_TRAIN_EPOCHS"
        "--discordance_epsilon=$DISCORDANCE_EPSILON"
        "--training_initialization_data_size=$training_initialization_data_size"
        "--report_to=$REPORT_TO"
    )

    args+=("${GPUDIRECTIVE[@]}")
    args+=("${DEBUG[@]}")
    args+=("${SAVE[@]}")

    printf 'Running:'
    printf ' %q' bash "$SCRIPT" "${args[@]}"
    printf '\n'

    bash "$SCRIPT" "${args[@]}"
done
