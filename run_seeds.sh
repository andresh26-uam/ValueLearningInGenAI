#!/bin/bash


GPUS="L40S:1"
CPU=True

DEBUG=()
SAVE=(--do_save=True --do_checkpointing=False)
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
#
DATASET=synth
CONFIG=run_configs/features_synth_ctx.json

#DATASET=apollo

#CONFIG=run_configs/features_apollo_ctx.json
DATASET=pku
CONFIG=run_configs/smol_ctx_linear_baseline.json

CTX_IMPLEMENTATION=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["context_implementation"])' "$CONFIG")
DOINIT=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("do_initialization", False))' "$CONFIG")
CTX_COEFF=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("ctx_coefficient", 0.0))' "$CONFIG")
VS_COEFF=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("vs_selection_coefficient", 0.0))' "$CONFIG")
DIRCVS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("direct_context_to_vs_relation", False))' "$CONFIG")
LAMBDA=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("vae_lambda_clustering", 0.0))' "$CONFIG")
TEMP=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("vae_initial_temperature", 0.0))' "$CONFIG")
SENTENCE_TRANS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("use_sentence_transformer", False))' "$CONFIG")
DOINIT_VS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("do_vs_initialization", False))' "$CONFIG")
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
  echo "Unknown dataset: $DATASET"
  exit 1
fi

if [[ "$CTX_IMPLEMENTATION" == "KMEANS_THEN_VS" ]]; then
        training_initialization_data_size="all"
else
    training_initialization_data_size=10000
fi

echo $CONFIG
if [[ "$CONFIG" == "run_configs/features_apollo_ctx.json" ]]; then
    NAME=ApolloVSLRMv17
elif [[ "$CONFIG" == "run_configs/features_synth_ctx.json" ]]; then
    NAME=SynthV17_savetest
elif [[ "$CONFIG" == "run_configs/llama_ctx_linear.json" ]]; then
    NAME=LlamaCtxVSLRMv5
elif [[ "$CONFIG" == "run_configs/llama_rlhf_linear.json" ]]; then
  NAME=LlamaBTRMv2
elif [[ "$CONFIG" == "run_configs/llama_linear_firstgr_thenvs.json" ]]; then
  NAME=LlamaSEQ-RMv2
elif [[ "$CONFIG" == "run_configs/llama_linear.json" ]]; then
  NAME=LlamaVSLRMv2
elif [[ "$CONFIG" == "run_configs/llama_grounding_linear.json" ]]; then
  NAME=LlamaGRRMv2
elif [[ "$CONFIG" == "run_configs/llama_nolag_linear.json" ]]; then
  NAME=LlamaVSL-NL-RMv2
elif [[ "$CONFIG" == "run_configs/smol_rlhf_linear.json" ]]; then
  NAME=SmolBTRMv2
elif [[ "$CONFIG" == "run_configs/smol_linear.json" ]]; then
  NAME=SmolVSLRMv2
elif [[ "$CONFIG" == "run_configs/smol_ctx_linear.json" ]]; then
  NAME=SmolCtxVSLRMv17_AEDET
elif [[ "$CONFIG" == "run_configs/smol_ctx_linear_ae_kmeans.json" ]]; then
  NAME=SmolCtxVSLRMv17_AE
elif [[ "$CONFIG" == "run_configs/smol_ctx_linear_gmm_and_classifier.json" ]]; then
  NAME=SmolCtxVSLRMv17_GMMC
elif [[ "$CONFIG" == "run_configs/smol_ctx_linear_baseline.json" ]]; then
  NAME=SmolCtxVSLRMv17_DIRECTVS
elif [[ "$CONFIG" == "run_configs/smol_ctx_linear_baseline_smooth.json" ]]; then
  NAME=SmolCtxVSLRMv17_BS
else
  NAME=test
fi

NAME="${NAME}_ST_${SENTENCE_TRANS}_${CTX_IMPLEMENTATION}_INITVS_${VS_WEIGHT_INIT}_INITCTX_${DOINIT}_INITVSGR_${DOINIT_VS}_VS_${VS_COEFF}_CTX_${CTX_COEFF}_TM_${TEMP}_LC_${LAMBDA}_NORM_${NORM}_DIRCVS_${DIRCVS}"
echo "GPUs: $GPUS"
echo "Training with config $CONFIG"
echo "On dataset: $DATASET with discordance_epsilon: $DISCORDANCE_EPSILON"
echo "and num_train_epochs: $NUM_TRAIN_EPOCHS"
echo "Run name: ${GPUS}${NAME}"



for seed in 42; do
    args=(
        "$SCRIPT_PYTHON"
        "$CONFIG"
        "--dataset=$DATASET"
        "--run_name=${GPUS}${NAME}"
        "--seed=$seed" 
        #"--recalculate_features"
        #"--repostprocess"
        #"--do_train=False"
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
