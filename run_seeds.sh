#!/bin/bash

GPUS="L40S:1"
GPUDIRECTIVE="--gpus=${GPUS}"
CPU=False
if [[ "$CPU" == "True" ]]; then
  GPUDIRECTIVE=""
  GPUS=""
  SCRIPT=cpu_from_json.sh
else
  SCRIPT=sbatch_from_json.sh
fi

#DATASET=ultra
#CONFIG=run_configs/smol_ctx_linear.json
DATASET=ultra
CONFIG=run_configs/llama_ultra_ctx.json
SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"

if [[ "$DATASET" == "ultra" ]]; then
  DISCORDANCE_EPSILON=0.25
  NUM_TRAIN_EPOCHS=10
  if [[ "$CONFIG" == "run_configs/llama_linear_firstgr_thenvs.json" ]]; then
    NUM_TRAIN_EPOCHS=15
    EVAL_EVERY_STEPS="--eval_every_steps=200"
    fi
elif [[ "$DATASET" == "pku" ]]; then
  DISCORDANCE_EPSILON=0.5
  NUM_TRAIN_EPOCHS=100
  if [[ "$CONFIG" == "run_configs/llama_linear_firstgr_thenvs.json" ]]; then
    NUM_TRAIN_EPOCHS=150
    EVAL_EVERY_STEPS="--eval_every_steps=200"
  fi
elif [[ "$DATASET" == "apollo" ]]; then
  DISCORDANCE_EPSILON=0.04
  NUM_TRAIN_EPOCHS=5000
  EVAL_EVERY_STEPS="--eval_every_steps=200"
elif [[ "$DATASET" == "synth" ]]; then
  DISCORDANCE_EPSILON=-1
  NUM_TRAIN_EPOCHS=5000
  EVAL_EVERY_STEPS="--eval_every_steps=100"

else
  echo "Unknown dataset: $DATASET" >&2
  exit 1
fi


EVAL_EVERY_STEPS=""
echo $CONFIG
if [[ "$CONFIG" == "run_configs/features_apollo_ctx.json" ]]; then
    NAME=LinearGR_NOTNORMED_BASICSMOOTH_10_Sharp_slow
elif [[ "$CONFIG" == "run_configs/features_synth_ctx.json" ]]; then
    NAME=LinearGR_Synth1_BAsicDetached_NotSharp
elif [[ "$CONFIG" == "run_configs/smol_ultra_ctx.json" ]]; then
    NAME=LinearGR_smolultra_NO_INIT_NO_CTX_BASIC_SMOOTH
elif [[ "$CONFIG" == "run_configs/llama_ultra_ctx.json" ]]; then
    NAME=LinearGR_llamaultra_NO_INIT_NO_CTX_BASIC_SMOOTH
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
  NAME=SmolCtxVSLRMv2
  SCRIPT_PYTHON="vsl-rm/no_context_vsl.py"
else
  NAME=test
fi


echo "GPUs: $GPUS"
echo "Training with config $CONFIG on dataset: $DATASET with discordance_epsilon: $DISCORDANCE_EPSILON and num_train_epochs: $NUM_TRAIN_EPOCHS"
echo "Run name: ${GPUS}${NAME}"

for seed in 42; do
  bash $SCRIPT $SCRIPT_PYTHON $CONFIG --dataset=$DATASET --run_name="${GPUS}${NAME}" --seed=$seed --num_train_epochs=$NUM_TRAIN_EPOCHS $EVAL_EVERY_STEPS --discordance_epsilon=$DISCORDANCE_EPSILON $GPUDIRECTIVE --recalculate_features
done

