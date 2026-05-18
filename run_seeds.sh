#!/bin/bash
GPUS="--gpus=L40S:1"
CPU=True
if [[ "$CPU" == "True" ]]; then
  GPUS=""
  SCRIPT=cpu_from_json.sh
else
  SCRIPT=sbatch_from_json.sh
fi
DATASET=ultra
CONFIG=run_configs/llama_grounding_linear.json

EVAL_EVERY_STEPS=""
if [[ "$CONFIG" == "run_configs/llama_rlhf_linear.json" ]]; then
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
else
  NAME=test
fi
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
else
  echo "Unknown dataset: $DATASET" >&2
  exit 1
fi

echo "GPUs: $GPUS"
echo "Training with config $CONFIG on dataset: $DATASET with discordance_epsilon: $DISCORDANCE_EPSILON and num_train_epochs: $NUM_TRAIN_EPOCHS"
echo "Run name: ${GPUS}${NAME}"

for seed in 42; do
  bash $SCRIPT $CONFIG --dataset=$DATASET --run_name="${GPUS}${NAME}" --seed=$seed --num_train_epochs=$NUM_TRAIN_EPOCHS $EVAL_EVERY_STEPS --discordance_epsilon=$DISCORDANCE_EPSILON $GPUS
done

