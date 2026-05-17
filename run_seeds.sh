#!/bin/bash
GPUS="--gpus=L40S:1"
CPU=True
if [[ "$CPU" == "True" ]]; then
  GPUS=""
  SCRIPT=cpu_from_json.sh
else
  SCRIPT=sbatch_from_json.sh
fi
DATASET=pku
CONFIG=run_configs/llama_linear.json

if [[ "$CONFIG" == "run_configs/llama_rlhf_linear.json" ]]; then
  NAME=LlamaBTRM
elif [[ "$CONFIG" == "run_configs/llama_linear.json" ]]; then
  NAME=LlamaVSLRM_NONORMAL_SMOOTHER
elif [[ "$CONFIG" == "run_configs/llama_grounding_linear.json" ]]; then
  NAME=LlamaGRRM
  elif [[ "$CONFIG" == "run_configs/llama_nolag_linear.json" ]]; then
  NAME=LlamaVSL-NL-RM
elif [[ "$CONFIG" == "run_configs/smol_rlhf_linear.json" ]]; then
  NAME=SmolBTRM
elif [[ "$CONFIG" == "run_configs/smol_linear.json" ]]; then
  NAME=SmolVSLRM
else
  NAME=test
fi
if [[ "$DATASET" == "ultra" ]]; then
  DISCORDANCE_EPSILON=0.25
  NUM_TRAIN_EPOCHS=10
elif [[ "$DATASET" == "pku" ]]; then
  DISCORDANCE_EPSILON=0.5
  NUM_TRAIN_EPOCHS=100
else
  echo "Unknown dataset: $DATASET" >&2
  exit 1
fi

echo "Script: $SCRIPT"
echo "GPUs: $GPUS"
echo "Training with config $CONFIG on dataset: $DATASET with discordance_epsilon: $DISCORDANCE_EPSILON and num_train_epochs: $NUM_TRAIN_EPOCHS"
echo "Run name: ${GPUS}${NAME}"

for seed in 42; do
  bash $SCRIPT $CONFIG --dataset=$DATASET --run_name="${GPUS}${NAME}" --seed=$seed --num_train_epochs=$NUM_TRAIN_EPOCHS --discordance_epsilon=$DISCORDANCE_EPSILON $GPUS 
done

