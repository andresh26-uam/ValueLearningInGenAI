#!/bin/bash
GPUS=L40S:1
DATASET=ultra
CONFIG=run_configs/llama_linear_firstgr_thenvs.json

EVAL_EVERY_STEPS=""
if [[ "$CONFIG" == "run_configs/llama_rlhf_linear.json" ]]; then
  NAME=LlamaBTRM
elif [[ "$CONFIG" == "run_configs/llama_linear_firstgr_thenvs.json" ]]; then
  NAME=LlamaSEQ-RM
elif [[ "$CONFIG" == "run_configs/llama_linear.json" ]]; then
  NAME=LlamaVSLRM
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
  bash sbatch_from_json.sh $CONFIG --dataset=$DATASET --run_name="${GPUS}${NAME}" --seed=$seed --num_train_epochs=$NUM_TRAIN_EPOCHS $EVAL_EVERY_STEPS --discordance_epsilon=$DISCORDANCE_EPSILON --gpus=$GPUS
done

