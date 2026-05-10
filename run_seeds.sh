#!/bin/bash
GPUS=L40S:1
DATASET=ultra
CONFIG=run_configs/llama_rlhf_linear.json

if [[ "$DATASET" == "ultra" ]]; then
  DISCORDANCE_EPSILON=0.25
  NUM_TRAIN_EPOCHS=10
elif [[ "$DATASET" == "pku" ]]; then
  DISCORDANCE_EPSILON=0.5
  NUM_TRAIN_EPOCHS=50
else
  echo "Unknown dataset: $DATASET" >&2
  exit 1
fi

echo "GPUs: $GPUS"
echo "Training with config $CONFIG on dataset: $DATASET with discordance_epsilon: $DISCORDANCE_EPSILON and num_train_epochs: $NUM_TRAIN_EPOCHS"

for seed in 42 43 44 45; do
  bash sbatch_from_json.sh $CONFIG --dataset=$DATASET --run_name="${GPUS}LlamaRLHFBTRM" --seed=$seed --num_train_epochs=$NUM_TRAIN_EPOCHS --discordance_epsilon=$DISCORDANCE_EPSILON --gpus=$GPUS 
done

