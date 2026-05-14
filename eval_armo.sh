
#!/bin/bash
GPUS=L40S:1
DATASET=pku

if [[ "$DATASET" == "ultra" ]]; then
  DISCORDANCE_EPSILON=0.25
elif [[ "$DATASET" == "pku" ]]; then
  DISCORDANCE_EPSILON=0.5
else
  echo "Unknown dataset: $DATASET" >&2
  exit 1
fi

echo "Evaluating ArmoRM-Llama3-8B-v0.1 on dataset: $DATASET with discordance_epsilon: $DISCORDANCE_EPSILON and num_train_epochs: $NUM_TRAIN_EPOCHS"

bash sbatch_from_json.sh run_configs/eval_armo.json --dataset=$DATASET --run_name="${GPUS}LlamaARMOEVAL" --seed=42 --num_train_epochs=1 --discordance_epsilon=$DISCORDANCE_EPSILON --gpus=$GPUS --use_frozen_base_model=True
