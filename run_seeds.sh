
GPUS=L40S:2
for seed in 42 43; do
  bash sbatch_from_json.sh run_configs/llama_linear.json --dataset=ultra --run_name="{${GPUS}}LlamaVSLRM" --seed=$seed --num_train_epochs=100 --gpus=$GPUS 
done

