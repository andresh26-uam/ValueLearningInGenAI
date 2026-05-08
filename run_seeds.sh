for seed in 42 43 44 45 46; do
  bash sbatch_from_json.sh run_configs/llama_linear.json --dataset=ultra --run_name=GPULlamaVSLRM --seed=$seed --num_train_epochs=50
done

