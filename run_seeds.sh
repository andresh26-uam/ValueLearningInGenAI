#!/bin/bash


GPUS="L40S:1"
CPU=True

CONFIG="run_configs/smol_ctx_linear_ae_kmeans.json"
DATASET="pku"
DISCORDANCE_EPSILON=""
NUM_TRAIN_EPOCHS_OVERRIDE=""
SEEDS=(42)
RUN_INDEX=""
RUN_NAME_FILE=""
MODEL_ALIAS="smol"


DEBUG=()
SAVE=(--do_save=True --do_checkpointing=False)
REPORT_TO=wandb

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift
      ;;
    --config=*)
      CONFIG="${1#*=}"
      ;;
    --dataset)
      DATASET="$2"
      shift
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      ;;
    --discordance_epsilon)
      DISCORDANCE_EPSILON="$2"
      shift
      ;;
    --discordance_epsilon=*)
      DISCORDANCE_EPSILON="${1#*=}"
      ;;
    --num_train_epochs)
      NUM_TRAIN_EPOCHS_OVERRIDE="$2"
      shift
      ;;
    --num_train_epochs=*)
      NUM_TRAIN_EPOCHS_OVERRIDE="${1#*=}"
      ;;
    --seed)
      SEEDS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SEEDS+=("$1")
        shift
      done
      continue
      ;;
    --seed=*)
      SEEDS=("${1#*=}")
      ;;
    --run_index)
      RUN_INDEX="$2"
      shift
      ;;
    --run_index=*)
      RUN_INDEX="${1#*=}"
      ;;
    --run_name_file)
      RUN_NAME_FILE="$2"
      shift
      ;;
    --run_name_file=*)
      RUN_NAME_FILE="${1#*=}"
      ;;
    --model)
      MODEL_ALIAS="$2"
      shift
      ;;
    --model=*)
      MODEL_ALIAS="${1#*=}"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done


if [[ "$CPU" == "True" ]]; then
    GPUS=""
    GPUDIRECTIVE=()
    SCRIPT=cpu_from_json.sh
else
    GPUDIRECTIVE=(--gpus="$GPUS")
    SCRIPT=sbatch_from_json.sh
fi

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

  [[ -n "$DISCORDANCE_EPSILON" ]] || DISCORDANCE_EPSILON=0.25
  training_initialization_data_size=5000
  NUM_TRAIN_EPOCHS=10
  if [[ "$CONFIG" == *"llama_linear_firstgr_thenvs.json"* ]]; then
    NUM_TRAIN_EPOCHS=15
    fi
elif [[ "$DATASET" == "pku" ]]; then
  [[ -n "$DISCORDANCE_EPSILON" ]] || DISCORDANCE_EPSILON=0.5

  training_initialization_data_size=10000
  NUM_TRAIN_EPOCHS=100
  if [[ "$CONFIG" == *"llama_linear_firstgr_thenvs.json"* ]]; then
    NUM_TRAIN_EPOCHS=150
  fi
elif [[ "$DATASET" == "apollo" ]]; then
  [[ -n "$DISCORDANCE_EPSILON" ]] || DISCORDANCE_EPSILON=0.04
  training_initialization_data_size=2000
  NUM_TRAIN_EPOCHS=5000
elif [[ "$DATASET" == "synth" ]]; then
  [[ -n "$DISCORDANCE_EPSILON" ]] || DISCORDANCE_EPSILON=-1
  training_initialization_data_size=1000
  NUM_TRAIN_EPOCHS=500
else
  echo "Unknown dataset: $DATASET"
  exit 1
fi

if [[ -n "$NUM_TRAIN_EPOCHS_OVERRIDE" ]]; then
  NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS_OVERRIDE"
fi

if [[ "$CTX_IMPLEMENTATION" == "KMEANS_THEN_VS" ]]; then
        training_initialization_data_size="all"
else
    training_initialization_data_size=10000
fi

echo $CONFIG
if [[ "$CONFIG" == *"features_apollo_ctx.json"* ]]; then
    NAME=ApolloVSLRMv17
elif [[ "$CONFIG" == *"features_synth_ctx.json"* ]]; then
    NAME=SynthV17_savetest
elif [[ "$CONFIG" == *"llama_ctx_linear.json"* ]]; then
    NAME=LlamaCtxVSLRMv5
elif [[ "$CONFIG" == *"llama_rlhf_linear.json"* ]]; then
  NAME=LlamaBTRMv2
elif [[ "$CONFIG" == *"llama_linear_firstgr_thenvs.json"* ]]; then
  NAME=LlamaSEQ-RMv2
elif [[ "$CONFIG" == *"llama_linear.json"* ]]; then
  NAME=LlamaVSLRMv2
elif [[ "$CONFIG" == *"llama_grounding_linear.json"* ]]; then
  NAME=LlamaGRRMv2
elif [[ "$CONFIG" == *"llama_nolag_linear.json"* ]]; then
  NAME=LlamaVSL-NL-RMv2
elif [[ "$CONFIG" == *"smol_rlhf_linear.json"* ]]; then
  NAME=SmolBTRMv2
elif [[ "$CONFIG" == *"smol_linear.json"* ]]; then
  NAME=SmolVSLRMv2
elif [[ "$CONFIG" == *"smol_ctx_linear.json"* ]]; then
  NAME=SmolCtxVSLRMv17_AEDET
elif [[ "$CONFIG" == *"smol_ctx_linear_ae_kmeans.json"* ]]; then
  NAME=SmolCtxVSLRMv17_AE
elif [[ "$CONFIG" == *"smol_ctx_linear_gmm_and_classifier.json"* ]]; then
  NAME=SmolCtxVSLRMv17_GMMC
elif [[ "$CONFIG" == *"smol_ctx_linear_baseline.json"* ]]; then
  NAME=SmolCtxVSLRMv17_DIRECTVS
elif [[ "$CONFIG" == *"smol_ctx_linear_baseline_smooth.json"* ]]; then
  NAME=SmolCtxVSLRMv17_BS
else
  NAME=test
fi

NAME="${NAME}_ST_${SENTENCE_TRANS}_${CTX_IMPLEMENTATION}_INITVS_${VS_WEIGHT_INIT}_INITCTX_${DOINIT}_INITVSGR_${DOINIT_VS}_VS_${VS_COEFF}_CTX_${CTX_COEFF}_TM_${TEMP}_LC_${LAMBDA}_NORM_${NORM}_DIRCVS_${DIRCVS}"
if [[ -n "$RUN_INDEX" ]]; then
  NAME="${NAME}_RUN_${RUN_INDEX}"
fi
if [[ -n "$RUN_NAME_FILE" ]]; then
  printf '%s\n' "$NAME" > "$RUN_NAME_FILE"
fi
echo "GPUs: $GPUS"
echo "Training with config $CONFIG"
echo "On dataset: $DATASET with discordance_epsilon: $DISCORDANCE_EPSILON"
echo "and num_train_epochs: $NUM_TRAIN_EPOCHS"
echo "Run name: ${GPUS}${NAME}"

for seed in "${SEEDS[@]}"; do
    # Create a unique temp file for this run to write its wandb run name
    RUN_NAME_TEMP_FILE=$(mktemp)

    args=(
        "$SCRIPT_PYTHON"
        "$CONFIG"
        "--dataset=$DATASET"
        "--run_name=${GPUS}${NAME}"
        "--seed=$seed"
        "--run_name_file=$RUN_NAME_TEMP_FILE"
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

    # Read the run name that was written by the training process
    if [[ -f "$RUN_NAME_TEMP_FILE" ]]; then
        CHECKPOINT_NAME=$(cat "$RUN_NAME_TEMP_FILE")
        rm "$RUN_NAME_TEMP_FILE"
    else
        echo "Error: Run name file was not created by training script" >&2
        exit 1
    fi
done

bash cpu_eval.sh \
    --model "$MODEL_ALIAS" \
    --dataset "$DATASET" \
    "--checkpoint_path=${CHECKPOINT_NAME}" \
    --do_llm_summarization=False \
    --tsne_perplexity=30 \
    --tsne_seed=0
