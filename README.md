# Value System Learning in Generative AI

This repository explores the problem of learning value systems using reward models from preference datasets. 

Our main algorithm, VSL-RM observes a preference dataset of prompt-response pairs based on both a given set of value labels and on the personal preference of diverse agents with different value systems (multiobjective preference weights). Then, it simultaneously learns a reward vector model that implements a value alignment specification for the given set of values and a set of preference weights (value system) that describes the personal preferences observed.

We will be exploring a context dependent variant of the problem when the set of weights will vary depending explicitly on contextual conditions given in the prompts.

Currently, we have experiments trained on the [Ultrafeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) and [PKU-Alignment (Align-Anything, text-to-text)](https://huggingface.co/datasets/PKU-Alignment/align-anything) datasets.

Trained models will be published shortly.

## Installation from GitHub

1. Clone this repository in the main folder. 
    - `git clone https://github.com/andresh26-uam/ValueLearningInGenAI.git`
    - `cd ValueLearningInGenAI`
2. Create a virtual environment with Python 3.10+ (can be inside the repo folder). We used 3.12.3 in the paper.
    - `python3.12 -m venv .venv`
    - `source .venv/bin/activate`
3. Requirements. Use the normal version, preferably, but if you have a Python version older than 3.12, you might need the minimal version that uses an older Transformers library.
    - `pip install -r requirements.txt` OR `pip install -r requirements_minimal.txt`


## Available actions

### Preprocessing preference datasets
We need to preprocess the datasets before training, and calculate the last hidden states of every prompt to speed up the training process. WARNING: DO NOT USE ACCELERATE HERE!!
- `python -O vsl-rm/pkualignment_preprocess.py` OR `python vsl-rm/ultrafeedback_preprocess.py` (These will parse the datasets into a common format for training purposes)
- `python -O vsl-rm/no_context_vsl.py --do_train=False --use_embeddings --recalculate_embeddings --retokenize --dataset=<ultra_or_pku>` (This will generate the embeddings of each prompt-response pair and save them for training)

### Training
We use SLURM commands in our available setup, but you can use this general command instead. Have the environment variable PYTHONOPTIMIZE=1 to avoid assertions and substantially decrease runtime.

- `PYTHONOPTIMIZE=1 accelerate launch vsl-rm/no_context_vsl.py --config_file=<select_one_from_run_configs_folder> --dataset=<ultra_or_pku> --run_name=<your_own_wandb_run_name> <optional_flags>` 
    -  Complete example: `accelerate launch vsl-rm/no_context_vsl.py  --config_file="run_configs/llama_linear.rlhf" --dataset=pku --run_name="Test" --num_train_epochs=10` 

Training is perfectly feasible in CPU, will assume full Float32 precision. Just add `--use_cpu` as an optional flag at the end of the previous training commands, and make sure to change to a different accelerator configuration file (e.g. `accelerate launch --config_file=accelerate_config/cpu_config.yaml vsl-rm/no_context_vsl.py...`). See the [cpu_from_json.sh](https://github.com/andresh26-uam/ValueLearningInGenAI/blob/main/cpu_from_json.sh) file for an example.
 
Regarding the `run_configs.json` folder, the `llama_linear.json` has the configuration to run the proposed training algorithm using the Llama-based base model from [Armo-RM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1). There is a counterpart `smol_linear.json` that trains the proposed model using [SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct) as base model. There are other variants used to train the baselines used in our experiments.

## Known Issues

Depending on your Transformers library version, you might get this error when running any training script using the Armo-RM as base model.

`ImportError: cannot import name 'LLAMA_INPUTS_DOCSTRING' from 'transformers.models.llama.modeling_llama'`

To solve this, I think it is best to just remove the import statement in the model's file, as well as its use in a function decorator (that justs adds a docstring).