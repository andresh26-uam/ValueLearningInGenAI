
import enum
from pathlib import Path

import json
from typing import Any, Dict
import numpy as np
from torch import nn

NO_RATING_MASK = float('-inf')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
assert PROJECT_ROOT.name == "ValueLearningInGenAI", f"Expected project root to be 'ValueLearningInGenAI', but got '{PROJECT_ROOT.name}'"
LOCAL_DATASET_PATH = str(PROJECT_ROOT / "processed_datasets")
MODEL_DIR = str(PROJECT_ROOT / "models")
RESULTS_DIR = str(PROJECT_ROOT / "results")


ULTRAFEEDBACK_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "ultrafeedback")
ULTRAFEEDBACK_EXTRA_KEYS = ["labelcontext1", "labelcontext2", "context1", "context2", "labels"]

PKUALIGNMENT_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "pkuAlignment")
PKUALIGNMENT_EXTRA_KEYS = ["labels"]

class SupportedDatasets(enum.Enum):
    ULTRAFEEDBACK = "ultra"
    PKUALIGNMENT = "pku"

class DatasetNames(enum.Enum):
    ULTRAFEEDBACK = "openbmb/UltraFeedback"
    PKUALIGNMENT = "PKU-Alignment/align-anything"


EXTRA_KEYS = {
    SupportedDatasets.PKUALIGNMENT: PKUALIGNMENT_EXTRA_KEYS,
    SupportedDatasets.ULTRAFEEDBACK: ULTRAFEEDBACK_EXTRA_KEYS,
}

PROCESSED_DATASET_PATHS = {
    SupportedDatasets.PKUALIGNMENT: PKUALIGNMENT_PROCESSED_PATH,
    SupportedDatasets.ULTRAFEEDBACK: ULTRAFEEDBACK_PROCESSED_PATH,
}

HAS_UNDEFINED_LABELS = {
    SupportedDatasets.PKUALIGNMENT: False,
    SupportedDatasets.ULTRAFEEDBACK: True,
}
HAS_CUSTOM_VAL_SETS = {
    SupportedDatasets.PKUALIGNMENT: True,
    SupportedDatasets.ULTRAFEEDBACK: 0.05,
}

HAS_CUSTOM_TEST_SETS = {
    SupportedDatasets.PKUALIGNMENT: True,
    SupportedDatasets.ULTRAFEEDBACK: 0.1,
}

ATTRIBUTES_ARMO_RM = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
   'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
   'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
   'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
   'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
   'code-style','code-explanation','code-instruction-following','code-readability']
ATTRIBUTES_ARMO_RM_INDEX_ULTRA = [9,8,7,6]
ATTRIBUTES_ARMO_RM_INDEX_HELPSTEER = [0,1,2,3,4]
REWARD_HEADS_OUTPUT = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': 'rewards',
}
REWARD_HEADS_INDICES = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': {
        SupportedDatasets.ULTRAFEEDBACK: ATTRIBUTES_ARMO_RM_INDEX_ULTRA,
        SupportedDatasets.PKUALIGNMENT: None,
    }
}
NUM_OBJECTIVES = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': 19,
}
VALUE_SYSTEM_OUTPUT = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': 'score',
}

MIN_EPSILON = 4.0e-2
SCORE_DIFF_EPSILON = 1.0/(1+np.exp(-MIN_EPSILON)) -0.5 # The difference in score that corresponds to a difference in target probability of epsilon, according to the Bradley-Terry model.
# 0.00999.

VALUE_LAYER_ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "Softplus": nn.Softplus,
    "SiLU": nn.SiLU,
    "none": None,
}

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "gemma": {
        "tokenizer_use_fast": None,
        "tokenizer_add_pad_token": False,
        "use_flash_attention_2": False,
        "tokenizer_use_auth_token": True,
    },
    "llama3": {
        "tokenizer_use_fast": False,
        "tokenizer_add_pad_token": True,
        "use_flash_attention_2": False,
        "tokenizer_use_auth_token": False,
    },
    "mistral": {
        "tokenizer_use_fast": False,
        "tokenizer_add_pad_token": True,
        "use_flash_attention_2": False,
        "tokenizer_use_auth_token": False,
    },
    "smol": {
        "tokenizer_use_fast": False,
        "tokenizer_add_pad_token": True,
        "use_flash_attention_2": False,
        "tokenizer_use_auth_token": False,
    },
}
def infer_variant(model_name: str, requested_variant: str) -> str:
    if requested_variant and requested_variant != "auto":
        variant = requested_variant.lower()
        if variant not in MODEL_PRESETS:
            raise ValueError(
                f"Unsupported model_variant={requested_variant}. "
                f"Expected one of: auto, gemma, llama3, mistral"
            )
        return variant

    lower_name = model_name.lower()
    if "gemma" in lower_name:
        return "gemma"
    if "llama" in lower_name:
        return "llama3"
    if "mistral" in lower_name:
        return "mistral"
    if "smol" in lower_name:
        return "smol"
    
    raise ValueError(
        "Could not infer model variant from model_name. "
        "Please set --model_variant to one of: gemma, llama3, mistral"
    )



class MOLossFunctions(enum.Enum):
    DEFAULT = "DEFAULT"
    ONLY_GROUNDING = "ONLY_GROUNDING"
    ONLY_VALUE_SYSTEM = "ONLY_VALUE_SYSTEM"
    ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS = "ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS"
    ONLY_VALUES_IN_KWARGS = "ONLY_VALUES_IN_KWARGS"
    EVALUATION_ONLY = "EVALUATION_ONLY"
    DEFAULT_BUT_STATIC_LAGRANGE = "DEFAULT_BUT_STATIC_LAGRANGE"
    

class MOLossFunctionsCategories():
    REQUIRES_GRAD_ON_EVERYTHING = [MOLossFunctions.DEFAULT]
    REQUIRES_GRAD_FOR_VALUE_SYSTEM_LOSS = [MOLossFunctions.ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS, 
                                           MOLossFunctions.ONLY_VALUE_SYSTEM, 
                                           MOLossFunctions.DEFAULT,
                                           MOLossFunctions.DEFAULT_BUT_STATIC_LAGRANGE]
    
    REQUIRES_GRAD_FOR_ONLY_SOME_GROUNDING_LOSSES = [MOLossFunctions.ONLY_VALUES_IN_KWARGS]
    REQUIRES_GRAD_FOR_ALL_GROUNDING_LOSSES = [MOLossFunctions.ONLY_GROUNDING, MOLossFunctions.DEFAULT,MOLossFunctions.DEFAULT_BUT_STATIC_LAGRANGE]
    REQUIRES_GRAD_FOR_SOME_OR_ALL_GROUNDING_LOSSES = REQUIRES_GRAD_FOR_ONLY_SOME_GROUNDING_LOSSES + REQUIRES_GRAD_FOR_ALL_GROUNDING_LOSSES
    
    SHOULD_APPLY_GRAD_ON_VALUE_SYSTEM_WEIGHTS = [MOLossFunctions.ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS, MOLossFunctions.ONLY_VALUE_SYSTEM, MOLossFunctions.DEFAULT,MOLossFunctions.DEFAULT_BUT_STATIC_LAGRANGE]
    SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS = [MOLossFunctions.ONLY_GROUNDING, MOLossFunctions.DEFAULT, MOLossFunctions.ONLY_VALUES_IN_KWARGS, MOLossFunctions.ONLY_VALUE_SYSTEM,MOLossFunctions.DEFAULT_BUT_STATIC_LAGRANGE]
    SHOULD_APPLY_GRAD_ON_PART_OF_GROUNDING_PARAMETERS = [MOLossFunctions.ONLY_VALUES_IN_KWARGS]
    SHOULD_APPLY_GRAD_ON_LAGRANGE_MULTIPLIERS = [MOLossFunctions.DEFAULT, MOLossFunctions.ONLY_GROUNDING, MOLossFunctions.ONLY_VALUES_IN_KWARGS]


    SHOULD_APPLY_GRAD_ON_GROUNDING_OR_VALUE_SYSTEM_PARAMS = SHOULD_APPLY_GRAD_ON_VALUE_SYSTEM_WEIGHTS + SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS
    
    NEEDS_NO_GRAD_EVER = [MOLossFunctions.EVALUATION_ONLY]


def get_validation_indices(dataset_name: SupportedDatasets) -> list[int] | float | None:
    maybe_proportion = HAS_CUSTOM_VAL_SETS[dataset_name]
    if isinstance(maybe_proportion, float):
        return maybe_proportion
    else:
        validation_indices_path = Path(PROCESSED_DATASET_PATHS[dataset_name]) / "preprocessed" / "validation_indices.json"
        if validation_indices_path.exists():
            with validation_indices_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    
def get_test_indices(dataset_name: SupportedDatasets) -> list[int] | float | None:
    maybe_proportion = HAS_CUSTOM_TEST_SETS[dataset_name]
    if isinstance(maybe_proportion, float):
        return maybe_proportion
    else:
        test_indices_path = Path(PROCESSED_DATASET_PATHS[dataset_name]) / "preprocessed" / "test_indices.json"
        if test_indices_path.exists():
            with test_indices_path.open("r", encoding="utf-8") as f:
                return json.load(f)