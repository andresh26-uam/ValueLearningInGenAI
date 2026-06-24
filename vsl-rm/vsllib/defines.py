
import enum
from pathlib import Path

import json
from typing import Any, Dict, List
import numpy as np
from torch import nn

from datasets import Dataset

NO_RATING_MASK = float('-inf')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
assert PROJECT_ROOT.name == "ValueLearningInGenAI", f"Expected project root to be 'ValueLearningInGenAI', but got '{PROJECT_ROOT.name}'"
LOCAL_DATASET_PATH = str(PROJECT_ROOT / "processed_datasets")

DOWNLOADED_DATASETS_PATH = str(Path(PROJECT_ROOT) / "local_datasets")
MODEL_DIR = str(PROJECT_ROOT / "models")
RESULTS_DIR = str(PROJECT_ROOT / "results")

ULTRAFEEDBACK_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "ultrafeedback")
ULTRAFEEDBACK_EXTRA_KEYS = ["labelcontext1", "labelcontext2", "context1", "context2", "labels"]

PKUALIGNMENT_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "pkuAlignment")
PKUALIGNMENT_EXTRA_KEYS = ["labels"]

PRISM_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "prism")

OASST_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "oasst")
OASSTFL_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "oasstfl")

APOLLO_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "apollo")

PRISM_EXTRA_KEYS = ["labels", "context1", "context2", "prompt1", "prompt2", "response1", "response2", "user_id"]
OASST_EXTRA_KEYS = ["labels", "context", "user_id", "lang", "rev_count", "rank"]
APOLLO_EXTRA_KEYS = ["labels", "user_id", "context"]
class SupportedDatasets(enum.Enum):
    ULTRAFEEDBACK = "ultra"
    PKUALIGNMENT = "pku"
    PRISM = "prism"
    OASST = "oasst"
    OASSTFL = "oasstfl"
    APOLLO = "apollo"

class DatasetNames(enum.Enum):
    ULTRAFEEDBACK = "openbmb/UltraFeedback"
    PKUALIGNMENT = "PKU-Alignment/align-anything"
    PRISM = "HannahRoseKirk/prism-alignment"
    OASST = "OpenAssistant/oasst2"
    OASSTFL = "OpenAssistant/oasst2"
    APOLLO = "apollo"

EXTRA_KEYS = {
    SupportedDatasets.PKUALIGNMENT: PKUALIGNMENT_EXTRA_KEYS,
    SupportedDatasets.ULTRAFEEDBACK: ULTRAFEEDBACK_EXTRA_KEYS,
    SupportedDatasets.PRISM: PRISM_EXTRA_KEYS,
    SupportedDatasets.OASST: OASST_EXTRA_KEYS,
    SupportedDatasets.OASSTFL: OASST_EXTRA_KEYS,
    SupportedDatasets.APOLLO: APOLLO_EXTRA_KEYS
}

PROCESSED_DATASET_PATHS = {
    SupportedDatasets.PKUALIGNMENT: PKUALIGNMENT_PROCESSED_PATH,
    SupportedDatasets.ULTRAFEEDBACK: ULTRAFEEDBACK_PROCESSED_PATH,
    SupportedDatasets.PRISM: PRISM_PROCESSED_PATH,
    SupportedDatasets.OASST: OASST_PROCESSED_PATH,
    SupportedDatasets.OASSTFL: OASSTFL_PROCESSED_PATH,
    SupportedDatasets.APOLLO: APOLLO_PROCESSED_PATH
}

HAS_UNDEFINED_LABELS = {
    SupportedDatasets.PKUALIGNMENT: False,
    SupportedDatasets.ULTRAFEEDBACK: True,
    SupportedDatasets.PRISM: False,
    SupportedDatasets.OASST: False,
    SupportedDatasets.OASSTFL: False,
    SupportedDatasets.APOLLO: False,
}
HAS_CUSTOM_VAL_SETS = {
    SupportedDatasets.PKUALIGNMENT: True,
    SupportedDatasets.ULTRAFEEDBACK: 0.02,
    SupportedDatasets.PRISM: True,
    SupportedDatasets.OASST: True,
    SupportedDatasets.OASSTFL: True,
    SupportedDatasets.APOLLO: True
}

HAS_CUSTOM_TEST_SETS = {
    SupportedDatasets.PKUALIGNMENT: True,
    SupportedDatasets.ULTRAFEEDBACK: 0.1,
    SupportedDatasets.PRISM: True,
    SupportedDatasets.OASST: True,
    SupportedDatasets.OASSTFL: True,
    SupportedDatasets.APOLLO: True
}

ATTRIBUTES_ARMO_RM = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
   'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
   'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
   'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
   'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
   'code-style','code-explanation','code-instruction-following','code-readability']

VALUES_PKU = ["promptfollowing", "objectivity", "clarity", "inforichness", "safety"]
VALUES_OASST = ["quality", "nontoxicity", "humor", "helpfulness", "creativity", "nonviolence", "appropriateness"]
VALUES_OASST_ORIG = ["quality", "toxicity", "humor", "helpfulness", "creativity", "violence", "not_appropriate"]

VALUES_APOLLO = ["Cost_Efficiency", "Time_Efficienccy", "Comfort"]
ATTRIBUTES_ARMO_RM_INDEX_ULTRA = [9,8,7,6]
#ATTRIBUTES_ARMO_RM_INDEX_HELPSTEER = [0,1,2,3,4]
ATTRIBUTES_ARMO_RM_INDEX_PKU = [6, 7, 2, 3, 10] # ultrainstruct -> promptfollowing
"""We map the most related attributes from the ArmoRM dataset to approximate the PKU-Alignment values as follows:
ultrafeedback-instruction_following -> promptfollowing
ultrafeedback-truthfulness -> objectivity
helpsteer-coherence -> clarity
helpsteer-complexity -> inforichness
beavertails-is_safe -> safety
"""
REWARD_HEADS_OUTPUT = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': 'rewards',
}
REWARD_HEADS_INDICES = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': {
        SupportedDatasets.ULTRAFEEDBACK: ATTRIBUTES_ARMO_RM_INDEX_ULTRA,
        SupportedDatasets.PKUALIGNMENT: ATTRIBUTES_ARMO_RM_INDEX_PKU,
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
    "mlp": {},
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
    if "mlp" in lower_name:
         return "mlp"
    
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
    ONLY_GROUNDING_NO_LAGRANGE = "ONLY_GROUNDING_NO_LAGRANGE"
    FIRST_GROUNDING_THEN_VALUE_SYSTEM = "FIRST_GROUNDING_THEN_VALUE_SYSTEM"
    
    CTX_DEFAULT = "CTX_DEFAULT"

class ContextImplementations(enum.Enum):
    NO_CONTEXT = "NO_CONTEXT"
    BASIC = "BASIC"
    SINGLE_LEVEL_GMM = "GMM"
    NESTED_GMM = "NESTED_GMM"

class MOLossFunctionsCategories():
    REQUIRES_GRAD_ON_EVERYTHING = [MOLossFunctions.DEFAULT, MOLossFunctions.CTX_DEFAULT]
    REQUIRES_GRAD_FOR_VALUE_SYSTEM_LOSS = [MOLossFunctions.ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS, 
                                           MOLossFunctions.ONLY_VALUE_SYSTEM, 
                                           MOLossFunctions.DEFAULT,
                                           MOLossFunctions.DEFAULT_BUT_STATIC_LAGRANGE,
                                           MOLossFunctions.CTX_DEFAULT]
    EPOCH_DEPENDENT_GRAD_REQUIREMENTS = [MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM]

    REQUIRES_GRAD_FOR_ONLY_SOME_GROUNDING_LOSSES = [MOLossFunctions.ONLY_VALUES_IN_KWARGS]
    REQUIRES_GRAD_FOR_ALL_GROUNDING_LOSSES = [MOLossFunctions.ONLY_GROUNDING,  MOLossFunctions.CTX_DEFAULT, MOLossFunctions.ONLY_GROUNDING_NO_LAGRANGE, MOLossFunctions.DEFAULT,MOLossFunctions.DEFAULT_BUT_STATIC_LAGRANGE]
    REQUIRES_GRAD_FOR_SOME_OR_ALL_GROUNDING_LOSSES = REQUIRES_GRAD_FOR_ONLY_SOME_GROUNDING_LOSSES + REQUIRES_GRAD_FOR_ALL_GROUNDING_LOSSES
    
    SHOULD_APPLY_GRAD_ON_VALUE_SYSTEM_WEIGHTS = [MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM, MOLossFunctions.CTX_DEFAULT, MOLossFunctions.ONLY_VALUE_SYSTEM_AND_ONLY_WEIGHTS, MOLossFunctions.ONLY_VALUE_SYSTEM, MOLossFunctions.DEFAULT,MOLossFunctions.DEFAULT_BUT_STATIC_LAGRANGE]
    SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS = [MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM, MOLossFunctions.CTX_DEFAULT, MOLossFunctions.ONLY_GROUNDING,  MOLossFunctions.ONLY_GROUNDING_NO_LAGRANGE, MOLossFunctions.DEFAULT, MOLossFunctions.ONLY_VALUES_IN_KWARGS, MOLossFunctions.ONLY_VALUE_SYSTEM,MOLossFunctions.DEFAULT_BUT_STATIC_LAGRANGE]
    SHOULD_APPLY_GRAD_ON_PART_OF_GROUNDING_PARAMETERS = [MOLossFunctions.ONLY_VALUES_IN_KWARGS]
    SHOULD_APPLY_GRAD_ON_LAGRANGE_MULTIPLIERS = [MOLossFunctions.DEFAULT, MOLossFunctions.CTX_DEFAULT, MOLossFunctions.ONLY_GROUNDING, MOLossFunctions.ONLY_VALUES_IN_KWARGS]


    SHOULD_APPLY_GRAD_ON_GROUNDING_OR_VALUE_SYSTEM_PARAMS = SHOULD_APPLY_GRAD_ON_VALUE_SYSTEM_WEIGHTS + SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS
    
    NEEDS_NO_GRAD_EVER = [MOLossFunctions.EVALUATION_ONLY]

    CONTEXT_DEPENDENT_LOSS = [MOLossFunctions.CTX_DEFAULT]

class MOLossManagement():
    def __init__(self, loss_func_type: MOLossFunctions, loss_func_kwargs: dict | None = None) -> None:
        self.loss_func_type = MOLossFunctions(loss_func_type)
        self.loss_func_kwargs = loss_func_kwargs or {}

    def _in_category(self, category: list[MOLossFunctions]) -> bool:
        return self.loss_func_type in category

    def requires_grad_on_everything(self) -> bool:
        return self._in_category(MOLossFunctionsCategories.REQUIRES_GRAD_ON_EVERYTHING)

    def needs_no_grad_ever(self) -> bool:
        return self._in_category(MOLossFunctionsCategories.NEEDS_NO_GRAD_EVER)

    def requires_grad_for_value_system_loss(self, epoch: int = 0, **kwargs) -> bool:
        if epoch == "EVAL":
                    return False # Assumedly, evaluation step.
        if self.loss_func_type in MOLossFunctionsCategories.EPOCH_DEPENDENT_GRAD_REQUIREMENTS:
            if epoch is None:
                 raise ValueError("Epoch must be provided for loss functions with epoch-dependent grad requirements.")
            if self.loss_func_type == MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM:
                n_epochs_for_grounding = int(self.loss_func_kwargs.get('n_epochs_for_grounding', 0))
                return epoch>=n_epochs_for_grounding
        return self._in_category(MOLossFunctionsCategories.REQUIRES_GRAD_FOR_VALUE_SYSTEM_LOSS)

    def requires_grad_for_all_grounding_losses(self, epoch: int = 0, **kwargs) -> bool:
        if epoch == "EVAL":
                    return False # Assumedly, evaluation step.
        if self.loss_func_type in MOLossFunctionsCategories.EPOCH_DEPENDENT_GRAD_REQUIREMENTS:
            if epoch is None:
                 raise ValueError("Epoch must be provided for loss functions with epoch-dependent grad requirements.")
            if self.loss_func_type == MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM:
                
                n_epochs_for_grounding = int(self.loss_func_kwargs.get('n_epochs_for_grounding', 0))
                return epoch < n_epochs_for_grounding
        return self._in_category(MOLossFunctionsCategories.REQUIRES_GRAD_FOR_ALL_GROUNDING_LOSSES)

    def requires_grad_for_only_some_grounding_losses(self, epoch: int = 0, **kwargs) -> bool:
        if epoch == "EVAL":
                    return False # Assumedly, evaluation step.if self.loss_func_type in MOLossFunctionsCategories.EPOCH_DEPENDENT_GRAD_REQUIREMENTS:
                
        if self.loss_func_type == MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM:
                return False
        return self._in_category(MOLossFunctionsCategories.REQUIRES_GRAD_FOR_ONLY_SOME_GROUNDING_LOSSES)

    def requires_grad_for_some_or_all_grounding_losses(self, epoch: int = 0, **kwargs) -> bool:
        if epoch == "EVAL":
                    return False # Assumedly, evaluation step.if self.loss_func_type in MOLossFunctionsCategories.EPOCH_DEPENDENT_GRAD_REQUIREMENTS:

        if self.loss_func_type == MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM:
            if epoch is None:
                 raise ValueError("Epoch must be provided for loss functions with epoch-dependent grad requirements.")
            if self.loss_func_type == MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM:
                    n_epochs_for_grounding = int(self.loss_func_kwargs.get('n_epochs_for_grounding', 0))
                    return epoch < n_epochs_for_grounding
        return self._in_category(MOLossFunctionsCategories.REQUIRES_GRAD_FOR_SOME_OR_ALL_GROUNDING_LOSSES)

    def should_apply_grad_on_value_system_weights(self, epoch: int = None, **kwargs) -> bool:
        if epoch is not None and epoch != "EVAL":
             if self.loss_func_type == MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM:
                if self.loss_func_type == MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM:
                    n_epochs_for_grounding = int(self.loss_func_kwargs.get('n_epochs_for_grounding', 0))
                    return epoch>=n_epochs_for_grounding
        return self._in_category(MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_VALUE_SYSTEM_WEIGHTS)

    def should_apply_grad_on_grounding_parameters(self, epoch: int = None, **kwargs) -> bool:
        if epoch is not None and epoch != "EVAL":
            if self.loss_func_type == MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM:
                if self.loss_func_type == MOLossFunctions.FIRST_GROUNDING_THEN_VALUE_SYSTEM:
                    n_epochs_for_grounding = int(self.loss_func_kwargs.get('n_epochs_for_grounding', 0))
                    return epoch < n_epochs_for_grounding
        return self._in_category(MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_GROUNDING_PARAMETERS)

    def should_apply_grad_on_part_of_grounding_parameters(self, epoch: int = None, **kwargs) -> bool:
        #TODO.
        return self._in_category(MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_PART_OF_GROUNDING_PARAMETERS)

    def should_apply_grad_on_lagrange_multipliers(self, **kwargs) -> bool:
        #TODO. maybe with epoch too...
        if self.loss_func_type in MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_PART_OF_GROUNDING_PARAMETERS and len(self.loss_func_type_kwargs.get('value_indices', [])) <= 1:
            return False
        return self._in_category(MOLossFunctionsCategories.SHOULD_APPLY_GRAD_ON_LAGRANGE_MULTIPLIERS)

    def should_apply_grad_on_grounding_or_value_system_params(self, epoch: int = None, **kwargs) -> bool:
        return self.should_apply_grad_on_grounding_parameters(epoch=epoch, **kwargs) or self.should_apply_grad_on_value_system_weights(epoch=epoch, **kwargs)

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
            

def save_processeddataset(processed_path: str, ds: Dataset, validation_indices: List[int]=None, test_indices: List[int]=None):
    print(f"Total pairs after processing: {len(ds)}")
    
    # Save to OASST_PROCESSED_PATH folder
    final_output = Path(processed_path).joinpath("preprocessed/")
    final_output.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed dataset to disk... {final_output}")
    ds.save_to_disk(final_output)

    if validation_indices is not None:


        validation_indices_path = final_output / f"validation_indices.json"
        with validation_indices_path.open("w", encoding="utf-8") as f:
            json.dump(validation_indices, f)

    if test_indices is not None:
        test_indices_path = final_output / f"test_indices.json"
        with test_indices_path.open("w", encoding="utf-8") as f:
            json.dump(test_indices, f)
    
    assert set(validation_indices).isdisjoint(set(test_indices))
