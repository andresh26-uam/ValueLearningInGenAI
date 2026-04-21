
import enum
from pathlib import Path

import json


NO_RATING_MASK = float('-inf')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
assert PROJECT_ROOT.name == "ValueLearningInGenAI", f"Expected project root to be 'ValueLearningInGenAI', but got '{PROJECT_ROOT.name}'"
LOCAL_DATASET_PATH = str(PROJECT_ROOT / "processed_datasets")


ULTRAFEEDBACK_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "ultrafeedback_pairs")
ULTRAFEEDBACK_EXTRA_KEYS = ["labelcontext1", "labelcontext2", "context1", "context2", "labels"]

PKUALIGNMENT_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "pkuAlignment_pairs")
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

TRAIN_PATHS = {
    SupportedDatasets.PKUALIGNMENT: PKUALIGNMENT_PROCESSED_PATH,
    SupportedDatasets.ULTRAFEEDBACK: ULTRAFEEDBACK_PROCESSED_PATH,
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
ATTRIBUTES_ARMO_RM_INDEX = [9,8,7,6]
REWARD_HEADS_OUTPUT = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': 'rewards',
}
REWARD_HEADS_INDICES = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': ATTRIBUTES_ARMO_RM_INDEX,
}
NUM_OBJECTIVES = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': 19,
}
VALUE_SYSTEM_OUTPUT = {
    'RLHFlow/ArmoRM-Llama3-8B-v0.1': 'score',
}


def get_validation_indices(dataset_name: SupportedDatasets) -> list[int] | float | None:
    maybe_proportion = HAS_CUSTOM_VAL_SETS[dataset_name]
    if isinstance(maybe_proportion, float):
        return maybe_proportion
    else:
        validation_indices_path = Path(TRAIN_PATHS[dataset_name]) / "validation_indices.json"
        if validation_indices_path.exists():
            with validation_indices_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    
def get_test_indices(dataset_name: SupportedDatasets) -> list[int] | float | None:
    maybe_proportion = HAS_CUSTOM_TEST_SETS[dataset_name]
    if isinstance(maybe_proportion, float):
        return maybe_proportion
    else:
        test_indices_path = Path(TRAIN_PATHS[dataset_name]) / "test_indices.json"
        if test_indices_path.exists():
            with test_indices_path.open("r", encoding="utf-8") as f:
                return json.load(f)