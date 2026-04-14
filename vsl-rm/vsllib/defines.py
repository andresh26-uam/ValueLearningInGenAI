
from pathlib import Path


NO_RATING_MASK = float('-inf')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
assert PROJECT_ROOT.name == "ValueLearningInGenAI", f"Expected project root to be 'ValueLearningInGenAI', but got '{PROJECT_ROOT.name}'"
LOCAL_DATASET_PATH = str(PROJECT_ROOT / "processed_datasets")

ULTRAFEEDBACK_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "ultrafeedback_pairs")
ULTRAFEEDBACK: str = "openbmb/UltraFeedback"
ULTRAFEEDBACK_EXTRA_KEYS = ["labelcontext1", "labelcontext2", "context1", "context2", "labels"]

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