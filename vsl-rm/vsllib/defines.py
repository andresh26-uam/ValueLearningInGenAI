
from pathlib import Path


NO_RATING_MASK = float('-inf')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
assert PROJECT_ROOT.name == "ValueLearningInGenAI", f"Expected project root to be 'ValueLearningInGenAI', but got '{PROJECT_ROOT.name}'"
LOCAL_DATASET_PATH = str(PROJECT_ROOT / "processed_datasets")

ULTRAFEEDBACK_PROCESSED_PATH = str(Path(LOCAL_DATASET_PATH) / "ultrafeedback_pairs")
ULTRAFEEDBACK: str = "openbmb/UltraFeedback"
ULTRAFEEDBACK_EXTRA_KEYS = ["labelcontext1", "labelcontext2", "context1", "context2", "labels"]