
from typing import Literal, LiteralString


LOCAL_DATASET_PATH: LiteralString = "/home/ubuntu/ValueLearningInGenAI/processed_datasets/"

ULTRAFEEDBACK_PROCESSED_PATH: LiteralString = LOCAL_DATASET_PATH + "ultrafeedback_pairs"
ULTRAFEEDBACK: LiteralString = "openbmb/UltraFeedback"
ULTRAFEEDBACK_EXTRA_KEYS = ["labelcontext1", "labelcontext2", "context1", "context2"]