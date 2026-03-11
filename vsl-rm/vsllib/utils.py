
# We need to define a special data collator that batches the data in our j vs k format.
from dataclasses import dataclass
from typing import Dict, List, Optional, Optional, Union

from typing import Any
from transformers import AutoTokenizer


from transformers.utils import PaddingStrategy


@dataclass
class MORewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []

        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_1"],
                    "attention_mask": feature["attention_mask_1j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_2"],
                    "attention_mask": feature["attention_mask_2"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch
