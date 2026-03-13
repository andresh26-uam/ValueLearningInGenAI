
# We need to define a special data collator that batches the data in our j vs k format.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers import AutoTokenizer


from transformers.utils import PaddingStrategy

import torch as th
import numpy as np
@dataclass
class MORewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        merged_labels = []
        for feature in features:
            pair_labels = feature["labels"]
            if hasattr(pair_labels, "tolist"):
                pair_labels = pair_labels.tolist()

            merged_features.append(
                {
                    "input_ids": feature["input_ids_1"],
                    "attention_mask": feature["attention_mask_1"],
                }
            )
            merged_labels.append(pair_labels[0])
            merged_labels.append(pair_labels[1])
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

        labels_np = np.asarray(merged_labels, dtype=np.float32)
        if self.return_tensors == "pt":
            labels = th.as_tensor(labels_np, dtype=th.float32)
        elif self.return_tensors == "np":
            labels = labels_np
        elif self.return_tensors == "tf":
            labels = labels_np
        else:
            labels = merged_labels
        
        assert batch["input_ids"].shape[:-1] == labels.shape[:-1], f"Input IDs shape: {batch['input_ids'].shape}, Labels shape: {labels.shape}"
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
            #"labels": th.cat([th.as_tensor(np.array(f['labels'], dtype=np.float16), dtype=th.float16) for f in features], dim=0).to(batch["input_ids"].device),
			#"score": th.tensor([f.get("score", 0.0) for f in merged_features], dtype=th.float32),
			#"value_ratings": [f.get("value_ratings", {}) for f in merged_features],
            "return_loss": True,
        }
        return batch
