

import json
from pathlib import Path
from typing import Any

# IMport HF_TOKEN from .env
from dotenv import load_dotenv
import numpy as np
load_dotenv()

from datasets import load_dataset, Dataset, concatenate_datasets

from vsllib.defines import LOCAL_DATASET_PATH, PKUALIGNMENT_PROCESSED_PATH, DatasetNames, save_processeddataset


def _build_pair_row(index_, completion_pair: dict[str, Any], val_indices, test_indices) -> dict[str, Any]:
    """
    if completion_pair.get("response_1") == completion_pair.get("response_2"):
        print(f"Warning: Found identical responses for question: {completion_pair.get('question')}. Skipping this pair.")
        print(f"Response: {completion_pair.get('response_1')}")
        print(f"Response: {completion_pair.get('response_2')}")
        print(f"Overall response: {completion_pair.get('overall_response')}")
        if val_indices is not None and index_ in val_indices:
            where_index = val_indices.index(index_)
            val_indices.remove(index_)
            for i in range(where_index, len(val_indices)):
                val_indices[i] -= 1
        if test_indices is not None and index_ in test_indices:
            where_index = test_indices.index(index_)
            test_indices.remove(index_)
            for i in range(where_index, len(test_indices)):
                test_indices[i] -= 1
        return None
        raise ValueError("Responses should differ between the two completions")"""

    return {
        "source": None,
        "prompt": completion_pair.get("question"),
        "response1": completion_pair.get("response_1"),
        "response2": completion_pair.get("response_2"),
        "score1": 1.0 if completion_pair.get("overall_response") == 1 else 0.0,
        "score2": 1.0 if completion_pair.get("overall_response") == 2 else 0.0,
        "value_pkupromptfollowing_1": completion_pair.get("prompt_following_rate_1"),
        "value_pkupromptfollowing_2": completion_pair.get("prompt_following_rate_2"),
        "value_pkuobjectivity_1": completion_pair.get("objective_rules_rate_1"),
        "value_pkuobjectivity_2": completion_pair.get("objective_rules_rate_2"),
        "value_pkuclarity_1": completion_pair.get("clarity_rate_1"),
        "value_pkuclarity_2": completion_pair.get("clarity_rate_2"),
        "value_pkuinforichness_1": completion_pair.get("information_richness_rate_1"),
        "value_pkuinforichness_2": completion_pair.get("information_richness_rate_2"),
        "value_pkusafety_1": completion_pair.get("safety_rate_1"),
        "value_pkusafety_2": completion_pair.get("safety_rate_2"), 
    }
def pkuAlignment_processor() -> tuple[int, list[int]]:

    train_dataset = load_dataset(DatasetNames.PKUALIGNMENT.value,name='text-to-text')['train']
    val_dataset = load_dataset(DatasetNames.PKUALIGNMENT.value,name='text-to-text')['val']

    
    ds = concatenate_datasets([train_dataset, val_dataset])

    
    validation_indices = list(range(len(train_dataset), len(ds)))
    
    rows = []
    skipped_amount = 0
    for i, completion_pair in enumerate(ds):
        row_or_skipped = _build_pair_row(i, completion_pair, validation_indices, None)
        if row_or_skipped is not None:
            
            rows.append(row_or_skipped)
        else:
            skipped_amount += 1
    
    assert len(validation_indices) == len(val_dataset), "Validation indices length should match the validation dataset length."    

    test_indices = list(range(len(rows)))[0:len(validation_indices)]  # Assuming test set is the same as validation set for now
    
    print(f"Skipped {skipped_amount} pairs due to identical responses.")
    ds_new = Dataset.from_dict( {k: [row[k] for row in rows] for k in rows[0].keys()})

    print(f"Total pairs after processing: {len(ds_new)}")
    
    save_processeddataset(PKUALIGNMENT_PROCESSED_PATH, ds_new, validation_indices, test_indices)
    # Save to PKUALIGNMENT_PROCESSED_PATH folder
    
    return len(rows), validation_indices, test_indices


if __name__ == "__main__":
    total_rows, validation_indices, test_indices = pkuAlignment_processor()
    print(f"Total merged rows: {total_rows}")
    print(f"Validation rows: {len(validation_indices)}")
    print(f"Test rows: {len(test_indices)}")