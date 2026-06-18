from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import random
from typing import List

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from datasets import load_dataset, Dataset, concatenate_datasets
from vsllib.defines import NO_RATING_MASK, OASST_PROCESSED_PATH, OASSTFL_PROCESSED_PATH, VALUES_OASST, VALUES_OASST_ORIG, DatasetNames, save_processeddataset

load_dotenv()

OASST_READY_TREES_FILENAME = "2023-11-05_oasst2_ready.trees.jsonl.gz"

SIGN_MAP = {
    "toxicity": -1,
    "humor": 1,
    "helpfulness": 1,
    "creativity": 1,
    "violence": -1,
    "quality": 1,
    "not_appropriate": -1
}
def extract_rating(labels_a: dict, labels_b: dict, name="quality"):
    if name not in labels_a or name not in labels_b:
        missing = 1
    else:
        missing = 0
    return labels_a.get(name, {}).get("value", NO_RATING_MASK)*SIGN_MAP[name], labels_b.get(name, {}).get("value", NO_RATING_MASK)*SIGN_MAP[name], missing

def download_ready_trees_file() -> Path:
    """Download the ready trees export from Hugging Face and cache it locally."""
    output_dir = Path(OASST_PROCESSED_PATH) / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_file = hf_hub_download(
        repo_id=DatasetNames.OASST.value,
        repo_type="dataset",
        filename=OASST_READY_TREES_FILENAME,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded_file)


def filter_message(message: dict) -> bool:
    return not message["deleted"] and message["review_result"]  #and message["lang"] == "en"

def load_ready_trees_dataset() -> tuple[pd.DataFrame, Path]:
    random.seed(42)
    """Download the ready trees file and load it into a pandas dataframe."""
    ready_trees_path = download_ready_trees_file()
    full_data = pd.read_json(ready_trees_path, lines=True, compression="gzip")

    test_data = load_dataset(DatasetNames.OASST.value, split="validation")
    test_ids = test_data.unique("message_id")

    dataset = full_data["prompt"]
    rows = []
    missing_total=0
    missing_rank=0
    total_scores=0
    for line in dataset:
        mt,mr,ts = process_line_rec(line, line, level=0, rows=rows, missing_total=missing_total, missing_rank=missing_rank, total_scores=total_scores)
        missing_total += mt
        missing_rank += mr
        total_scores += ts
    pdrows = pd.DataFrame(rows)
    pdrows = pdrows.reset_index(drop=True)

    pdrows["split"] = "train"
    pdrows["split"] = np.select(
        [pdrows["prompt_id"].isin(test_ids),],
        ["test",],
        default="train",
    )
    test_indices = pdrows[pdrows["split"] == "test"].index.tolist()
    #assert set(validation_indices).isdisjoint(set(test_indices)), "Validation and test indices should be disjoint."

    print(f"Total rows: {len(pdrows)}")
    print(f"Total train rows: {len(pdrows[pdrows['split'] == 'train'])}")
    print(f"Total test rows: {len(pdrows[pdrows['split'] == 'test'])}, {len(test_indices)}")
    
    print(f"Total missing ratings: {float(missing_total)/len(VALUES_OASST)} out of {float(total_scores)/len(VALUES_OASST)} scores ({missing_total/total_scores*100:.2f}%)")
    print(f"Total missing ranks: {missing_rank}")
    # Now delete all the instances that have missing ranks or missing scores
    pdrows = pdrows[(pdrows["score1"] != NO_RATING_MASK) & (pdrows["score2"] != NO_RATING_MASK)]
    pdrows = pdrows.reset_index(drop=True)

    for value_name in VALUES_OASST:
        pdrows = pdrows[(pdrows[f"value_{value_name}_1"] != NO_RATING_MASK) & (pdrows[f"value_{value_name}_2"] != NO_RATING_MASK)]
    pdrows = pdrows.reset_index(drop=True)
    
    # Repeat the redefinition of validation and testing indices:
    
    
    test_indices = pdrows[pdrows["split"] == "test"].index.tolist()

    validation_indices = pdrows[pdrows["split"] == "train"].index.tolist()
    random.shuffle(validation_indices)
    validation_indices = validation_indices[:len(test_indices)]



    assert max(validation_indices) < len(pdrows), "Validation indices should be within the range of the dataset."
    assert max(test_indices) < len(pdrows), "Test indices should be within the range of the dataset."
    
    assert set(validation_indices).isdisjoint(set(test_indices)), "Validation and test indices should be disjoint."
    pdrows.loc[validation_indices, "split"] = "validation"
    print("----")
    print(f"Total rows after removing missing ranks and scores: {len(pdrows)}") 
    print(f"Total train rows: {len(pdrows[pdrows['split'] == 'train'])}")
    print(f"Total validation rows: {len(pdrows[pdrows['split'] == 'validation'])}, {len(validation_indices)}")
    print(f"Total test rows: {len(pdrows[pdrows['split'] == 'test'])}, {len(test_indices)}")
    

    pdrows_fl = pdrows[pdrows["level"] == 0].reset_index(drop=True)
    pdrows_fl["split"] = "train"
    pdrows_fl["split"] = np.select(
        [pdrows_fl["prompt_id"].isin(test_ids),],
        ["test",],
        default="train",
    )
    test_indices_fl = pdrows_fl[pdrows_fl["split"] == "test"].index.tolist()

    validation_indices_fl = pdrows_fl[pdrows_fl["split"] == "train"].index.tolist()
    random.shuffle(validation_indices_fl)
    validation_indices_fl = validation_indices_fl[:len(test_indices_fl)]
    pdrows_fl.loc[validation_indices_fl, "split"] = "validation"

    print("----")
    print(f"Total rows after only first level data: {len(pdrows_fl)}") 
    print(f"Total train rows: {len(pdrows_fl[pdrows_fl['split'] == 'train'])}")
    print(f"Total validation rows: {len(pdrows_fl[pdrows_fl['split'] == 'validation'])}, {len(validation_indices_fl)}")
    print(f"Total test rows: {len(pdrows_fl[pdrows_fl['split'] == 'test'])}, {len(test_indices_fl)}")
    



    return pdrows, validation_indices, test_indices, pdrows_fl, validation_indices_fl, test_indices_fl

def process_line_rec(base_line: dict, line: dict, level: int, rows: list[dict] = [], missing_total: int = 0, missing_rank: int = 0, total_scores: int = 0) -> tuple[int, int, int]:
    
    missing_total = 0
    missing_rank = 0
    total_scores = 0
    if not filter_message(line):
        return 0,0,0
    instance = {}
    replies = [r for r  in line.get("replies", []) if filter_message(r)]

    if line["role"] == "assistant":
        for reply in replies:
            mt,mr,ts = process_line_rec(base_line, reply, level=level+1, rows=rows, missing_total=missing_total, missing_rank=missing_rank, total_scores=total_scores)
            missing_total += mt
            missing_rank += mr
            total_scores += ts
        return missing_total, missing_rank, total_scores
    assert line["role"] == "prompter", f"Line role must be 'prompter', got {line['role']}"
    if len(replies) <  2:
        return missing_total, missing_rank, total_scores

    for reply_a, reply_b in itertools.combinations(replies, 2):
        if reply_a["lang"] != line["lang"] or line["lang"] != reply_b["lang"]:
            continue
        labels_a = reply_a["labels"]
        labels_b = reply_b["labels"]
        assert reply_a["role"] == "assistant" and reply_b["role"] == "assistant", f"Replies must be from the assistant role, got {reply_a['role']} and {reply_b['role']}"
        instance["prompt"] = line["text"]
        instance["context1"] = base_line["text"] # TODO maybe use the full conversation??
        instance["context2"] = base_line["text"] # TODO maybe use the full conversation??
        instance["response1"] = reply_a["text"]
        instance["response2"] = reply_b["text"]

        instance["level"] = level
        instance["review_count"] = line["review_count"]
        instance["lang"] = line["lang"]
        instance["user_prompt_id"] = line["user_id"]
        instance["user_response1_id"] = reply_a["user_id"]
        instance["user_response2_id"] = reply_b["user_id"]

        instance["prompt_id"] = line["message_id"]
        instance["parent_id"] = line["parent_id"]

        if reply_a["rank"] is None or reply_b["rank"] is None:
            missing_rank+=1
            instance["score1"] = NO_RATING_MASK
            instance["score2"] = NO_RATING_MASK
        else:
            better_reply = int(reply_a["rank"] < reply_b["rank"])
            assert reply_a["rank"] != reply_b["rank"], f"Replies have the same rank: {reply_a['rank']} vs {reply_b['rank']}"
            instance["score1"], instance["score2"] = better_reply, 1 - better_reply
        for value_name, value_orig in zip(VALUES_OASST, VALUES_OASST_ORIG):
            instance[f"value_{value_name}_1"], instance[f"value_{value_name}_2"], missing = extract_rating(labels_a, labels_b, name=value_orig)
            missing_total+=missing
            total_scores+=1
        #assert -np.sign(reply_a["rank"] - reply_b["rank"]) == np.sign(instance["score1"] - instance["score2"]), f"Rank difference does not match score difference: rank {reply_a['rank']} vs {reply_b['rank']}, score {instance['score1']} vs {instance['score2']}"
        
        rows.append(instance)
        mt,mr,ts = process_line_rec(base_line, reply_a, level=level+1, rows=rows, missing_total=missing_total, missing_rank=missing_rank, total_scores=total_scores)
        missing_total += mt
        missing_rank += mr
        total_scores += ts
        mt,mr,ts = process_line_rec(base_line, reply_b, level=level+1, rows=rows, missing_total=missing_total, missing_rank=missing_rank, total_scores=total_scores)
        missing_total += mt
        missing_rank += mr
        total_scores += ts
    return missing_total, missing_rank, total_scores

def oasst_processor() -> tuple[int, list[int]]:
    """Fetch the OASST2 ready trees export and return its row count."""
    pddataset, validation_indices, test_indices, pddatasetfl, validation_indicesfl, test_indicesfl = load_ready_trees_dataset()

    ds_new = Dataset.from_pandas(pddataset)
    ds_newfl = Dataset.from_pandas(pddatasetfl)
    print(f"Loaded {len(pddataset)} train/val/test instances from the ready export.")

    save_processeddataset(OASST_PROCESSED_PATH, ds_new, validation_indices, test_indices)
    save_processeddataset(OASSTFL_PROCESSED_PATH, ds_newfl, validation_indicesfl, test_indicesfl)

    print(pddataset.iloc[test_indices[-1]])
    return len(pddataset), validation_indices, test_indices


if __name__ == "__main__":
    total_rows, validation_indices, test_indices = oasst_processor()

    print(f"Total merged rows: {total_rows}")
    print(f"Validation rows: {len(validation_indices)}")
    print(f"Test rows: {len(test_indices)}")
    
    

