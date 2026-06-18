

import itertools
import json
from pathlib import Path
from typing import Any

# TODO Unsure how to interpret this dataset in terms of preferences... Ideally we would want different data for grounding (MSE loss) and value system learning (Pairwise preferences)

# IMport HF_TOKEN from .env
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pandas import DataFrame
from sympy import per
load_dotenv()

import random
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm.auto import tqdm

from vsllib.defines import LOCAL_DATASET_PATH, PRISM_PROCESSED_PATH, DatasetNames, save_processeddataset


def _extract_rating(completion: dict[str, Any], value_name: str) -> Any:
    annotations = completion.get("performance_attributes") or {}
    value =  annotations.get(value_name)
    if value is None:
        return 0.0
    elif isinstance(value, int):
        return float(value)/100.0
    elif isinstance(value, float):
        return value/100.0
    else:
        print(value, type(value))
        exit(0)
        return 0.0


def _build_context(completion: dict[str, Any], user: dict[str, Any]) -> str:
    custom_system_prompt = user.get("system_string") or None
    #self_description = user.get("self_description") or None #???

    return custom_system_prompt


def _build_pair_row(completion_a: dict[str, Any], completion_b: dict[str, Any], user) -> dict[str, Any]:
    assert completion_a.get("user_id") == completion_b.get("user_id"), "User IDs do not match"
    return {
        "conversation_id1": completion_a.get("conversation_id"),
        "conversation_id2": completion_b.get("conversation_id"),
        "prompt1": completion_a.get("opening_prompt"),
        "prompt2": completion_b.get("opening_prompt"),
        "context1": _build_context(completion_a, user),
        "context2": _build_context(completion_b, user),
        "balanced1": completion_a.get("included_in_balanced_subset"),
        "balanced2": completion_b.get("included_in_balanced_subset"),
        "conv_type1": completion_a.get("conversation_type"),
        "conv_type2": completion_b.get("conversation_type"),
        "user_id": completion_a.get("user_id"),
        "response1": completion_a.get("response"),
        "response2": completion_b.get("response"),
        "model1": completion_a.get("model_name"),
        "model2": completion_b.get("model_name"),
        "provider1": completion_a.get("model_provider"),
        "provider2": completion_b.get("model_provider"),
        "score1": completion_a.get("score"),
        "score2": completion_b.get("score"),
        "value_values_1": _extract_rating(completion_a, "values"),
        "value_values_2": _extract_rating(completion_b, "values"),
        "value_fluency_1": _extract_rating(completion_a, "fluency"),
        "value_fluency_2": _extract_rating(completion_b, "fluency"),
        "value_factuality_1": _extract_rating(completion_a, "factuality"),
        "value_factuality_2": _extract_rating(completion_b, "factuality"),
        "value_safety_1": _extract_rating(completion_a, "safety"),
        "value_safety_2": _extract_rating(completion_b, "safety"),
        "value_diversity_1": _extract_rating(completion_a, "diversity"),
        "value_diversity_2": _extract_rating(completion_b, "diversity"),
        "value_creativity_1": _extract_rating(completion_a, "creativity"),
        "value_creativity_2": _extract_rating(completion_b, "creativity"),
        "value_helpfulness_1": _extract_rating(completion_a, "helpfulness"),
        "value_helpfulness_2": _extract_rating(completion_b, "helpfulness"),
        }




def prism_processor() -> tuple[int, list[int]]:
    # We take the conversations dataset. 
    # It has the groundings as "performance attributes" as well as the value system with weights. 
    # It also has the information of value system preference within the first turn of the conversation.
    # We have two choices: let the choice attributes be the value system and calculate a final value system score with that,
    # Or take the risk of using the "Score attribute and compare different p-r pairs.
    # same user maybe alone? That would be very few comparisons... lets check.
 
    # We do not take the balanced subset, it was is important for our goals.

    conversations = load_dataset(DatasetNames.PRISM.value,name='conversations')["train"]#.shuffle(seed=42)
    users = load_dataset(DatasetNames.PRISM.value,name='survey')["train"]#.shuffle(seed=42)
    users_grouped = users.to_pandas().groupby("user_id")
    rows = []
    completions = []
    # Convert to pandas and filter out users with fewer than 2 utterances
    conv_df = conversations.to_pandas()
    user_counts = conv_df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    convgrouped = conv_df[conv_df['user_id'].isin(valid_users)].groupby("user_id")

    print(f"Filtered users with >=2 utterances: {len(valid_users)}")
    for u, convos in tqdm(convgrouped, total=len(convgrouped)):
        u = users_grouped.get_group(u).iloc[0].to_dict()
        conv_binations = itertools.combinations(range(len(convos)), 2)

        for completion_a, completion_b in conv_binations:
            completions = []
            pair = convos.iloc[[completion_a, completion_b]].to_dict(orient="records")
            for sample in pair:
                #assert sample.get("user_id") == u.get("user_id")
                responses = sample.get("conversation_history")
                chosen_response = None
                i=0
                while chosen_response is None:
                    r = responses[i]
                    if r.get("if_chosen") == "true" or r.get("if_chosen") == True:
                        assert r.get("turn") == 0
                        chosen_response = r
                    i+=1
                
                sample["response"] = chosen_response.get("content")
                sample["score"] = chosen_response.get("score")
                sample["model_provider"] = chosen_response.get("model_provider")
            
                completions.append({
                    "user_id": sample.get("user_id"),
                    "opening_prompt": sample.get("opening_prompt"),
                    "conversation_type": sample.get("conversation_type"),
                    "model_name": sample.get("model_name"),
                    "response": chosen_response.get("content"),
                    "score": chosen_response.get("score"),
                    "model_provider": chosen_response.get("model_provider"),
                    "performance_attributes": sample.get("performance_attributes"),
                    "choice_attributes": sample.get("choice_attributes"),
                    "included_in_balanced_subset": sample.get("included_in_balanced_subset"),
                })
            #assert len(completions) == 2
            #assert completions[0].get("user_id") == completions[1].get("user_id")   
            #assert completions[0].get("user_id") == u.get("user_id")
            row = _build_pair_row(completions[0], completions[1], u)
            rows.append(row)
    # Create HuggingFace dataset
    hf_dataset = Dataset.from_list(rows)#from_dict( {k: [row[k] for row in rows] for k in rows[0].keys()})
    print("Total comparisons", len(hf_dataset))
    # 500 cases for preferences of known users in the train set and 500 cases for preferences of new users in the validation set.
    users = hf_dataset.unique("user_id")
    print(users)
    random.seed(42)
    random.shuffle(users)

    user_ids_random = users
    assert len(user_ids_random) == 1371, f"Total user IDs should be 1371. {len(user_ids_random)}"
    user_ids_train = user_ids_random[0:1171]
    user_ids_val = user_ids_random[1171:1272]
    user_ids_test = user_ids_random[1272:1371]

    cases_new_users_val = hf_dataset.filter(lambda x: str(x.get("user_id")) in user_ids_val).list_indexes()
    print(f"Cases new users val: {len(cases_new_users_val)}")
    cases_new_users_test = hf_dataset.filter(lambda x: str(x.get("user_id")) in user_ids_test).list_indexes()
    print(f"Cases new users test: {len(cases_new_users_test)}")
    print(cases_new_users_test[0:100])
    #TODO
    exit(0)
    n_cases_new_users_val = len(cases_new_users_val)
    n_cases_new_users_test = len(cases_new_users_test)

    dataset_not_new_users = hf_dataset.filter(lambda x: str(x.get("user_id")) in user_ids_train)

    cases_train_users_val = dataset_not_new_users[0:len(cases_new_users_val)].get("conversation_id1")
    cases_train_users_test = dataset_not_new_users[len(cases_new_users_val):len(cases_new_users_val)+len(cases_new_users_test)]

    

    for user_id, group in per_user:
        if len(group) != 1:
            print(f"User {user_id} has {len(group)} samples, expected 1.")
            exit(0)


    validation_indices = list(range(len(train_dataset), len(ds)))
    
    assert len(validation_indices) == len(val_dataset), "Validation indices length should match the validation dataset length."    

    rows = []
    skipped_amount = 0
    for i, completion_pair in enumerate(ds):
        row_or_skipped = _build_pair_row(i, completion_pair, validation_indices, None)
        if row_or_skipped is not None:
            
            rows.append(row_or_skipped)
        else:
            skipped_amount += 1
    print(f"Skipped {skipped_amount} pairs due to identical responses.")
    ds_new = Dataset.from_dict( {k: [row[k] for row in rows] for k in rows[0].keys()})
    print(f"Total pairs after processing: {len(ds_new)}")
    test_indices = list(range(len(ds_new)))[0:len(validation_indices)]  # Assuming test set is the same as validation set for now
    
    # Save to PRISM_PROCESSED_PATH folder
    save_processeddataset(PRISM_PROCESSED_PATH, ds_new, validation_indices, test_indices)


if __name__ == "__main__":
    total_rows, validation_indices, test_indices = prism_processor()
    print(f"Total merged rows: {total_rows}")
    print(f"Validation rows: {len(validation_indices)}")
    print(f"Test rows: {len(test_indices)}")