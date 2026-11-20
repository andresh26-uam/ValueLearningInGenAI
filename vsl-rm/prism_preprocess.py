

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
    
    #self_description = user.get("self_description") or None #???
    profile = {
        "age": user.get("age") or None,
        "gender": user.get("gender") or None,
        "employment_status": user.get("employment_status") or None,
        "education": user.get("education") or None,
        "marital_status": user.get("marital_status") or None,
        "study_locale": user.get("study_locale") or None,
        "ethnicity": user["ethnicity"]["categorised"],
        "religion": user["religion"]["categorised"],
        "location": user["location"]["birth_country"],
        "self_description": user.get("self_description") or None,
        "system_string": user.get("system_string") 
        #"lm_use_cases": user["lm_use_cases"], # 'lm_usecases': {'homework_assistance': 0.0, 'research': 1.0, 'source_suggestions': 0.0, 'professional_work': 0.0, 'creative_writing': 0.0, 'casual_conversation': 0.0, 'personal_recommendations': 0.0, 'daily_productivity': 0.0, 'technical_or_programming_help': 0.0, 'travel_guidance': 0.0, 'lifestyle_and_hobbies': 1.0, 'well-being_guidance': 1.0, 'medical_guidance': 0.0, 'financial_guidance': 0.0, 'games': 1.0, 'historical_or_news_insight': 0.0, 'relationship_advice': 0.0, 'language_learning': 0.0, 'other': 0.0, 'other_text': None}
        #"order_lm_use_cases": user["order_lm_use_cases"], # 'order_lm_usecases': 'order_lm_usecases': {'homework_assistance': 11.0, 'research': 16.0, 'source_suggestions': 7.0, 'professional_work': 13.0, 'creative_writing': 14.0, 'casual_conversation': 17.0, 'personal_recommendations': 6.0, 'daily_productivity': 12.0, 'technical_or_programming_help': 2.0, 'travel_guidance': 10.0, 'lifestyle_and_hobbies': 9.0, 'well-being_guidance': 18.0, 'medical_guidance': 8.0, 'financial_guidance': 15.0, 'games': 3.0, 'historical_or_news_insight': 1.0, 'relationship_advice': 4.0, 'language_learning': 5.0, 'other': 19.0}
    }
    return str(profile)


def _build_pair_row(completion_a: dict[str, Any], completion_b: dict[str, Any], user) -> dict[str, Any]:
    assert completion_a.get("user_id") == completion_b.get("user_id"), "User IDs do not match"
    return {
        "conversation_id1": completion_a.get("conversation_id"),
        "conversation_id2": completion_b.get("conversation_id"),
        "prompt1": f"{completion_a.get("opening_prompt")}",
        "prompt2": f"{completion_a.get("opening_prompt")}",
        "context": _build_context(completion_b, user),
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
    # print the number of instances per user. Print the instances of the user with the most instances. 
    
    random.seed(42)
    random.shuffle(users)

    user_ids_random = users
    assert len(user_ids_random) == 1371, f"Total user IDs should be 1371. {len(user_ids_random)}"
    user_ids_train = user_ids_random[0:1171]
    train_user_ids = {str(user_id) for user_id in user_ids_train}
    user_ids_val = user_ids_random[1171:1271]
    val_user_ids = {str(user_id) for user_id in user_ids_val}

    user_ids_test = user_ids_random[1271:1371]
    test_user_ids = {str(user_id) for user_id in user_ids_test}

    indexes_train = [
        i
        for i, user_id in enumerate(hf_dataset["user_id"])
        if str(user_id) in train_user_ids
    ]
    indexes_val = [
        i
        for i, user_id in enumerate(hf_dataset["user_id"])
        if str(user_id) in val_user_ids
    ]
    indexes_test = [
            i
            for i, user_id in enumerate(hf_dataset["user_id"])
            if str(user_id) in test_user_ids
        ]

    cases_new_users_val = indexes_val
    cases_new_users_test = indexes_test
    cases_train_users_val = indexes_train[0:len(indexes_val)]
    cases_train_users_test = indexes_train[len(indexes_val):len(indexes_val)+len(indexes_test)]

    validation_indices = cases_new_users_val + cases_train_users_val
    test_indices = cases_new_users_test + cases_train_users_test

    
    
    # Save to PRISM_PROCESSED_PATH folder
    save_processeddataset(PRISM_PROCESSED_PATH, hf_dataset, validation_indices, test_indices)
    return len(hf_dataset), validation_indices, test_indices

if __name__ == "__main__":
    total_rows, validation_indices, test_indices = prism_processor()
    print(f"Total merged rows: {total_rows}")
    print(f"Validation rows: {len(validation_indices)}")
    print(f"Test rows: {len(test_indices)}")