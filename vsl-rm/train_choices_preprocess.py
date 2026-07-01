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
from tqdm import tqdm
from vsllib.defines import APOLLO_PROCESSED_PATH, NO_RATING_MASK, OASST_PROCESSED_PATH, OASSTFL_PROCESSED_PATH, VALUES_APOLLO, VALUES_OASST, VALUES_OASST_ORIG, DOWNLOADED_DATASETS_PATH, DatasetNames, save_processeddataset

load_dotenv()

LOCAL_FILENAME = str(Path(DOWNLOADED_DATASETS_PATH) / "apolloDataset.csv") 


def process_dataset() -> tuple[pd.DataFrame, Path]:
    random.seed(42)
    """Download the ready trees file and load it into a pandas dataframe."""
    full_data = pd.read_csv(LOCAL_FILENAME, header='infer')

    to_norm_features = ["hh_inc_abs", ("tt1", "tt2"), ("hw1", "hw2"), ("ch1", "ch2"), ("tc1", "tc2")]
    for f in to_norm_features:
        if isinstance(f, tuple):
            f1, f2 = f[0], f[1]
            all_data = np.concatenate([full_data[f1].to_numpy(), full_data[f2].to_numpy()])
            for f_ in (f1,f2):
                full_data[f_ + "_NORM"] = ((full_data[f_])-np.mean(all_data))/np.std(all_data)
        else:
            full_data[f + "_NORM"] = ((full_data[f])-full_data[f].mean())/full_data[f].std()
    
    to_scale_features = ["hh_inc_abs", ("tt1", "tt2"), ("hw1", "hw2"), ("ch1", "ch2"), ("tc1", "tc2")]
    for f in to_scale_features:
        if isinstance(f, tuple):
            f1, f2 = f[0], f[1]
            all_data = np.concatenate([full_data[f1].to_numpy(), full_data[f2].to_numpy()])
            for f_ in (f1,f2):
                full_data[f_ + "_SCALED"] = ((full_data[f_]))/max(all_data)
        else:
            full_data[f + "_SCALED"] = full_data[f]/max(full_data[f].to_numpy())

    print(full_data.head(5))

    rows = []
    for i, line in tqdm(full_data.iterrows()):
        rows.append(process_line(line,i))
        
    pdrows = pd.DataFrame(rows)
    pdrows = pdrows.reset_index(drop=True)
     # ID,choice,tt1,tc1,hw1,ch1,tt2,tc2,hw2,ch2,hh_inc_abs,car_availability,commute,shopping,business,leisure

    return pdrows

def process_line(line: dict, i) -> dict:
    # ID,choice,tt1,tc1,hw1,ch1,tt2,tc2,hw2,ch2,hh_inc_abs,car_availability,commute,shopping,business,leisure

    instance = {}
    instance["state"] = i
    instance["user_id"] = int(line["ID"])
    instance["context"] = np.array([line["hh_inc_abs"],line["car_availability"],line["commute"],line["shopping"],line["business"],line["leisure"]], dtype=np.int_) # TODO maybe use the full conversation??
    instance["action1"] = 1
    instance["action2"] = 2

    instance["grounding_1"] = np.array([line["tt1"], line["tc1"], line["hw1"], line["ch1"]], dtype=np.float32)
    instance["grounding_2"] = np.array([line["tt2"], line["tc2"], line["hw2"], line["ch2"]], dtype=np.float32)

    instance["context_features"] = np.array([line["hh_inc_abs_NORM"],line["car_availability"],line["commute"],line["shopping"],line["business"],line["leisure"]], dtype=np.float32)
   # print(instance["context_features"] )
    use_normalized_features = "scale"
    if not use_normalized_features:
        instance["grounding_features_1"] = np.array([line["tt1"], line["tc1"], line["hw1"], line["ch1"]], dtype=np.float32)

        instance["grounding_features_2"] = np.array([line["tt2"], line["tc2"], line["hw2"], line["ch2"]], dtype=np.float32)
    elif use_normalized_features == "norm":
        instance["grounding_features_1"] = np.array([line["tt1_NORM"], line["tc1_NORM"], line["hw1_NORM"], line["ch1_NORM"]], dtype=np.float32)

        instance["grounding_features_2"] = np.array([line["tt2_NORM"], line["tc2_NORM"], line["hw2_NORM"], line["ch2_NORM"]], dtype=np.float32)
    else:
        instance["grounding_features_1"] = np.array([line["tt1_SCALED"], line["tc1_SCALED"], line["hw1_SCALED"], line["ch1_SCALED"]], dtype=np.float32)

        instance["grounding_features_2"] = np.array([line["tt2_SCALED"], line["tc2_SCALED"], line["hw2_SCALED"], line["ch2_SCALED"]], dtype=np.float32)
    better_option_is_1 = int(line["choice"] == 1)
    instance["score1"], instance["score2"] = better_option_is_1, 1 - better_option_is_1
    for value_name in VALUES_APOLLO:
        if "cost" in value_name.lower():
            instance[f"value_{value_name}_1"] = -line["tc1"]
            instance[f"value_{value_name}_2"] = -line["tc2"]
        if "time" in value_name.lower():
            instance[f"value_{value_name}_1"] = -line["tt1"]
            instance[f"value_{value_name}_2"] = -line["tt2"]
        if "comf" in value_name.lower():
            case1better = - \
                line["hw1"] + line["hw2"] > 0 and - \
                line["ch1"] + line["ch2"] >= 0.0
            case1better2 = - \
                line["hw1"] + line["hw2"] >= 0 and - \
                line["ch1"] + line["ch2"] > 0.0
            caseworse1 = - \
                line["hw1"] + line["hw2"] < 0 and - \
                line["ch1"] + line["ch2"] <= 0
            caseworse2 = - \
                line["hw1"] + line["hw2"] <= 0 and - \
                line["ch1"] + line["ch2"] < 0
            instance[f"value_{value_name}_1"] = 1.0 if (case1better or case1better2) else 0.0 if (caseworse1 or caseworse2) else 0.5
            instance[f"value_{value_name}_2"] = 0.0 if (case1better or case1better2) else 1.0 if (caseworse1 or caseworse2) else 0.5
    
        

    #assert -np.sign(reply_a["rank"] - reply_b["rank"]) == np.sign(instance["score1"] - instance["score2"]), f"Rank difference does not match score difference: rank {reply_a['rank']} vs {reply_b['rank']}, score {instance['score1']} vs {instance['score2']}"
    
    
    return instance

def apollo_processor() -> tuple[int, list[int]]:
    """Fetch the OASST2 ready trees export and return its row count."""
    pddataset = process_dataset()
    print(pddataset[pddataset["value_Comfort_1"]==0.5].count()/len(pddataset))


    

    hf_dataset = Dataset.from_pandas(pddataset)

    users = hf_dataset.unique("user_id")
    
    random.seed(42)
    np.random.seed(42)
    random.shuffle(users)

    user_ids_random = np.array(users, dtype=int)
    assert len(user_ids_random) == 388, f"Total user IDs should be 388. {len(user_ids_random)}"
    user_ids_train = set(user_ids_random[0:388-70])
    user_ids_val = set(user_ids_random[388-70:388-35])
    user_ids_test = set(user_ids_random[388-35:388])
    
    cases_val = []
    cases_new_users_val = hf_dataset.filter(lambda x: x["user_id"] in user_ids_val)["state"]
    cases_val.extend(cases_new_users_val)
    assert len(cases_val) == len(cases_new_users_val)

    print(f"Cases new users val: {len(cases_new_users_val)}")
    cases_test = []
    cases_new_users_test = hf_dataset.filter(lambda x: x["user_id"] in user_ids_test)["state"]
    cases_test.extend(cases_new_users_test)
    print(f"Cases new users test: {len(cases_new_users_test)}")
    assert len(cases_test) == len(cases_new_users_test)



    cases_test_train_users = []
    cases_val_train_users = []
    for n in user_ids_train:
        index_ = np.random.choice(pddataset[pddataset["user_id"]==n]["state"], replace=False, size=2)
        cases_test_train_users.append(index_[0])
        cases_val_train_users.append(index_[1])
    print(f"Cases old users val: {len(cases_val_train_users)}")
    print(f"Cases old users test: {len(cases_test_train_users)}")

    cases_test.extend(cases_test_train_users)
    cases_val.extend(cases_val_train_users)

    validation_indices = [int(c) for c in cases_val]
    test_indices = [int(c) for c in cases_test]

    assert len(set(validation_indices)) == len(validation_indices)
    assert len(set(test_indices)) == len(set(test_indices))
    assert set(validation_indices).isdisjoint(set(test_indices))
    print(f"Loaded {len(pddataset)} train/val/test instances from the ready export.")

    save_processeddataset(APOLLO_PROCESSED_PATH, hf_dataset, validation_indices, test_indices)
    
    print(pddataset.iloc[test_indices[-1]])
    return len(pddataset), validation_indices, test_indices


if __name__ == "__main__":
    total_rows, validation_indices, test_indices = apollo_processor()

    print(f"Total merged rows: {total_rows}")
    print(f"Validation rows: {len(validation_indices)}")
    print(f"Test rows: {len(test_indices)}")
    
    

