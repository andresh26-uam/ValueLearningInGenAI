#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
from pyexpat import model
import shutil
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch as th
from vsllib.defines import CONTEXT_EMBEDDING_FEATURE_NAME, CONTEXT_FEATURE_NAME, NO_RATING_MASK
from vsllib.utils import convert_to_tensors

# IMport HF_TOKEN from .env
from dotenv import load_dotenv
load_dotenv()

from copy import deepcopy
from datasets import Column, Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from transformers import AutoModelForCausalLM, AutoTokenizer
from vsllib.training_utils import MORewardDataCollator, MORewardDataCollatorWithPadding



def maybe_strip_bos_token(text: str, bos_token: Optional[str]) -> str:
    if bos_token:
        return text.replace(bos_token, "")
    return text

def tokenize_sample(sample: dict, tokenizer: Any, value_keys: list, delete_other_keys: bool = True, extra_keep_keys: list = None, use_context: bool =True) -> dict:
    keep_keys = ["option1", "option2", "input_ids_1", "attention_mask_1", "input_ids_2", "attention_mask_2", "labels"]
    if extra_keep_keys:
        keep_keys.extend(extra_keep_keys)
    sample['option1'] = maybe_strip_bos_token(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response1']}], tokenize=False, add_generation_prompt=False), tokenizer.bos_token)
    sample['option2'] = maybe_strip_bos_token(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response2']}], tokenize=False, add_generation_prompt=False), tokenizer.bos_token)
    if use_context:
        if sample.get("context", None) is not None:
            ctx = sample['context']
        else:
            ctx = sample['prompt']
        ctemplate = maybe_strip_bos_token(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': ctx}], tokenize=False, add_generation_prompt=False), tokenizer.bos_token)
        tok_context = tokenizer(ctemplate, truncation=True)
        sample['context_input_ids'] = tok_context["input_ids"]
        sample['context_attention_mask'] = tok_context["attention_mask"]
        keep_keys.extend(["context_input_ids", "context_attention_mask", "context"])
    
    tokenized_pos = tokenizer(sample['option1'], truncation=True)
    tokenized_neg = tokenizer(sample['option2'], truncation=True)
    sample["input_ids_1"] = tokenized_pos["input_ids"]
    sample["attention_mask_1"] = tokenized_pos["attention_mask"]
    sample["input_ids_2"] = tokenized_neg["input_ids"]
    sample["attention_mask_2"] = tokenized_neg["attention_mask"]
    value_ratings1 = []
    value_ratings2 = []
    for key in value_keys:
        if sample.get(f"{key}_1", 'N/A') == 'N/A' or sample.get(f"{key}_2", 'N/A') == 'N/A':
            value_ratings1.append(NO_RATING_MASK)
            value_ratings2.append(NO_RATING_MASK)
        else:
            value_ratings1.append(float(sample.get(f"{key}_1", NO_RATING_MASK)))
            value_ratings2.append(float(sample.get(f"{key}_2", NO_RATING_MASK)))
            
    sample["labels"] = th.tensor(np.array([[*value_ratings1, sample.get("score1", NO_RATING_MASK)] , [*value_ratings2, sample.get("score2", NO_RATING_MASK)]]), dtype=th.float16)
    
    if delete_other_keys:	
        keys_to_delete = [key for key in sample.keys() if key not in keep_keys and not key.startswith("value_")]
    for key in keys_to_delete:
        del sample[key]	
    return sample

def embed_sample(sample: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, collator: MORewardDataCollatorWithPadding, use_context: bool =True, device: th.device = th.device("cpu")) -> dict:
    # THIS ASSUMES BATCHED MAPPING FUNCTION.
    with th.no_grad():
        model_device = device
        for ic, case in enumerate([("input_ids_1", "attention_mask_1", "embedding_1"), ("input_ids_2", "attention_mask_2", "embedding_2"), ("context_input_ids", "context_attention_mask", CONTEXT_EMBEDDING_FEATURE_NAME)]):
            if ic == 2 and not use_context:
                continue
            merged_features = {
                "input_ids": sample[case[0]],
                "attention_mask": sample[case[1]],
            }
            
            batch = tokenizer.pad(
                merged_features,
                padding=collator.padding,
                max_length=collator.max_length,
                pad_to_multiple_of=collator.pad_to_multiple_of,
                return_tensors=collator.return_tensors,
            )
            inputs = batch["input_ids"].to(model_device)
            atm = batch["attention_mask"].to(model_device)
            output = model(
                input_ids=inputs,
                attention_mask=atm,
                return_dict=True,
            ).last_hidden_state
            # Pick the last non-padding token embedding for each sequence.
            last_token_idx = atm.sum(dim=1) - 1

            sample[case[2]] = output[np.arange(output.size(0)), last_token_idx].detach().cpu()
            del output
            
        return sample

def check_format(dataset: DatasetDict) -> None:
    required_keys = {"prompt", "response1", "response2", "score1", "score2"}
    for split_name, split_dataset in dataset.items():
        for key in required_keys:
            if key not in split_dataset.column_names:
                raise ValueError(f"Dataset split '{split_name}' is missing required key '{key}'. Found keys: {split_dataset.column_names}")

def save_dataset(dataset: Dataset, path: str) -> None:
    save_path = Path(path) 
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_path = save_path.parent / f".{save_path.name}.tmp-{uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    dataset.save_to_disk(str(tmp_path))
    if save_path.exists():
        shutil.rmtree(save_path)
    tmp_path.replace(save_path)

    return dataset
    


class BasePairwisePreferenceDataset():
    
    postprocessor_method = lambda x, value_keys=[], delete_other_keys=True, extra_keep_keys=[], use_context=True, **pp_kwargs: x
    feature_extractor_method = lambda x, use_context=True, device=th.device("cuda"), **fe_kwargs: x

    def calculate_suggested_epsilon(self) -> float:
        suggested_epsilon = float('inf')
        for i in range(len(self.eval_dataset)):
            label_pair = np.asarray(self.eval_dataset[i]['labels'])
            diff = np.abs(label_pair[0]-label_pair[1])
            # Get the smallest non-zero difference from this pair
            nonzero_diff = diff[diff > 0]
            if len(nonzero_diff) > 0:
                smallest_diff_in_pair = np.min(nonzero_diff)
                suggested_epsilon = min(suggested_epsilon, smallest_diff_in_pair)
        return suggested_epsilon/2.0

    def postprocessor_method_after_save(self):
        pass

    def __init__(self, path: str, context_feature_name: str = CONTEXT_FEATURE_NAME, sub_path: str = "postproc", from_disk: bool = True, extra_keep_keys: list = None, repostprocess: bool = False, recalculate_features: bool = False, use_extracted_features: bool = True, use_context: bool = True, split_seed: int = 42, cleanup_cache_files: bool = True, eval_proportion_or_indices: Union[float, List[int]] = 0.05, test_proportion_or_indices: Union[float, List[int]] = 0.1, pp_kwargs: Dict = {}, fe_kwargs: Dict = {}):
        
        self.data: Dataset 
        self.context_feature_name = context_feature_name
        self._cached_context_embeddings=None
        print(f"Loading dataset from {path} with from_disk={from_disk}")

        preprocessed_dataset_path = os.path.join(path, f"preprocessed")
        os.makedirs(preprocessed_dataset_path, exist_ok=True)
        
        postprocessed_dataset_output_path = None
        postprocessed_dataset_output_path = os.path.join(path, f"{sub_path}")
        os.makedirs(postprocessed_dataset_output_path, exist_ok=True)
        """if recalculate_embeddings :
            shutil.rmtree(embedded_or_tokenized_dataset_output_path, ignore_errors=True)"""

        if from_disk:

            if postprocessed_dataset_output_path is None:
                self.data = load_from_disk(preprocessed_dataset_path)
                print(f"Loaded dataset from {preprocessed_dataset_path}")
            else:
                print(postprocessed_dataset_output_path)
                try:
                    self.data = load_from_disk(postprocessed_dataset_output_path)
                    print(f"Loaded embedded/tokenized dataset from {postprocessed_dataset_output_path}")
                except FileNotFoundError:
                    print(f"Embedded/Tonkenized dataset not found at {postprocessed_dataset_output_path}. Loading (tentatively tokenized) dataset from {path}.")
                    self.data = load_from_disk(preprocessed_dataset_path)
                    print(f"Copying dataset to {postprocessed_dataset_output_path} for processing.")
                    output_path = Path(postprocessed_dataset_output_path)
                    self.data = save_dataset(self.data, output_path)
                    print(f"Saved embedded dataset to {output_path}")
        else:
            self.data = load_dataset(path)
            check_format(self.data) # This might be tricky. Might need code to join the splits, then get the indices.
        
        
        if ((self.data[0].get("embedding_1", None) is None) or repostprocess) and use_extracted_features:
            print("KEYS?", self.data[0].keys())
            print("RECALCULATING EMBEDDINGS WITH MODEL")
            recalculate_features = True

        # Extract value keys from the first data item
        if self.data:
            self.value_keys = [key for key in self.data[0].keys() if key.startswith("value_") and key.endswith("_1")]
            self.value_keys = [key.replace("_1", "") for key in self.value_keys]
        else:
            self.value_keys = []
        
        if self.data[0].get("labels") is None:
            print("Adding labels and tokens to dataset")
            print(pp_kwargs)
            self.data: DatasetDict = self.data.map(lambda x: self.postprocessor_method(x, value_keys=self.value_keys, delete_other_keys=True, extra_keep_keys=extra_keep_keys, use_context=use_context, **pp_kwargs), 
                                                   num_proc=16, 
                                                   load_from_cache_file=not repostprocess)
            
            self.data = save_dataset(self.data, postprocessed_dataset_output_path)
            
        
    
        
        if recalculate_features:
            
            #self.data = self.data.select(range(min(1000, len(self.data))))
            with th.no_grad():
                self.data = self.calculate_features(recalculate_features, use_context, fe_kwargs)
                self.data = save_dataset(self.data, postprocessed_dataset_output_path)
            
        
        self.postprocessor_method_after_save()
        
        if cleanup_cache_files:
            removed_cache_files = self.data.cleanup_cache_files()
            print(f"Removed {removed_cache_files} dataset cache files")
            
        assert self.data[0].get("labels") is not None, "Labels are required in the dataset for training."


        
        if isinstance(test_proportion_or_indices, float):
            self.data = self.data.shuffle(seed=split_seed)
        
            assert isinstance(eval_proportion_or_indices, float), "If test_proportion_or_indices is a float, eval_proportion_or_indices must also be a float."
            self.data: DatasetDict = self.data.train_test_split(test_size=test_proportion_or_indices, seed=split_seed) # pyright: ignore[reportAttributeAccessIssue]
            self.train_dataset, self.test_dataset = self.data['train'], self.data['test']
            self.train_dataset = self.train_dataset.train_test_split(test_size=eval_proportion_or_indices, seed=split_seed)
            self.train_dataset, self.eval_dataset = self.train_dataset['train'], self.train_dataset['test']
	
        else:
            print("Using custom test and eval indices for dataset splitting.")
            #print(f"Test indices: {test_proportion_or_indices}")
            #print(f"Eval indices: {eval_proportion_or_indices}")
            #print("LEN DATA GETTING SPLITS", len(self.data))
            assert isinstance(test_proportion_or_indices, list) and isinstance(eval_proportion_or_indices, list), "If test_proportion_or_indices is not a float, it must be a list of indices. Same for eval_proportion_or_indices."
            assert np.intersect1d(test_proportion_or_indices, eval_proportion_or_indices).size == 0, "Test and eval indices should not overlap."
            self.test_dataset = self.data.select(test_proportion_or_indices)
            self.eval_dataset = self.data.select(eval_proportion_or_indices)
            remaining_indices = [i for i in range(len(self.data)) if (i not in test_proportion_or_indices) and (i not in eval_proportion_or_indices)]
            self.train_dataset = self.data.select(remaining_indices)
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Eval dataset size: {len(self.eval_dataset)}")
            print(f"Test dataset size: {len(self.test_dataset)}")
        #self.train_dataset = self.train_dataset.select(range(min(len(self.train_dataset), 200)))
        #self.test_dataset = self.test_dataset.select(range(min(len(self.train_dataset), 50)))
        #self.eval_dataset = self.eval_dataset.select(range(min(len(self.train_dataset), 50)))
        self.get_all_contexts_embeddings()

    def calculate_features(self, recalculate_features, use_context, fe_kwargs, batch_size=32, num_proc=4):
        
        self.data = self.data.map(
                            lambda x: self.feature_extractor_method(x, use_context=use_context, device=th.device("cpu"), **fe_kwargs),
                            load_from_cache_file=not recalculate_features,
                            batched=True,
                            batch_size=batch_size,
                            num_proc=1
                        )
        return self.data
        

    def get_all_contexts_embeddings(self, recalculate=False) -> List[th.Tensor]:
        
        return np.asarray(self.data[self.context_feature_name])
        
    
    def __len__(self):
        return len(self.data)
    


def postprocess_sample(sample: dict, value_keys: list = None, delete_other_keys: bool = True, extra_keep_keys: list = None, use_context: bool =True) -> dict:
  
    keep_keys = ["option1", "option2", "grounding_features_1", "grounding_features_2", "labels"]
    if extra_keep_keys:
        keep_keys.extend(extra_keep_keys)
    sample["state"] =np.asarray(sample["state"])
    if len(sample["state"].shape) == 0:
        sample["state"] = sample["state"].reshape((1,))
    sample["action1"] =np.asarray(sample["action1"])
    if len(sample["action1"].shape) == 0:
        sample["action1"] = sample["action1"].reshape((1,))
    sample["action2"] =np.asarray(sample["action2"])
    if len(sample["action2"].shape) == 0:
        sample["action2"] = sample["action2"].reshape((1,))


    sample['option1'] = np.concatenate([sample["state"],sample["action1"]], axis=-1)
    sample['option2'] = np.concatenate([sample["state"],sample["action2"]], axis=-1)
    if use_context:
        if sample.get("context", None) is not None:
            ctx = sample['context']
        else:
            ctx = sample['state']
        
        sample['context'] = ctx
        if sample.get(CONTEXT_FEATURE_NAME, None) is None:
            sample[CONTEXT_FEATURE_NAME] = ctx
            
        
        keep_keys.extend([CONTEXT_FEATURE_NAME, "context"])
    
    sample["grounding_features_1"] = sample.get("grounding_features_1", sample["option1"])
    sample["grounding_features_2"] = sample.get("grounding_features_2", sample["option2"])
    
    value_ratings1 = []
    value_ratings2 = []
    for key in value_keys:
        if sample.get(f"{key}_1", 'N/A') == 'N/A' or sample.get(f"{key}_2", 'N/A') == 'N/A':
            value_ratings1.append(NO_RATING_MASK)
            value_ratings2.append(NO_RATING_MASK)
        else:
            value_ratings1.append(float(sample.get(f"{key}_1", NO_RATING_MASK)))
            value_ratings2.append(float(sample.get(f"{key}_2", NO_RATING_MASK)))
            
    sample["labels"] = th.tensor(np.array([[*value_ratings1, sample.get("score1", NO_RATING_MASK)] , [*value_ratings2, sample.get("score2", NO_RATING_MASK)]]), dtype=th.float16)
    
    if delete_other_keys:	
        keys_to_delete = [key for key in sample.keys() if key not in keep_keys and not key.startswith("value_")]
    for key in keys_to_delete:
        del sample[key]	
    return sample


def feature_extract_sample(sample: dict, collator: MORewardDataCollator, use_context: bool =True, device: th.device = th.device("cpu")) -> dict:
    # THIS ASSUMES BATCHED MAPPING FUNCTION.
    
    with th.no_grad():
        model_device = device
        for ic, case in enumerate([("option1", "grounding_features_1"), ("option2", "grounding_features_2"), ("context", CONTEXT_FEATURE_NAME)]):
            if ic == 2 and not use_context:
                continue
            merged_features = {
                case[0]: sample[case[0]],
                case[1]: sample[case[1]],
            }
            
            batch = convert_to_tensors(
                merged_features,
                tensor_type=collator.return_tensors,
            )
            sample[case[0]] = batch[case[0]].to(device =model_device)
            sample[case[1]] = batch[case[1]].to(device =model_device)
        
        return sample
class FeatureBasedPreferenceDataset(BasePairwisePreferenceDataset):


    def __init__(self, path: str, from_disk: bool = True, normalize_context: bool = False, extra_keep_keys: List = None, repostprocess: bool = False, recalculate_features: bool = False, collator: MORewardDataCollator = None, use_extracted_features: bool = True, use_context: bool = True, split_seed: int = 42, cleanup_cache_files: bool = True, eval_proportion_or_indices: float | List[int] = 0.05, test_proportion_or_indices: float | List[int] = 0.1, pp_kwargs: Dict = {}, fe_kwargs: Dict = {}):
        fe_kwargs.update({"collator": collator})
        sub_path = "postproc"
        self.postprocessor_method = postprocess_sample
        self.feature_extractor_method = feature_extract_sample
        self.normalize_context = normalize_context
        super().__init__(context_feature_name = CONTEXT_FEATURE_NAME, path=path, sub_path=sub_path, from_disk=from_disk, extra_keep_keys=extra_keep_keys, 
                         repostprocess=repostprocess, recalculate_features=recalculate_features, 
                         use_extracted_features=use_extracted_features, use_context=use_context, 
                         split_seed=split_seed, cleanup_cache_files=cleanup_cache_files, 
                         eval_proportion_or_indices= eval_proportion_or_indices, 
                         test_proportion_or_indices=test_proportion_or_indices, pp_kwargs=pp_kwargs, fe_kwargs=fe_kwargs)
        
        
    def postprocessor_method_after_save(self):
        if self.normalize_context:
            print("NORMALIZING")
            normalizer = StandardScaler()
            ctx_features = self.get_all_contexts_embeddings()
            ctx_features = normalizer.fit_transform(ctx_features)


    

class PairwisePreferenceDataset(BasePairwisePreferenceDataset):
    

    def calculate_features(self, recalculate_features: bool, use_context: bool, fe_kwargs: dict, batch_size=32, num_proc=4):
        model_reference = fe_kwargs.pop("model_reference")
        model_reference = model_reference.cpu()
        assert self.data[0].get("input_ids_1", None) is not None, "Input IDs missing after tokenization step."
        assert self.data[0].get("labels", None) is not None, "Labels   are missing after tokenization step."
        def _embed_shard(dataset_shard, local_model, device):
            with th.no_grad():
                return dataset_shard.map(
                            lambda x: self.feature_extractor_method(x, local_model, use_context=use_context, device=device, **fe_kwargs),
                            load_from_cache_file=not recalculate_features,
                            batched=True,
                            batch_size=batch_size,
                        )

        if th.cuda.is_available() and th.cuda.device_count() > 1:
            n_gpus = th.cuda.device_count()
            print(f"Embedding map sharded across {n_gpus} GPUs")

            shards = [
                        self.data.shard(num_shards=n_gpus, index=i, contiguous=True)
                        for i in range(n_gpus)
                    ]

            model_copies = [deepcopy(model_reference).to(th.device(f"cuda:{i}")).eval() for i in range(n_gpus)]

            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = [
                            executor.submit(_embed_shard, shard, model_copies[i],th.device(f"cuda:{i}"))
                            for i, shard in enumerate(shards)
                        ]
                mapped_shards = [f.result() for f in futures]
            for m in model_copies:
                del m
            self.data = concatenate_datasets(mapped_shards)
        elif th.cuda.is_available():
            print("Embedding dataset on single GPU")
            model_reference = model_reference.to(th.device("cuda"))
            model_reference.eval()
            self.data = self.data.map(
                            lambda x: self.feature_extractor_method(x, model_reference, use_context=use_context, device=th.device("cuda"), **fe_kwargs),
                            load_from_cache_file=not recalculate_features,
                            batched=True,
                            batch_size=batch_size,
                        )

        else:
                print("WARNING: No GPU available, embedding dataset on CPU. This may be very slow.")
                model_reference = model_reference.to(th.device("cpu"))
                model_reference.eval()
                self.data = self.data.map(
                            lambda x: self.feature_extractor_method(x, model_reference, use_context=use_context, device=th.device("cpu"), **fe_kwargs),
                            load_from_cache_file=not recalculate_features,
                            batched=True,
                            batch_size=batch_size,
                            num_proc=num_proc
                        )
        return self.data

    def postprocessor_method_after_save(self) -> None:
        #self.normalize_context = False
        if self.normalize_context:
            print("NORMALIZING")
            def _normalize_context_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
                contexts = np.asarray(batch[self.context_feature_name], dtype=np.float32)
                norms = np.linalg.norm(contexts, ord=2, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise ValueError("Normalization failed: found zero-norm context embeddings.")
                batch[self.context_feature_name] = contexts / norms
                return batch

            self.data = self.data.map(
                _normalize_context_batch,
                batched=True,
                load_from_cache_file=False,
            )

            normalized_contexts = np.asarray(self.data[self.context_feature_name], dtype=np.float32)
            normalized_norms = np.linalg.norm(normalized_contexts, ord=2, axis=1)
            assert np.allclose(normalized_norms, np.ones_like(normalized_norms)), "Normalization failed: not all context embeddings have unit norm."
        
    def __init__(self, path: str, tokenizer, normalize_context=False, from_disk: bool = True, extra_keep_keys: list = None, retokenize: bool = False, recalculate_embeddings: bool = False, use_embeddings: bool = True, model_reference: AutoModelForCausalLM = None, collator: MORewardDataCollatorWithPadding = None, use_context: bool = True, split_seed: int = 42, cleanup_cache_files: bool = True, eval_proportion_or_indices: Union[float, List[int]] = 0.05, test_proportion_or_indices: Union[float, List[int]] = 0.1):
        self.normalize_context = normalize_context
        pp_kwargs = {
            "tokenizer": tokenizer,
        }
        fe_kwargs = {
            "tokenizer": tokenizer,
            "model_reference": model_reference,
            "collator": collator
        }
        self.postprocessor_method = tokenize_sample
        self.feature_extractor_method = embed_sample

        
        super().__init__(path=path,
                         context_feature_name = CONTEXT_EMBEDDING_FEATURE_NAME,
                         from_disk=from_disk,
                         sub_path=model_reference.config._name_or_path.replace('/', '_') if model_reference is not None else "only_tokenized",
                         extra_keep_keys=extra_keep_keys,
                         recalculate_features=recalculate_embeddings,
                         repostprocess=retokenize,
                         use_context=use_context,
                         split_seed=split_seed,
                         use_extracted_features=use_embeddings,
                         cleanup_cache_files=cleanup_cache_files,
                         eval_proportion_or_indices=eval_proportion_or_indices,
                         test_proportion_or_indices=test_proportion_or_indices,
                         pp_kwargs=pp_kwargs,
                         fe_kwargs=fe_kwargs
                         
                        )

        
    
	           
		