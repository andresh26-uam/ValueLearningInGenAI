#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from typing import Any, List, Optional, Union
from uuid import uuid4

import numpy as np
import torch as th
from vsllib.defines import NO_RATING_MASK

# IMport HF_TOKEN from .env
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import AutoModelForCausalLM, AutoTokenizer
from vsllib.utils import MORewardDataCollatorWithPadding

def tokenize_sample(sample: dict, tokenizer: Any, value_keys: list, delete_other_keys: bool = True, extra_keep_keys: list = None, use_context: bool =True) -> dict:
    keep_keys = ["option1", "option2", "input_ids_1", "attention_mask_1", "input_ids_2", "attention_mask_2", "labels"]
    if extra_keep_keys:
        keep_keys.extend(extra_keep_keys)
    sample['option1'] = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response1']}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    sample['option2'] = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response2']}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    if use_context:
        if sample.get("context", None) is not None:
            ctx = sample['context']
        else:
            ctx = sample['prompt']
        ctemplate = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': ctx}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
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

def embed_sample(sample: dict, model: BaseModelOutputWithPast, tokenizer: AutoTokenizer, collator: MORewardDataCollatorWithPadding, use_context: bool =True) -> dict:
    # THIS ASSUMES BATCHED MAPPING FUNCTION.
    model_device = next(model.parameters()).device
    for ic, case in enumerate([("input_ids_1", "attention_mask_1", "embedding_1"), ("input_ids_2", "attention_mask_2", "embedding_2"), ("context_input_ids", "context_attention_mask", "context_embedding")]):
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

class PairwisePreferenceDataset():
    
    def __init__(self, path: str, tokenizer, from_disk: bool = True, extra_keep_keys: list = None, retokenize: bool = False, recalculate_embeddings: bool = False, use_embeddings: bool = True, model_for_embeddings: AutoModelForCausalLM = None, collator: MORewardDataCollatorWithPadding = None, use_context: bool = True, split_seed: int = 42, embedded_dataset_output_path: Optional[str] = None, cleanup_cache_files: bool = True, eval_proportion_or_indices: Union[float, List[int]] = 0.05, test_proportion_or_indices: Union[float, List[int]] = 0.1):
        should_rewrite_embedded_dataset = bool(retokenize or recalculate_embeddings)
        self.data: Dataset 

        if model_for_embeddings is not None and embedded_dataset_output_path is None:
            embedded_dataset_output_path = f"{path.rstrip('/')}_embed_{model_for_embeddings.config._name_or_path.replace('/', '_')}"

        if recalculate_embeddings:
            shutil.rmtree(embedded_dataset_output_path, ignore_errors=True)

        if from_disk:

            if model_for_embeddings is not None and not should_rewrite_embedded_dataset:
                try:
                    self.data = load_from_disk(embedded_dataset_output_path)
                except FileNotFoundError:
                    print(f"Embedded dataset not found at {embedded_dataset_output_path}. Loading (tentatively tokenized) dataset from {path}.")
                    self.data = load_from_disk(path)
            else:
                
                self.data: Dataset = load_from_disk(path)
        else:
            self.data = load_dataset(path, split="train")
        
        self.data = self.data.shuffle(seed=split_seed)
        
        
        if (self.data[0].get("embedding_1", None) is None or retokenize) and use_embeddings:
            recalculate_embeddings = True

        self.max_length = tokenizer.model_max_length
        # Extract value keys from the first data item
        if self.data:
            self.value_keys = [key for key in self.data[0].keys() if key.startswith("value_") and key.endswith("_1")]
            self.value_keys = [key.replace("_1", "") for key in self.value_keys]
        else:
            self.value_keys = []
        
        if self.data[0].get("labels") is None:
            self.data: DatasetDict = self.data.map(lambda x: tokenize_sample(x, tokenizer, value_keys=self.value_keys, delete_other_keys=True, extra_keep_keys=extra_keep_keys, use_context=use_context), num_proc=16, load_from_cache_file=not retokenize)
        
        
        if model_for_embeddings is not None and recalculate_embeddings:
            batch_size = 4
            #self.data = self.data.select(range(min(1000, len(self.data))))
            with th.no_grad():
                def _embed_shard(dataset_shard, device):
                    local_model = deepcopy(model_for_embeddings).to(device)
                    local_model.eval()
                    return dataset_shard.map(
                        lambda x: embed_sample(x, local_model, tokenizer, collator, use_context=use_context),
                        load_from_cache_file=False,
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

                    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                        futures = [
                            executor.submit(_embed_shard, shard, th.device(f"cuda:{i}"))
                            for i, shard in enumerate(shards)
                        ]
                        mapped_shards = [f.result() for f in futures]

                    self.data = concatenate_datasets(mapped_shards)
                else:
                        model_for_embeddings = model_for_embeddings.to(th.device("cpu"))
                        model_for_embeddings.eval()
                        self.data = self.data.map(
                            lambda x: embed_sample(x, model_for_embeddings, tokenizer, collator, use_context=use_context),
                            load_from_cache_file=False,
                            batched=True,
                            batch_size=batch_size,
                            num_proc=4
                        )
        print("AFTER EMBED?", len(self.data))
        
        if model_for_embeddings is not None and should_rewrite_embedded_dataset:
            output_path = Path(embedded_dataset_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_output_path = output_path.parent / f".{output_path.name}.tmp-{uuid4().hex}"
            if temp_output_path.exists():
                shutil.rmtree(temp_output_path)
            self.data.save_to_disk(str(temp_output_path))
            if output_path.exists():
                shutil.rmtree(output_path)
            temp_output_path.replace(output_path)
            print(f"Saved embedded dataset to {output_path}")
        print("AFTER SAVE", len(self.data))
        
        if cleanup_cache_files:
            removed_cache_files = self.data.cleanup_cache_files()
            print(f"Removed {removed_cache_files} dataset cache files")
        print("AFTER REMOVE CACHE", len(self.data))
        
        assert self.data[0].get("labels") is not None, "Labels are required in the dataset for training."

        if isinstance(test_proportion_or_indices, float):
            assert isinstance(eval_proportion_or_indices, float), "If test_proportion_or_indices is a float, eval_proportion_or_indices must also be a float."
            self.data: DatasetDict = self.data.train_test_split(test_size=test_proportion_or_indices, seed=split_seed) # pyright: ignore[reportAttributeAccessIssue]
            self.train_dataset, self.test_dataset = self.data['train'], self.data['test']
            self.train_dataset = self.train_dataset.train_test_split(test_size=eval_proportion_or_indices, seed=split_seed)
            self.train_dataset, self.eval_dataset = self.train_dataset['train'], self.train_dataset['test']	
        else:
            print("Using custom test and eval indices for dataset splitting.")
            print(f"Test indices: {test_proportion_or_indices}")
            print(f"Eval indices: {eval_proportion_or_indices}")
            print("LEN DATA GETTING SPLITS", len(self.data))
            assert isinstance(test_proportion_or_indices, list) and isinstance(eval_proportion_or_indices, list), "If test_proportion_or_indices is not a float, it must be a list of indices. Same for eval_proportion_or_indices."
            assert np.intersect1d(test_proportion_or_indices, eval_proportion_or_indices).size == 0, "Test and eval indices should not overlap."
            self.test_dataset = self.data.select(test_proportion_or_indices)
            self.eval_dataset = self.data.select(eval_proportion_or_indices)
            remaining_indices = [i for i in range(len(self.data)) if (i not in test_proportion_or_indices) and (i not in eval_proportion_or_indices)]
            self.train_dataset = self.data.select(remaining_indices)
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Eval dataset size: {len(self.eval_dataset)}")
            print(f"Test dataset size: {len(self.test_dataset)}")

    def __len__(self):
        return len(self.data)
    
			           
		