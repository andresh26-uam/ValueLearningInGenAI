from typing import Any, Dict, List, Optional, Union
from datasets.arrow_dataset import Dataset
import torch as th
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer
from datasets import DatasetDict, load_dataset, load_from_disk
from dataclasses import dataclass
from transformers.utils.generic import PaddingStrategy
@dataclass
class RewardDataCollatorWithPadding:
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
                    "attention_mask": feature["attention_mask_1"],
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
			#"score": th.tensor([f.get("score", 0.0) for f in merged_features], dtype=th.float32),
			#"value_ratings": [f.get("value_ratings", {}) for f in merged_features],
            "return_loss": True,
        }
        return batch

def tokenize_sample(sample: dict, tokenizer: Any, delete_other_keys: bool = True, extra_keep_keys: list = None) -> dict:
	keep_keys = ["option1", "option2", "input_ids_1", "attention_mask_1", "input_ids_2", "attention_mask_2", "score1", "score2"]
	if extra_keep_keys:
		keep_keys.extend(extra_keep_keys)
	sample['option1'] = tokenizer.apply_chat_template(
		[{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response1']}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
	sample['option2'] = tokenizer.apply_chat_template(
		[{'role': 'user', 'content': sample['prompt']}, {'role': 'assistant', 'content': sample['response2']}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
	tokenized_pos = tokenizer(sample['option1'], truncation=True)
	tokenized_neg = tokenizer(sample['option2'], truncation=True)
	sample["input_ids_1"] = tokenized_pos["input_ids"]
	sample["attention_mask_1"] = tokenized_pos["attention_mask"]
	sample["input_ids_2"] = tokenized_neg["input_ids"]
	sample["attention_mask_2"] = tokenized_neg["attention_mask"]
	if delete_other_keys:	
		keys_to_delete = [key for key in sample.keys() if key not in keep_keys and not key.startswith("value_")]
		for key in keys_to_delete:
			del sample[key]	
	return sample

class PairwisePreferenceDataset(Dataset):
    
    def __init__(self, path: str, tokenizer, from_disk: bool = True, extra_keep_keys: list = None, retokenize: bool = False, split_seed: int = 42):
        if from_disk:
            self.data = load_from_disk(path).shuffle(seed=split_seed) 
        else:
            self.data = load_dataset(path, split="train").shuffle(seed=split_seed) 
        
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        # Extract value keys from the first data item
        if self.data:
            self.value_keys = [key for key in self.data[0].keys() if key.startswith("value_") and key.endswith("_1")]
            self.value_keys = [key.replace("_1", "") for key in self.value_keys]
        else:
            self.value_keys = []
        
        self.data: DatasetDict = self.data.map(lambda x: tokenize_sample(x, self.tokenizer, delete_other_keys=True, extra_keep_keys=extra_keep_keys), num_proc=16, load_from_cache_file=not retokenize)
        self.data: DatasetDict = self.data.train_test_split(test_size=0.1, seed=split_seed) # pyright: ignore[reportAttributeAccessIssue]
        self.train_dataset, self.test_dataset = self.data['train'], self.data['test']	
        self.train_dataset = self.train_dataset.train_test_split(test_size=0.1, seed=split_seed)
        self.train_dataset, self.eval_dataset = self.train_dataset['train'], self.train_dataset['test']


	
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
		
        value_ratings1 = {}
        value_ratings2 = {}
        for key in self.value_keys:
            value_ratings1[key] = row.get(f"{key}_1", None)
            value_ratings2[key] = row.get(f"{key}_2", None)
		
        return {
			"input_ids_1": row["input_ids_1"],
			"attention_mask_1": row["attention_mask_1"],
			"input_ids_2": row["input_ids_2"],
			"attention_mask_2": row["attention_mask_2"],
			"score1": row.get("score1", None),
			"score2": row.get("score2", None),
			"value_ratings1": value_ratings1,
			"value_ratings2": value_ratings2,
		}





def logits_BT(x: th.Tensor, y: th.Tensor, threshold=50.0) -> th.Tensor:
    # print("DIFF", th.max(x - y))
    assert isinstance(x, th.Tensor) and isinstance(
        y, th.Tensor), f"Expected th.Tensor, got {type(x)} and {type(y)}"
    returns_diff = x - y
    returns_diff = th.clip(returns_diff, -threshold, threshold)
    assert max(returns_diff) <= threshold and min(returns_diff) >= - \
        threshold, f"Clipping failed: max {max(returns_diff)}, min {min(returns_diff)}, threshold {threshold}"
    

    if returns_diff.requires_grad:
        assert returns_diff.grad_fn is not None, "The returned tensor does not require gradients."
    return returns_diff

def grounding_loss(reward1: th.Tensor, reward2: th.Tensor, scores1: th.Tensor=None, scores2: th.Tensor=None, label: th.Tensor=None, reward_diff_threshold: float=50.0):
    """Multi-objective Cross-entropy loss: target_probs(1,2)*log(exp(r1) / (exp(r1) + exp(r2)))- (1-target_probs(1,2))*log(exp(r2) / (exp(r1) + exp(r2)))"""
    # label = 1: reward1 should be higher.
    # label = 0: reward2 should be higher.
    # label = 0.5: no preference.
    logits = logits_BT(reward1, reward2, threshold=reward_diff_threshold)
    with th.no_grad():
        if scores1 is None or scores2 is None:	
            target_probs: th.Tensor = label
        else:
            target_probs = th.sigmoid(logits_BT(scores1, scores2, threshold=reward_diff_threshold))

    assert target_probs.shape == logits.shape, f"Target probabilities shape {target_probs.shape} does not match logits shape {logits.shape}"	
    assert not any(logits.isnan()) and not any(logits.isinf()), f"Logits contain NaN or Inf values: {logits}"
    assert not any(target_probs.isnan()) and not any(target_probs.isinf()), f"Target probabilities contain NaN or Inf values: {target_probs}"	
    assert all(target_probs.detach() >= 0.0) and all(target_probs.detach() <= 1.0), f"Target probabilities should be in [0, 1], but got {target_probs}"

    loss = th.nn.functional.binary_cross_entropy_with_logits(
                # /sum(weights)
                logits, target_probs, reduction='none')
    assert loss.shape == (reward1.shape[0],), f"Expected loss shape {(reward1.shape[0],)}, got {loss.shape}"
    return loss




class MORewardTrainer(Trainer):
    
    def compute_metrics(eval_pred):
        result = {}
        pos_predictions_scores = eval_pred.predictions[0]
        neg_predictions_scores = eval_pred.predictions[1]

        print(eval_pred)
        print(eval_pred.predictions)
        input("Eval??")
        # We assume that the first sample is preferred by default in groundtruth
        result['representativeness'] = np.sum(
            pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
        
        result['representativeness'] = np.sum(
            pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
        
        return result

    # This assumes that the data is collated using RewardDataCollatorWithPadding, and that the model returns multiple rewards for each input.
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        print(rewards.shape)


        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        grounding_loss_value = grounding_loss(
            rewards_j, rewards_k, scores1=inputs.get("score1"), scores2=inputs.get("score2"), label=inputs.get("label")
        )
        #loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss