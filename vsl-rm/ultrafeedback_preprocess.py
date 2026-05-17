
import itertools
from pathlib import Path
from typing import Any

# IMport HF_TOKEN from .env
import os
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset, load_from_disk, Dataset

from vsllib.defines import LOCAL_DATASET_PATH, ULTRAFEEDBACK_PROCESSED_PATH

# Load HF tokens.
def _extract_rating(completion: dict[str, Any], value_name: str) -> Any:
	annotations = completion.get("annotations") or {}
	value_block = annotations.get(value_name)

	if isinstance(value_block, list) and value_block:
		first_entry = value_block[0]
		if isinstance(first_entry, dict):
			return first_entry.get("Rating")

	if isinstance(value_block, dict):
		return value_block.get("Rating")

	return None


def _build_context(completion: dict[str, Any]) -> str:
	custom_system_prompt = completion.get("custom_system_prompt") or None
	return custom_system_prompt


def _build_pair_row(sample: dict[str, Any], completion_a: dict[str, Any], completion_b: dict[str, Any]) -> dict[str, Any]:
	principle_a = completion_a.get("principle")
	principle_b = completion_b.get("principle")

	return {
		"source": sample.get("source"),
		"prompt": sample.get("instruction"),
		#"context1": _build_context(completion_a),
		#"labelcontext1": principle_a,
		#"context2": _build_context(completion_b),
		#"labelcontext2": principle_b,
		"response1": completion_a.get("response"),
		"response2": completion_b.get("response"),
		"model1": completion_a.get("model"),
		"model2": completion_b.get("model"),
		"score1": completion_a.get("overall_score"),
		"score2": completion_b.get("overall_score"),
		"value_uhelpfulness_1": _extract_rating(completion_a, "helpfulness"),
		"value_uhelpfulness_2": _extract_rating(completion_b, "helpfulness"),
		"value_uhonesty_1": _extract_rating(completion_a, "honesty"),
		"value_uhonesty_2": _extract_rating(completion_b, "honesty"),
		"value_utruthfulness_1": _extract_rating(completion_a, "truthfulness"),
		"value_utruthfulness_2": _extract_rating(completion_b, "truthfulness"),
		"value_uinstruction_following_1": _extract_rating(completion_a, "instruction_following"),
		"value_uinstruction_following_2": _extract_rating(completion_b, "instruction_following"),
	}


def ultrafeedback_processor(
	output_path: str = "ultrafeedback_pairs",
	dataset_name: str = "openbmb/UltraFeedback",
) -> int:
	dataset: Dataset = load_dataset(dataset_name, split="train")
	
	rows = []
	for sample in dataset:
		if isinstance(sample, dict):
			completions: list[dict[str, str|None]] = sample.get("completions") or []
		else:
			continue
			
		if not isinstance(completions, list) or len(completions) < 2:
			continue

		for completion_a, completion_b in itertools.combinations(completions, 2):
			assert completion_a.get("response") != completion_b.get("response") or completion_a.get("model") != completion_b.get("model") 
			row = _build_pair_row(sample, completion_a, completion_b)
			rows.append(row)
	
	# Create HuggingFace dataset
	hf_dataset = Dataset.from_dict( {k: [row[k] for row in rows] for k in rows[0].keys()})
	
	# Save to LOCAL_PROCESSED_DATASETS folder
	local_datasets_path = Path(ULTRAFEEDBACK_PROCESSED_PATH).joinpath("preprocessed")
	local_datasets_path.mkdir(parents=True, exist_ok=True)
	final_output = local_datasets_path
	hf_dataset.save_to_disk(final_output)
	
	return len(rows)


if __name__ == "__main__":
    ultrafeedback_processor()
    # Show the rows with different principles for the same prompt upto 100 rows:
    """processed_dataset = load_from_disk(os.path.join(LOCAL_DATASET_PATH, "ultrafeedback_pairs"))

    count = 0
    per_principle_count = {}
    per_principle_count_same_principle = {}
    for i, row in enumerate(processed_dataset):
        if "labelcontext1" not in row or "labelcontext2" not in row:
            continue
        if row["labelcontext1"] == row["labelcontext2"] and row["labelcontext1"] is not None and row["labelcontext2"] is not None:
            pair = (row["labelcontext1"], row["labelcontext2"])
            # order pair alphabetically to avoid counting (A, B) and (B, A) separately
            pair = tuple(sorted(pair))

        if pair not in per_principle_count_same_principle:
            per_principle_count_same_principle[pair] = 0

            per_principle_count_same_principle[pair] += 1

        if row["labelcontext1"] != row["labelcontext2"]:
            pair = (row["labelcontext1"], row["labelcontext2"])
            # order pair alphabetically to avoid counting (A, B) and (B, A) separately
            pair = tuple(sorted(pair))

        if pair not in per_principle_count:
            per_principle_count[pair] = 0

            per_principle_count[pair] += 1
            assert row['prompt'] == row['prompt']
            count +=1
        if count < 100:
            print("-----START--------------------")
            print(row['context1'])
            print("-------------------------")
            print(row['context2'])
            print("-----END--------------------")
    print("Total cases where responses have different principles for the same prompt:", count)
    print("Count of different principles across all pairs:", per_principle_count)
    print("Count of same principles across all pairs:", per_principle_count_same_principle)
    print("Total rows processed:", i + 1)"""