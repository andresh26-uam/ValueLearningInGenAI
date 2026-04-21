
import numpy as np
import torch as th


import traceback
import sys
def check_inf_vs_missing_mask(tensor: th.Tensor, missing_mask: th.Tensor = None, name: str = "tensor", assume_torch: bool = True) -> None:
    """
    Debug function to count float -inf values and compare with missing mask.
    Prints indices and traceback without raising exceptions.
    
    Args:
        tensor: Tensor to check for -inf values
        missing_mask: Optional mask indicating missing values
        name: Name of the tensor for logging purposes
    """
    if tensor is None or missing_mask is None:
        return
    
    # Count -inf values
    if assume_torch:
        neg_inf_mask = th.isinf(tensor) & (tensor < 0)
    else:
        neg_inf_mask = np.isinf(tensor) & (tensor < 0)
    
    if assume_torch:
        neg_inf_count = neg_inf_mask.sum().item()
        
        # Count missing mask true values
        missing_count = missing_mask.sum().item()
    else:
        neg_inf_count = np.sum(neg_inf_mask)
        missing_count = np.sum(missing_mask)
    
    if neg_inf_count > 0:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"DEBUG: Found {neg_inf_count} float -inf values in {name}", file=sys.stderr)
        print(f"DEBUG: Missing mask has {missing_count} True values", file=sys.stderr)
        
        if neg_inf_count == missing_count:
            raising=False
            print(f"✓ Counts match perfectly!", file=sys.stderr)
        else:
            raising=True
            print(f"✗ Counts DO NOT match! (Difference: {abs(neg_inf_count - missing_count)})", file=sys.stderr)
        
        # Get indices of -inf values
        if assume_torch:
            neg_inf_indices = th.where(neg_inf_mask)
        else:
            neg_inf_indices = np.where(neg_inf_mask)
        print(f"\nIndices of -inf values:", file=sys.stderr)
        for i, idx_tuple in enumerate(zip(*neg_inf_indices)):
            if i < 10:  # Limit output to first 10 for readability
                print(f"  Index {i}: {idx_tuple}", file=sys.stderr)
            elif i == 10:
                print(f"  ... and {neg_inf_count - 10} more", file=sys.stderr)
                break
        
        # Print traceback
        print(f"\nTraceback:", file=sys.stderr)
        for line in traceback.format_stack()[:-1]:
            print(line.rstrip(), file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        if raising:
            raise ValueError(f"Found {neg_inf_count} float -inf values in {name}, which does not match missing mask count of {missing_count}. See debug output for details.")


def __slowed_debug_forward(self, *args, **kwargs):
        #self: MORMForSequenceClassification
        # use this instead of the normal forward to debug the reward computation step by step, checking that the reward extracted from the embeddings is consistent with the full model execution.
        if 'embeddings' in kwargs:
            # If embeddings are provided, bypass the base model and directly compute rewards from embeddings.
            embeddings = kwargs.pop('embeddings')
            # embeding tuple=?????
            all_rewards = self.score(embeddings)
            grounding_ideal = self.reward_heads_ideal(embeddings)
            ar = SequenceClassifierOutputWithPastAndIdeal(logits=all_rewards, ideal_logits=grounding_ideal)
        
        all_rewards_2 = GenericForSequenceClassification.forward(self, *args, **kwargs)
        all_rewards_4 = GenericForSequenceClassification.forward(self, *args, **kwargs)
        th.testing.assert_close(all_rewards_2.logits, all_rewards_4.logits, atol=1e-4, rtol=1e-4)

        lhs = self.full_model(*args, **kwargs).last_hidden_state
        input_ids = kwargs.get("input_ids", None)
        non_pad_mask = (input_ids != self.config.pad_token_id).to(lhs.device, th.int32)
        token_indices = th.arange(input_ids.shape[-1], device=lhs.device, dtype=th.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        embedding = lhs[th.arange(input_ids.size(0), device=lhs.device), last_non_pad_token]

        scores_all = self.score(lhs)
        selected_scores = scores_all[th.arange(input_ids.size(0), device=lhs.device), last_non_pad_token]
        score_embed = self.score(embedding)
        print("last_non_pad_token", last_non_pad_token)
        print("SCORES ALL SHAPE", scores_all.shape)
        print("SCORES Selected SHAPE", selected_scores.shape)
        print("SCORES EMBED SHAPE", score_embed.shape)
        print("SCORE 1", selected_scores[0])
        print("SCORE EMBED 1", score_embed[0])
        print("SCORE 2", selected_scores[1])
        print("SCORE EMBED 2", score_embed[1])
        
        all_rewards_5 = SequenceClassifierOutputWithPast(logits=selected_scores)
        all_rewards_3 = SequenceClassifierOutputWithPast(logits=score_embed)
        th.testing.assert_close(all_rewards_5.logits, all_rewards_3.logits, atol=1e-1, rtol=1e-1)

        print("SHAPE", all_rewards_2.logits.shape)
        assert all_rewards_2.logits.shape == all_rewards.shape, f"Expected logits shape {all_rewards_2.logits.shape} to match pooled_logits shape {all_rewards_2.pooled_logits.shape}"
        th.testing.assert_close(all_rewards_3.logits, ar.logits, atol=1e-4, rtol=1e-4)
        th.testing.assert_close(all_rewards_3.logits, all_rewards_2.logits, atol=1e-1, rtol=1e-1)
        #th.testing.assert_close(ar.logits, all_rewards_2.logits, atol=1e-4, rtol=1e-4)
        return all_rewards_2