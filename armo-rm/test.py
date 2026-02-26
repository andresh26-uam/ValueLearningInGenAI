# Cloned at version 26 Feb 2026. https://github.com/RLHFlow/RLHF-Reward-Modeling/commit/4cc54482fdbbc69534bf2307de0cc5abb4a06bed
import torch, time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
device = "cpu"
path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device, 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
# We load a random sample from the validation set of the HelpSteer dataset

start_time = time.time()
prompt = 'What are some synonyms for the word "beautiful"?'
response = "Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant"
messages = [{"role": "user", "content": prompt},
           {"role": "assistant", "content": response}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
print(input_ids.shape)
print(model)
with torch.no_grad():
   output = model(input_ids)
   # Multi-objective rewards for the response
   multi_obj_rewards = output.rewards.cpu().float() 
   # The gating layer's output is conditioned on the prompt
   gating_output = output.gating_output.cpu().float()
   # The preference score for the response, aggregated from the 
   # multi-objective rewards with the gating layer
   preference_score = output.score.cpu().float()  
# We apply a transformation matrix to the multi-objective rewards
# before multiplying with the gating layer's output. This mainly aims
# at reducing the verbosity bias of the original reward objectives
obj_transform = model.reward_transform_matrix.data.cpu().float()
# The final coefficients assigned to each reward objective
multi_obj_coeffs = gating_output @ obj_transform.T
# The preference score is the linear combination of the multi-objective rewards with
# the multi-objective coefficients, which can be verified by the following assertion
assert torch.isclose(torch.sum(multi_obj_rewards * multi_obj_coeffs, dim=1), preference_score, atol=1e-3) 
# Find the top-K reward objectives with coefficients of the highest magnitude
K = 3
top_obj_dims = torch.argsort(torch.abs(multi_obj_coeffs), dim=1, descending=True,)[:, :K]
top_obj_coeffs = torch.gather(multi_obj_coeffs, dim=1, index=top_obj_dims)

# The attributes of the 19 reward objectives
attributes = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
   'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
   'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
   'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
   'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
   'code-style','code-explanation','code-instruction-following','code-readability']

example_index = 0
for i in range(K):
   attribute = attributes[top_obj_dims[example_index, i].item()]
   coeff = top_obj_coeffs[example_index, i].item()
   print(f"{attribute}: {round(coeff,5)}")
# code-complexity: 0.19922
# helpsteer-verbosity: -0.10864
# ultrafeedback-instruction_following: 0.07861

# The actual rewards of this example from the HelpSteer dataset
# are [3,3,4,2,2] for the five helpsteer objectives: 
# helpfulness, correctness, coherence, complexity, verbosity
# We can linearly transform our predicted rewards to the 
# original reward space to compare with the ground truth
helpsteer_rewards_pred = multi_obj_rewards[0, :5] * 5 - 0.5
print(helpsteer_rewards_pred)
end_time = time.time()
print(f"Inference time: {round(end_time - start_time, 2)} seconds")