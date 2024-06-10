import torch
import torch.nn.functional as F
from cs336_alignment.utils import load_model, HOME_DIR
import os

def per_instance_dpo_loss(pi_theta, pi_ref, tokenizer, prompt: str, yw: str, yl: str) -> torch.Tensor:
    # Format the prompt and responses using the Alpaca template
    with open(os.path.join(HOME_DIR, 'cs336_alignment/prompts/alpaca_sft.prompt'), "r") as f:
        template = f.read()
    prompt_yw = template.format(prompt=prompt, response=yw) + tokenizer.eos_token
    prompt_yl = template.format(prompt=prompt, response=yl) + tokenizer.eos_token

    # Tokenize the concatenated prompt and responses
    inputs_yw = tokenizer(prompt_yw, return_tensors='pt', padding=True, truncation=True)
    inputs_yl = tokenizer(prompt_yl, return_tensors='pt', padding=True, truncation=True)

    # Ensure the inputs are on the same device as the models
    device = next(pi_theta.parameters()).device
    inputs_yw = {key: value.to(device) for key, value in inputs_yw.items()}
    inputs_yl = {key: value.to(device) for key, value in inputs_yl.items()}

    # Compute log-probabilities of the concatenated sequences under the models
    with torch.no_grad():
        outputs_yw_ref = pi_ref(**inputs_yw)
        outputs_yl_ref = pi_ref(**inputs_yl)

    outputs_yw_theta = pi_theta(**inputs_yw)
    outputs_yl_theta = pi_theta(**inputs_yl)

    # Calculate the log-probabilities
    log_probs_yw_ref = F.log_softmax(outputs_yw_ref.logits, dim=-1)
    log_probs_yl_ref = F.log_softmax(outputs_yl_ref.logits, dim=-1)

    log_probs_yw_theta = F.log_softmax(outputs_yw_theta.logits, dim=-1)
    log_probs_yl_theta = F.log_softmax(outputs_yl_theta.logits, dim=-1)

    # Sum log-probabilities over the token dimension to get the sequence log-probabilities
    log_prob_yw_ref = log_probs_yw_ref.gather(dim=-1, index=inputs_yw['input_ids'].unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    log_prob_yl_ref = log_probs_yl_ref.gather(dim=-1, index=inputs_yl['input_ids'].unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    log_prob_yw_theta = log_probs_yw_theta.gather(dim=-1, index=inputs_yw['input_ids'].unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    log_prob_yl_theta = log_probs_yl_theta.gather(dim=-1, index=inputs_yl['input_ids'].unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    # Compute the DPO loss
    dpo_loss = -torch.log(torch.sigmoid(log_prob_yw_theta - log_prob_yl_theta - (log_prob_yw_ref - log_prob_yl_ref)))

    return dpo_loss
