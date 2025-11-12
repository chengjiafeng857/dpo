import torch
import torch.nn.functional as F

# calculate the log probability
def compute_log_prob(logits, labels, prompt_length):
    # do the shift
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()

    # compute the log_prob
    log_prob = F.log_softmax(logits, dim=-1)

    # Gather the real label tokens log_prob
    token_log_prob = torch.gather(input=log_prob, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # define masks for only calculating the response log_prob (assume the model no pad (GPT/Llama: pad=eos))
    seq_len = labels.shape[1] 
    pos_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) + 1
    mask = (pos_mask >= prompt_length.unsqueeze(1)) 
    mask_f = mask.float()

    # the response log_prob (average)
    resp_log_prob = (token_log_prob * mask_f).sum(dim=-1)
    resp_lengths = mask_f.sum(dim=-1).clamp(min=1)
    avg_log_prob = resp_log_prob / resp_lengths

    return avg_log_prob
    

# calculate the dpo loss
def dpo_loss(policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob, beta):
    chosen_log_prob = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_log_prob = policy_rejected_log_prob - ref_rejected_log_prob
    
    # compute the loss
    loss = - F.logsigmoid(beta * (chosen_log_prob - rejected_log_prob)).mean()

    # monitor the training results
    reward_accuracy = (chosen_log_prob > rejected_log_prob).float().mean()
    reward_margins = (chosen_log_prob - rejected_log_prob).mean()

    return loss, chosen_log_prob.mean(), rejected_log_prob.mean(), reward_accuracy, reward_margins

