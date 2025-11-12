import torch
from dpo_loss import compute_log_prob

def compute_batch_log_prob(batch, policy, ref_model):
    policy_chosen_logits = policy(
         input_ids=batch['prompt_chosen_ids'],
         attention_mask=batch['prompt_chosen_mask']
         ).logits
    
    policy_rejected_logits = policy(
         input_ids=batch['prompt_rejected_ids'],
         attention_mask=batch['prompt_rejected_mask']
         ).logits

    # calculate the log_prob
    policy_chosen_log_prob = compute_log_prob(
         logits=policy_chosen_logits,
         labels=batch['prompt_chosen_ids'],
         prompt_length=batch['prompt_length']
         )

    policy_rejected_log_prob = compute_log_prob(
         logits=policy_rejected_logits,
         labels=batch['prompt_rejected_ids'],
         prompt_length=batch['prompt_length']
         )

    with torch.no_grad():
        ref_chosen_logits = ref_model(
             input_ids=batch['prompt_chosen_ids'],
             attention_mask=batch['prompt_chosen_mask']
             ).logits
                
        ref_rejected_logits = ref_model(
             input_ids=batch['prompt_rejected_ids'],
             attention_mask=batch['prompt_rejected_mask']
             ).logits

        # calculate the log_prob
        ref_chosen_log_prob = compute_log_prob(
             logits=ref_chosen_logits,
             labels=batch['prompt_chosen_ids'],
             prompt_length=batch['prompt_length']
             )

        ref_rejected_log_prob = compute_log_prob(
            logits=ref_rejected_logits,
            labels=batch['prompt_rejected_ids'],
            prompt_length=batch['prompt_length']
            )
    
    return policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob


