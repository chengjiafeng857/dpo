
import torch
import torch.nn.functional as F

# calculate the log probability
def compute_log_prob(logits, labels, average_log_prob: bool = False):
    """ Compute the log probability for each token:

    logits: the output without softmax computation, shape: (batch_size, seq_length, vocab_size)

    labels: labels` is the "supervised version" of `input_ids to help the model 
    The purpose of labels is to help the model correctly align the responses.
    input_ids: [prompt, response], labels only care about response, so the prompt part is mask [mask, response]
    labels.shape: (batch_size, seq_length)
    """
    # do the shift
    # remove the last one of logits and the first one of labels, 
    # let the first one in the logits corresponds to the secode one in the labels
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()

    # filter the label
    # In huggingface and pytorch, label = -100 means this token will not do the loss calculation
    # so we need to filter out token with -100 in labels
    # loss_mask: -100=False, others=Ture, shape:(batch_size, seq_length)
    loss_mask = (labels != -100)
    # the gather() does not allow -100, so we change the -100 to 0
    labels[labels == -100] = 0

    # dim=2, vocab_size
    # index=labels.unsqueeze(2): (batch_size, seq_length) -> (batch_size, seq_length, 1) : gather needs same dim
    # (batch_size, seq_length, 1) -> (batch_size, seq_length)
    per_token_log_prob = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_log_prob * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_log_prob * loss_mask).sum(-1)
    
# calculate the dpo loss
# we will do the loss.mean() in the training step
def dpo_loss(policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob, beta):
    chosen_log_prob = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_log_prob = policy_rejected_log_prob - ref_rejected_log_prob

    # record the M that we need to calculate
    # we can detach it in training step
    policy_diff = policy_chosen_log_prob - policy_rejected_log_prob
    ref_diff =  ref_chosen_log_prob - ref_rejected_log_prob
    model_margin = policy_diff - ref_diff
    
    # compute the loss
    loss = - F.logsigmoid(beta * (chosen_log_prob - rejected_log_prob))

    return loss, chosen_log_prob.mean(), rejected_log_prob.mean(), model_margin

