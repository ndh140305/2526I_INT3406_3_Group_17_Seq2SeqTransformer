import torch
import torch.nn.functional as F

def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    vocab_size = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )
    return loss
