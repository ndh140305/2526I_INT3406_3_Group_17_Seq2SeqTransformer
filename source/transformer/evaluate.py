import torch
import math
from typing import List, Tuple
from collections import Counter


def greedy_decode(
    model,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    bos_token_id: int,
    eos_token_id: int,
):
    model.eval()
    device = src_ids.device

    with torch.no_grad():
        enc_mask = model.create_padding_mask(src_mask) if hasattr(model, "create_padding_mask") else None
        enc_output = model.encode(src_ids, src_mask)

        batch_size = src_ids.size(0)
        ys = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        tgt_mask = torch.ones_like(ys, device=device)

        for _ in range(max_len - 1):
            logits = model.decode(ys, enc_output, src_mask, tgt_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            tgt_mask = torch.ones_like(ys, device=device)
            if (next_token == eos_token_id).all():
                break

    return ys


def beam_search(
    model,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    bos_token_id: int,
    eos_token_id: int,
    beam_width: int = 5,
) -> torch.Tensor:
    assert src_ids.dim() == 2, f"src_ids must be 2D, got {src_ids.dim()}D"
    assert beam_width > 0, f"beam_width must be > 0, got {beam_width}"
    assert max_len > 0, f"max_len must be > 0, got {max_len}"
    assert bos_token_id >= 0 and eos_token_id >= 0, "Token IDs must be non-negative"
    
    model.eval()
    device = src_ids.device
    batch_size = src_ids.size(0)
    
    with torch.no_grad():
        enc_output = model.encode(src_ids, src_mask)
        
        beams = torch.full((batch_size, beam_width, 1), bos_token_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros((batch_size, beam_width), device=device)
        beam_scores[:, 1:] = -1e9 
        
        for step in range(1, max_len):
            active_beams = beams[:, :, :step].reshape(batch_size * beam_width, step)
            active_masks = torch.ones_like(active_beams, device=device)
            
            logits = model.decode(active_beams, enc_output.repeat_interleave(beam_width, 0), 
                                 src_mask.repeat_interleave(beam_width, 0), active_masks)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            
            log_probs = log_probs.reshape(batch_size, beam_width, -1)
            
            scores = beam_scores.unsqueeze(2) + log_probs
            scores = scores.reshape(batch_size, -1)
         
            vocab_size = log_probs.size(2)
            assert vocab_size > 0, "Vocabulary size must be > 0"
            
            top_scores, top_indices = torch.topk(scores, beam_width, dim=1)
            top_beams = top_indices // vocab_size
            top_tokens = top_indices % vocab_size
            
            new_beams = torch.cat([beams[torch.arange(batch_size).unsqueeze(1), top_beams], 
                                   top_tokens.unsqueeze(2)], dim=2)
            beams = new_beams
            beam_scores = top_scores
            
            if (top_tokens == eos_token_id).all():
                break
        
        return beams[:, 0] 

def compute_bleu(references: List[List[int]], hypotheses: List[List[int]], max_n: int = 4, smoothing: float = 1e-9) -> float:
    def ngram_counts(seq, n):
        return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))

    total_log_prec = 0.0
    total_weights = 0

    for n in range(1, max_n + 1):
        matches = 0
        possible = 0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = ngram_counts(ref, n)
            hyp_counts = ngram_counts(hyp, n)
            possible += max(len(hyp) - n + 1, 0)
            for ng, c in hyp_counts.items():
                matches += min(c, ref_counts.get(ng, 0))
        precision = (matches + smoothing) / (possible + smoothing)
        total_log_prec += math.log(precision)
        total_weights += 1

    geo_mean = math.exp(total_log_prec / total_weights)

    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * geo_mean

def compute_gemini_score(references: List[List[int]], hypotheses: List[List[int]]) -> float:
    def get_unigrams(seq):
        return Counter(seq)
    
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    count = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_unigrams = get_unigrams(ref)
        hyp_unigrams = get_unigrams(hyp)
        
        matched = sum(min(hyp_unigrams[token], ref_unigrams.get(token, 0)) 
                     for token in hyp_unigrams)
        precision = matched / max(len(hyp), 1)
        
        recall = matched / max(len(ref), 1)
        
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        count += 1
    
    if count == 0:
        return 0.0
    
    gemini = (0.5 * (total_f1 / count) + 
              0.3 * (total_precision / count) + 
              0.2 * (total_recall / count))
    
    return gemini