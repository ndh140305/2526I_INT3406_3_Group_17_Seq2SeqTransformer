import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
	def __init__(self, dropout: float = 0.0):
		super().__init__()
		self.dropout = nn.Dropout(dropout)

	def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
		d_k = Q.size(-1)
		scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
		if mask is not None:
			scores = scores.masked_fill(mask == 0, float('-inf'))
		attn = F.softmax(scores, dim=-1)
		attn = self.dropout(attn)
		output = torch.matmul(attn, V)
		return output, attn


class MultiHeadAttention(nn.Module):
	def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
		super().__init__()
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = d_model // num_heads

		self.W_q = nn.Linear(d_model, d_model)
		self.W_k = nn.Linear(d_model, d_model)
		self.W_v = nn.Linear(d_model, d_model)
		self.W_o = nn.Linear(d_model, d_model)

		self.attn = ScaledDotProductAttention(dropout)
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(d_model)

	def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
		bsz, seq_len, _ = x.size()
		x = x.view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
		return x

	def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
		bsz, heads, seq_len, d_k = x.size()
		x = x.transpose(1, 2).contiguous().view(bsz, seq_len, heads * d_k)
		return x

	def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, mask: torch.Tensor | None = None):
		residual = x_q

		Q = self.W_q(x_q)
		K = self.W_k(x_k)
		V = self.W_v(x_v)

		Q = self._split_heads(Q)
		K = self._split_heads(K)
		V = self._split_heads(V)

		if mask is not None:
			if mask.dim() == 2:
				mask = mask.unsqueeze(1).unsqueeze(1)

		context, attn = self.attn(Q, K, V, mask)
		context = self._combine_heads(context)
		output = self.W_o(context)
		output = self.dropout(output)
		output = self.layer_norm(output + residual)
		return output, attn