import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import FeedForwardNetwork

class EncoderLayer(nn.Module):
	def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
		super().__init__()
		self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
		self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

	def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
		x, _ = self.self_attn(x, x, x, mask=mask)
		#self attn
		x = self.ffn(x)
		#ffn
		return x


class EncoderStack(nn.Module):
	def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
		super().__init__()
		self.layers = nn.ModuleList([
			EncoderLayer(d_model, num_heads, d_ff, dropout) 
			for _ in range(num_layers)
		])
		self.layer_norm = nn.LayerNorm(d_model)

	def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
		for layer in self.layers:
			x = layer(x, mask)
		x = self.layer_norm(x)
		return x
