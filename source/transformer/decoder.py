import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ffn import FeedForwardNetwork

class DecoderLayer(nn.Module):
	def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
		super().__init__()
		self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
		self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
		self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

	def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
	            self_mask: torch.Tensor | None = None, 
	            cross_mask: torch.Tensor | None = None) -> torch.Tensor:
	
        #masked self attention
		x, _ = self.self_attn(x, x, x, mask=self_mask)
		
        #cross attention
		x, _ = self.cross_attn(x, encoder_output, encoder_output, mask=cross_mask)
		
        #ffn
		x = self.ffn(x)
		return x


class DecoderStack(nn.Module):
	def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
		super().__init__()
		self.layers = nn.ModuleList([
			DecoderLayer(d_model, num_heads, d_ff, dropout) 
			for _ in range(num_layers)
		])
		self.layer_norm = nn.LayerNorm(d_model)

	def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
	            self_mask: torch.Tensor | None = None,
	            cross_mask: torch.Tensor | None = None) -> torch.Tensor:
		for layer in self.layers:
			x = layer(x, encoder_output, self_mask, cross_mask)
		x = self.layer_norm(x)
		return x
