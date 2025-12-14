import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .embedding import InputEmbeddings, PositionalEncoding
from .encoder import EncoderStack
from .decoder import DecoderStack

def create_padding_mask(mask: torch.Tensor) -> torch.Tensor:
	if mask.dim() != 2:
		raise ValueError("Padding mask must have shape (batch, seq_len)")
	return mask.unsqueeze(1).unsqueeze(1)


def create_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
	mask = torch.tril(torch.ones(size, size, device=device))
	return mask.unsqueeze(0).unsqueeze(0)

class Transformer(nn.Module):
	def __init__(
		self,
		src_vocab_size: int,
		tgt_vocab_size: int,
		d_model: int = 512,
		num_heads: int = 8,
		num_encoder_layers: int = 6,
		num_decoder_layers: int = 6,
		d_ff: int = 2048,
		dropout: float = 0.1,
		max_seq_len: int = 512,
		pad_token_id: int = 0,
	):
		super().__init__()

		self.src_embed = InputEmbeddings(d_model, src_vocab_size)
		self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
		self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

		self.encoder = EncoderStack(d_model, num_heads, d_ff, num_encoder_layers, dropout)
		self.decoder = DecoderStack(d_model, num_heads, d_ff, num_decoder_layers, dropout)

		self.output_proj = nn.Linear(d_model, tgt_vocab_size)
		self.pad_token_id = pad_token_id

	@staticmethod
	def create_padding_mask(mask: torch.Tensor) -> torch.Tensor:
		return create_padding_mask(mask)

	@staticmethod
	def create_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
		return create_causal_mask(size, device=device)

	def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
		src = self.pos_encoding(self.src_embed(src_ids))
		enc_mask = create_padding_mask(src_mask) if src_mask is not None else None
		return self.encoder(src, enc_mask)

	def decode(
		self,
		tgt_ids: torch.Tensor,
		enc_output: torch.Tensor,
		src_mask: torch.Tensor | None = None,
		tgt_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		tgt = self.pos_encoding(self.tgt_embed(tgt_ids))
		enc_mask = create_padding_mask(src_mask) if src_mask is not None else None
		dec_pad_mask = create_padding_mask(tgt_mask) if tgt_mask is not None else None
		causal = create_causal_mask(tgt.size(1), device=tgt.device)
		self_mask = torch.minimum(dec_pad_mask, causal) if dec_pad_mask is not None else causal
		dec_output = self.decoder(tgt, enc_output, self_mask=self_mask, cross_mask=enc_mask)
		logits = self.output_proj(dec_output)
		return logits

	def forward(
		self,
		src_ids: torch.Tensor,
		tgt_ids: torch.Tensor,
		src_mask: torch.Tensor | None = None,
		tgt_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		if src_ids.dim() != 2:
			raise ValueError(f"src_ids must be 2D (batch, seq_len), got shape {src_ids.shape}")
		if tgt_ids.dim() != 2:
			raise ValueError(f"tgt_ids must be 2D (batch, seq_len), got shape {tgt_ids.shape}")
		if src_mask is not None and src_mask.shape != src_ids.shape:
			raise ValueError(f"src_mask shape {src_mask.shape} != src_ids shape {src_ids.shape}")
		if tgt_mask is not None and tgt_mask.shape != tgt_ids.shape:
			raise ValueError(f"tgt_mask shape {tgt_mask.shape} != tgt_ids shape {tgt_ids.shape}")
		
		src = self.pos_encoding(self.src_embed(src_ids))
		tgt = self.pos_encoding(self.tgt_embed(tgt_ids))

		enc_mask = create_padding_mask(src_mask) if src_mask is not None else None
		dec_pad_mask = create_padding_mask(tgt_mask) if tgt_mask is not None else None

		causal = create_causal_mask(tgt.size(1), device=tgt.device)
		if dec_pad_mask is not None:
			self_mask = torch.minimum(dec_pad_mask, causal)
		else:
			self_mask = causal

		enc_output = self.encoder(src, enc_mask)
		dec_output = self.decoder(tgt, enc_output, self_mask=self_mask, cross_mask=enc_mask)

		logits = self.output_proj(dec_output)
		return logits
