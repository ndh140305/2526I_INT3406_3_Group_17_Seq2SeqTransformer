import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
	def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
		super().__init__()
		self.linear1 = nn.Linear(d_model, d_ff) #ánh xạ lên không gian cao hơn
		self.linear2 = nn.Linear(d_ff, d_model) #ánh xạ về
		self.dropout = nn.Dropout(dropout) #tắt chiều ngẫu nhiên giảm overfit
		self.relu = nn.ReLU() #phi tuyến dữ liệu
		self.layer_norm = nn.LayerNorm(d_model) #chuẩn hóa

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = x
		x = self.linear1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.linear2(x)
		x = self.dropout(x)
		x = self.layer_norm(x + residual)
		return x
