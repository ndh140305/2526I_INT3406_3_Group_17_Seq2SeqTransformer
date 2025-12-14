import torch
import torch.optim as optim
import json
from pathlib import Path

from .transformer import Transformer
from .loss import sequence_cross_entropy
from .evaluate import compute_bleu, compute_gemini_score, greedy_decode, beam_search

class WarmupScheduler:
	def __init__(self, optimizer, warmup_steps: int, base_lr: float):
		self.optimizer = optimizer
		self.warmup_steps = warmup_steps
		self.base_lr = base_lr
		self.step_num = 0

	def step(self):
		self.step_num += 1
		if self.step_num <= self.warmup_steps:
			scale = self.step_num / self.warmup_steps
			lr = self.base_lr * scale
			for param_group in self.optimizer.param_groups:
				param_group["lr"] = lr
		self.optimizer.step()


def shift_target(tgt_ids: torch.Tensor):
	return tgt_ids[:, :-1], tgt_ids[:, 1:]


def train_one_epoch(model, dataloader, optimizer, scheduler, pad_token_id: int, device: torch.device, log_every: int = 100):
	model.train()
	total_loss = 0.0
	steps = 0
	for batch in dataloader:
		src_ids = batch["source_ids"].to(device)
		tgt_ids = batch["target_ids"].to(device)
		src_mask = batch["source_mask"].to(device)
		tgt_mask = batch["target_mask"].to(device)

		optimizer.zero_grad()

		decoder_input, decoder_target = shift_target(tgt_ids)
		decoder_input_mask = tgt_mask[:, :-1]
		# Forward
		logits = model(src_ids, decoder_input, src_mask=src_mask, tgt_mask=decoder_input_mask)
		loss = sequence_cross_entropy(logits, decoder_target, pad_token_id)

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		scheduler.step()

		total_loss += loss.item()
		steps += 1

		if log_every and steps % log_every == 0:
			print(f"step {steps}: loss={loss.item():.4f}")

	return total_loss / max(steps, 1)

def evaluate(model, dataloader, pad_token_id: int, device: torch.device, compute_bleu_score: bool = False, bos_token_id: int = 1, eos_token_id: int = 2):
	model.eval()
	total_loss = 0.0
	steps = 0
	all_references = []
	all_hypotheses = []
	
	with torch.no_grad():
		for batch in dataloader:
			src_ids = batch["source_ids"].to(device)
			tgt_ids = batch["target_ids"].to(device)
			src_mask = batch["source_mask"].to(device)
			tgt_mask = batch["target_mask"].to(device)

			decoder_input, decoder_target = shift_target(tgt_ids)
			decoder_input_mask = tgt_mask[:, :-1]

			logits = model(src_ids, decoder_input, src_mask=src_mask, tgt_mask=decoder_input_mask)
			loss = sequence_cross_entropy(logits, decoder_target, pad_token_id)

			total_loss += loss.item()
			steps += 1
			
			# Compute BLEU if requested
			if compute_bleu_score:
				for ref, hyp in zip(decoder_target.cpu().numpy().tolist(), torch.argmax(logits, dim=-1).cpu().numpy().tolist()):
					all_references.append(ref)
					all_hypotheses.append(hyp)

	avg_loss = total_loss / max(steps, 1)
	perplexity = torch.exp(torch.tensor(avg_loss)).item()
	bleu_score = 0.0
	gemini_score = 0.0
	
	if compute_bleu_score and all_references and all_hypotheses:
		bleu_score = compute_bleu(all_references, all_hypotheses)
		gemini_score = compute_gemini_score(all_references, all_hypotheses)
	
	return avg_loss, bleu_score, gemini_score, perplexity


def build_optimizer(model, lr: float = 3e-4, weight_decay: float = 0.0):
	return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))


def run_training(
	model: Transformer,
	train_loader,
	val_loader,
	pad_token_id: int,
	device: torch.device,
	epochs: int = 1,
	lr: float = 3e-4,
	warmup_steps: int = 400,
	log_every: int = 100,
	checkpoint_dir: str = "training/checkpoints",
	metrics_file: str = "training/metrics.json",
	compute_bleu: bool = True,
):
	model.to(device)
	optimizer = build_optimizer(model, lr=lr)
	scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps, base_lr=lr)
	
	# Create checkpoint dir
	Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
	Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
	
	metrics_history = []

	for epoch in range(1, epochs + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, pad_token_id, device, log_every=log_every)
		val_loss, val_bleu, val_gemini, val_perplexity = evaluate(model, val_loader, pad_token_id, device, compute_bleu_score=compute_bleu)
		
		epoch_metrics = {
			"epoch": epoch,
			"train_loss": train_loss,
			"val_loss": val_loss,
			"val_bleu": val_bleu,
			"val_gemini": val_gemini,
			"val_perplexity": val_perplexity,
			"lr": optimizer.param_groups[0]["lr"]
		}
		metrics_history.append(epoch_metrics)
		
		print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_bleu={val_bleu:.4f}, val_gemini={val_gemini:.4f}, perplexity={val_perplexity:.4f}")
		
		# Save checkpoint
		ckpt_path = Path(checkpoint_dir) / f"model_epoch_{epoch}.pth"
		torch.save({
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"train_loss": train_loss,
			"val_loss": val_loss,
			"val_bleu": val_bleu,
			"val_gemini": val_gemini,
			"val_perplexity": val_perplexity,
		}, ckpt_path)
		print(f"Checkpoint saved to {ckpt_path}")
	
	# Save best model (last epoch)
	best_model_path = Path(checkpoint_dir) / "best_model.pth"
	torch.save(model.state_dict(), best_model_path)
	print(f"Best model saved to {best_model_path}")
	
	# Save metrics
	with open(metrics_file, "w") as f:
		json.dump(metrics_history, f, indent=2)
	print(f"Metrics saved to {metrics_file}")
	
	return metrics_history

