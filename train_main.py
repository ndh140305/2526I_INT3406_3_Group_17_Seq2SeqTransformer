import argparse
import numpy as np
import torch

from source.transformer.transformer import Transformer
from source.transformer.train import run_training
from source.data_processing.dataloader import create_dataloader


def build_masks(ids: np.ndarray, pad_id: int) -> np.ndarray:
    return (ids != pad_id).astype(np.int64)


def split_train_val(src_ids, tgt_ids, src_mask, tgt_mask, val_ratio=0.05):
    n = len(src_ids)
    split = max(1, int(n * (1 - val_ratio)))
    return (
        (src_ids[:split], tgt_ids[:split], src_mask[:split], tgt_mask[:split]),
        (src_ids[split:], tgt_ids[split:], src_mask[split:], tgt_mask[split:]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_ids", default="data/processed/train_source_ids.npy")
    parser.add_argument("--target_ids", default="data/processed/train_target_ids.npy")
    parser.add_argument("--source_mask", default=None)
    parser.add_argument("--target_mask", default=None)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bos_token_id", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=2)
    parser.add_argument("--resume_checkpoint", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="Starting epoch number")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #tải data
    source_ids = np.load(args.source_ids)
    target_ids = np.load(args.target_ids)
    if args.source_mask:
        source_mask = np.load(args.source_mask)
    else:
        source_mask = build_masks(source_ids, args.pad_token_id)
    if args.target_mask:
        target_mask = np.load(args.target_mask)
    else:
        target_mask = build_masks(target_ids, args.pad_token_id)

    #split
    (tr_src, tr_tgt, tr_src_m, tr_tgt_m), (va_src, va_tgt, va_src_m, va_tgt_m) = split_train_val(
        source_ids, target_ids, source_mask, target_mask, val_ratio=args.val_ratio
    )

    train_loader = create_dataloader(
        tr_src, tr_tgt, tr_src_m, tr_tgt_m,
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = create_dataloader(
        va_src, va_tgt, va_src_m, va_tgt_m,
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )

    src_vocab_size = int(source_ids.max()) + 1
    tgt_vocab_size = int(target_ids.max()) + 1
    max_seq_len = max(source_ids.shape[1], target_ids.shape[1])

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_token_id=args.pad_token_id,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=max_seq_len,
    )
    
    if args.resume_checkpoint:
        print(f"\n Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Checkpoint loaded. Starting from epoch {args.start_epoch}")

    metrics_history = run_training(
        model,
        train_loader,
        val_loader,
        pad_token_id=args.pad_token_id,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        log_every=100,
        checkpoint_dir="training/checkpoints",
        metrics_file="training/metrics.json",
        compute_bleu=True,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model saved to: training/checkpoints/best_model.pth")
    print(f"Metrics saved to: training/metrics.json")
    print("\nFinal Metrics:")
    for metric in metrics_history:
        print(f"  Epoch {metric['epoch']}: train_loss={metric['train_loss']:.4f}, val_loss={metric['val_loss']:.4f}")
        print(f"    BLEU={metric['val_bleu']:.4f}, Gemini={metric['val_gemini']:.4f}, Perplexity={metric['val_perplexity']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
