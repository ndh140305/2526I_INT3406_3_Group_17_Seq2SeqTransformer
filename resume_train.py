import argparse
import numpy as np
import torch
import os
from pathlib import Path
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
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument("--checkpoint", default=None, 
                       help="Path to checkpoint to resume from (auto-detect if None)")
    parser.add_argument("--from_epoch", type=int, default=None,
                       help="Resume from specific epoch (e.g. 5 → loads model_epoch_5.pth)")
    parser.add_argument("--source_ids", default="data/processed/train_source_ids.npy")
    parser.add_argument("--target_ids", default="data/processed/train_target_ids.npy")
    parser.add_argument("--source_mask", default=None)
    parser.add_argument("--target_mask", default=None)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10, help="Additional epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.from_epoch:
            checkpoint_path = f"training/checkpoints/model_epoch_{args.from_epoch}.pth"
        else:
            checkpoint_path = "training/checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"Available checkpoints:")
        ckpt_dir = "training/checkpoints"
        if os.path.exists(ckpt_dir):
            for f in sorted(os.listdir(ckpt_dir)):
                print(f"  - {f}")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Loading data...")
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

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    src_vocab_size = state_dict['src_embed.embedding.weight'].shape[0]
    tgt_vocab_size = state_dict['tgt_embed.embedding.weight'].shape[0]
    max_seq_len = state_dict['pos_encoding.pe'].shape[1]

    print(f"  Checkpoint architecture:")
    print(f"  src_vocab: {src_vocab_size}, tgt_vocab: {tgt_vocab_size}")
    print(f"  max_seq_len: {max_seq_len}")

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_token_id=args.pad_token_id,
        d_model=256,  
        num_heads=8,
        num_encoder_layers=4, 
        num_decoder_layers=4, 
        d_ff=1024, 
        dropout=0.1,
        max_seq_len=max_seq_len,
    ).to(device)

    model.load_state_dict(state_dict)
    print(f"  Model loaded with checkpoint weights")

    print(f"\nResuming training for {args.epochs} more epochs...")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    
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
    )

    print(f"\nTraining complete!")
    print(f"Final metrics: {metrics_history[-1] if metrics_history else 'N/A'}")
    
    if os.path.exists("training/checkpoints/best_model.pth"):
        try:
            from google.colab import drive

            drive.mount('/content/drive', force_remount=True)
            
            backup_dir = '/content/drive/MyDrive/nlp_project_checkpoint'
            os.makedirs(backup_dir, exist_ok=True)
            
            import shutil
            backup_path = os.path.join(backup_dir, 'best_model.pth')
            shutil.copy('training/checkpoints/best_model.pth', backup_path)
            print(f"Checkpoint backed up to: {backup_path}")
        except ImportError:
            print("Not running on Colab, skipping Google Drive backup")


if __name__ == '__main__':
    main()
