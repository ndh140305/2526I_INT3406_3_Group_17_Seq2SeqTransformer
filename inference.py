import torch
import numpy as np
import sentencepiece as spm
import argparse
from source.transformer.transformer import Transformer
from source.transformer.evaluate import greedy_decode, beam_search


def _pad_rows(tensor, new_rows):
    """Pad a weight matrix (rows, dim) to new_rows with zeros."""
    old_rows, dim = tensor.shape
    if new_rows <= old_rows:
        return tensor
    pad = torch.zeros(new_rows - old_rows, dim, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


def _pad_vector(vec, new_rows):
    old_rows = vec.shape[0]
    if new_rows <= old_rows:
        return vec
    pad = torch.zeros(new_rows - old_rows, dtype=vec.dtype, device=vec.device)
    return torch.cat([vec, pad], dim=0)


def load_model(checkpoint_path, d_model=128, num_encoder_layers=2, num_decoder_layers=2, 
               d_ff=512, num_heads=4, device='cuda', src_vocab_override=None, tgt_vocab_override=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    ckpt_src_vocab = state_dict['src_embed.embedding.weight'].shape[0]
    ckpt_tgt_vocab = state_dict['tgt_embed.embedding.weight'].shape[0]
    max_seq_len = state_dict['pos_encoding.pe'].shape[1]

    src_vocab_size = src_vocab_override or ckpt_src_vocab
    tgt_vocab_size = tgt_vocab_override or ckpt_tgt_vocab

    if src_vocab_size > ckpt_src_vocab:
        state_dict['src_embed.embedding.weight'] = _pad_rows(state_dict['src_embed.embedding.weight'], src_vocab_size)
    if tgt_vocab_size > ckpt_tgt_vocab:
        state_dict['tgt_embed.embedding.weight'] = _pad_rows(state_dict['tgt_embed.embedding.weight'], tgt_vocab_size)
        state_dict['output_proj.weight'] = _pad_rows(state_dict['output_proj.weight'], tgt_vocab_size)
        if 'output_proj.bias' in state_dict:
            state_dict['output_proj.bias'] = _pad_vector(state_dict['output_proj.bias'], tgt_vocab_size)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        pad_token_id=0
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def translate(model, sp_src, sp_tgt, text, device, max_len=50, use_beam_search=False, beam_width=5):
    src_tokens = sp_src.EncodeAsIds(text)
    src_ids = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_mask = torch.ones_like(src_ids)
    
    with torch.no_grad():
        if use_beam_search:
            output = beam_search(model, src_ids, src_mask, max_len, 
                                bos_token_id=1, eos_token_id=2, beam_width=beam_width)
        else:
            output = greedy_decode(model, src_ids, src_mask, max_len, 
                                  bos_token_id=1, eos_token_id=2)
    
    output_ids = output[0].cpu().numpy().tolist()
    
    output_ids = [id for id in output_ids if id not in [0, 1, 2]]
    translation = sp_tgt.DecodeIds(output_ids)
    
    return translation


def main():
    parser = argparse.ArgumentParser(description='Translate text using trained Transformer model')
    parser.add_argument('--checkpoint', default='training/checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--source_sp', default='data/processed/source_sp.model', help='Source SentencePiece model')
    parser.add_argument('--target_sp', default='data/processed/target_sp.model', help='Target SentencePiece model')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='FFN hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--beam_search', action='store_true', help='Use beam search instead of greedy')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search')
    parser.add_argument('--text', type=str, help='Text to translate (if not provided, enters interactive mode)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    print(f" Loading tokenizers...")
    sp_src = spm.SentencePieceProcessor()
    sp_tgt = spm.SentencePieceProcessor()
    sp_src.Load(args.source_sp) 
    sp_tgt.Load(args.target_sp)  
    
    print(f" Loading model from {args.checkpoint}...")
    model = load_model(
        args.checkpoint,
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        device=device,
        src_vocab_override=sp_src.vocab_size(),
        tgt_vocab_override=sp_tgt.vocab_size()
    )
    print("Model loaded successfully!")
    
    decode_method = "Beam Search" if args.beam_search else "Greedy"
    print(f"Decoding method: {decode_method}")
    if args.beam_search:
        print(f"  Beam width: {args.beam_width}")
    
    print("\n" + "="*60)
    print("TRANSFORMER TRANSLATION")
    print("="*60)
    
    if args.text:
        translation = translate(model, sp_src, sp_tgt, args.text, device, 
                               use_beam_search=args.beam_search, beam_width=args.beam_width)
        print(f"\nInput:  {args.text}")
        print(f"Output: {translation}")
    else:
        print("\nInteractive Translation Mode")
        print("   Type your text (Vietnamese) and press Enter")
        print("   Type 'quit', 'exit', or 'q' to exit\n")
        
        while True:
            try:
                text = input("🇻🇳 Vietnamese: ")
                if text.lower().strip() in ['quit', 'exit', 'q', '']:
                    break
                
                if text.strip():
                    translation = translate(model, sp_src, sp_tgt, text, device,
                                          use_beam_search=args.beam_search, beam_width=args.beam_width)
                    print(f"🇬🇧 English:    {translation}\n")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    print("\n" + "="*60)
    print(" Done!")

if __name__ == "__main__":
    main()
