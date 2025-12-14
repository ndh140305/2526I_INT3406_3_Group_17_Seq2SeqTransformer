import os
import numpy as np
from source.data_processing.loader import load_data, extract_tgz
from source.data_processing.cleaner import clean_data
from source.data_processing.tokenizer import train_sentencepiece, tokenize_data
from source.data_processing.vocab import build_vocab
from source.data_processing.convert import convert_to_ids
from source.data_processing.padding import pad_and_create_mask
from source.utils.metrics import report_data_statistics

def run_preprocessing_pipeline(tgz_path: str, extract_path: str, processed_path: str, vocab_size: int = 32000):
    os.makedirs(processed_path, exist_ok=True)
    
    extract_tgz(tgz_path, extract_path)
    file_en = os.path.join(extract_path, "train.en")
    file_vi = os.path.join(extract_path, "train.vi")
    
    source, target = load_data(file_vi, file_en)
    source, target = clean_data(source, target)
    
    source_sp = train_sentencepiece(source, os.path.join(processed_path, "source_sp"), vocab_size)
    target_sp = train_sentencepiece(target, os.path.join(processed_path, "target_sp"), vocab_size)
    
    tokenized_source = tokenize_data(source_sp, source)
    tokenized_target = tokenize_data(target_sp, target)
    
    source_vocab = build_vocab(source_sp)
    target_vocab = build_vocab(target_sp)
    
    source_ids, target_ids = convert_to_ids(tokenized_source, tokenized_target, source_sp, target_sp)
    
    pad_id_source = source_sp.PieceToId('<pad>')
    pad_id_target = target_sp.PieceToId('<pad>')
    padded_source, source_mask = pad_and_create_mask(source_ids, pad_id_source)
    padded_target, target_mask = pad_and_create_mask(target_ids, pad_id_target)
    
    np.save(os.path.join(processed_path, "train_source_ids.npy"), padded_source)
    np.save(os.path.join(processed_path, "train_target_ids.npy"), padded_target)
    
    report_data_statistics(source, target, source_vocab, target_vocab)
    
    print("Preprocessing pipeline finished. Processed files saved in:", processed_path)

if __name__ == "__main__":
    tgz_path = "data/raw/train-en-vi.tgz"
    extract_path = "data/extracted/"
    processed_path = "data/processed/"
    run_preprocessing_pipeline(tgz_path, extract_path, processed_path)
