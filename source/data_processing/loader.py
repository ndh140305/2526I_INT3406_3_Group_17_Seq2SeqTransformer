import os
import tarfile
from typing import Tuple, List

def extract_tgz(tgz_path: str, extract_path: str):
    try:
        if not os.path.exists(tgz_path):
            raise FileNotFoundError(f"File not found: {tgz_path}")
        
        os.makedirs(extract_path, exist_ok=True)
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        
        extracted_files = os.listdir(extract_path)
        print(f"Extracted {len(extracted_files)} files to {extract_path}")
        return extracted_files
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except tarfile.TarError as e:
        print(f"Error extracting tar file: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def load_data(file_vi: str, file_en: str) -> Tuple[List[str], List[str]]:
    try:
        if not os.path.exists(file_vi):
            raise FileNotFoundError(f"Vietnamese file not found: {file_vi}")
        if not os.path.exists(file_en):
            raise FileNotFoundError(f"English file not found: {file_en}")
        
        with open(file_vi, 'r', encoding='utf-8') as f:
            target_sentences = [line.strip() for line in f if line.strip()]
        with open(file_en, 'r', encoding='utf-8') as f:
            source_sentences = [line.strip() for line in f if line.strip()]
        
        if not source_sentences or not target_sentences:
            raise ValueError("One or both files are empty")
        
        min_len = min(len(source_sentences), len(target_sentences))
        print(f"Loaded {min_len} sentence pairs")
        return source_sentences[:min_len], target_sentences[:min_len]
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise