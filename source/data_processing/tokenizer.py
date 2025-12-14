import os
import sentencepiece as spm
from typing import List, Tuple

def train_sentencepiece(sentences: List[str], model_prefix: str, vocab_size: int = 32000) -> spm.SentencePieceProcessor:
    try:
        if not sentences:
            raise ValueError("Sentences list cannot be empty")
        
        if not isinstance(sentences, list):
            raise TypeError("Sentences must be a list of strings")
        
        if not all(isinstance(s, str) for s in sentences):
            raise TypeError("All items in sentences must be strings")
        
        if not model_prefix:
            raise ValueError("model_prefix cannot be empty")
        
        tmp_file = f"{model_prefix}_input.txt"
        
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                for line in sentences:
                    f.write(line.strip() + "\n")
            
            print(f"Training SentencePiece model with {len(sentences)} sentences...")
            
            spm.SentencePieceTrainer.Train(
                input=tmp_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=1.0,
                model_type='bpe'
            )
            
            sp = spm.SentencePieceProcessor()
            sp.Load(f"{model_prefix}.model")
            
            print(f"SentencePiece model trained successfully. Vocab size: {sp.piece_size()}")
            
            return sp
            
        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            raise
        except Exception as e:
            print(f"Error during SentencePiece training: {e}")
            raise
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
    
    except (ValueError, TypeError) as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error in train_sentencepiece: {e}")
        raise

def tokenize_data(sp: spm.SentencePieceProcessor, sentences: List[str]) -> List[List[str]]:
    try:
        if not isinstance(sp, spm.SentencePieceProcessor):
            raise TypeError("sp must be a SentencePieceProcessor instance")
        
        if not sentences:
            raise ValueError("Sentences list cannot be empty")
        
        if not isinstance(sentences, list):
            raise TypeError("Sentences must be a list of strings")
        
        tokenized = []
        
        for sentence in sentences:
            try:
                if not isinstance(sentence, str):
                    print(f"Warning: Skipping non-string sentence: {sentence}")
                    continue
                
                tokens = sp.EncodeAsPieces(sentence.strip())
                tokenized.append(tokens)
                
            except Exception as e:
                print(f"Warning: Error tokenizing sentence - {e}")
                continue
        
        if not tokenized:
            raise ValueError("No valid sentences were tokenized")
        
        print(f"Tokenized {len(tokenized)} sentences")
        return tokenized
        
    except (ValueError, TypeError) as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error in tokenize_data: {e}")
        raise