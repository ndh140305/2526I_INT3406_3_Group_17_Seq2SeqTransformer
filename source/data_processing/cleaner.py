import re
from typing import List, Tuple

def clean_data(source_list: List[str], target_list: List[str], max_len: int = 100) -> Tuple[List[str], List[str]]:
    try:
        if not isinstance(source_list, list) or not isinstance(target_list, list):
            raise TypeError("Both inputs must be lists")
        
        if len(source_list) != len(target_list):
            raise ValueError(f"Lists must have equal length. Got {len(source_list)} and {len(target_list)}")
        
        if not source_list:
            raise ValueError("Input lists cannot be empty")
        
        clean_source, clean_target = [], []
        
        for s, t in zip(source_list, target_list):
            try:
                if not isinstance(s, str) or not isinstance(t, str):
                    continue
                
                s_clean = s.strip()
                t_clean = t.strip()
                
                if not s_clean or not t_clean:
                    continue
                
                s_words = s_clean.split()
                t_words = t_clean.split()
                
                if not (0 < len(s_words) <= max_len and 0 < len(t_words) <= max_len):
                    continue
                
                if not _check_balanced_brackets(s_clean) or not _check_balanced_brackets(t_clean):
                    continue
                
                s_normalized = _normalize_text(s_clean)
                t_normalized = _normalize_text(t_clean)
                
                clean_source.append(s_normalized.lower())
                clean_target.append(t_normalized.lower())
                
            except Exception as e:
                print(f"Warning: Skipping sentence pair - {e}")
                continue
        
        if not clean_source:
            raise ValueError("No valid sentence pairs found after cleaning")
        
        print(f"Cleaned {len(clean_source)} sentence pairs from {len(source_list)} original pairs")
        return clean_source, clean_target
        
    except (ValueError, TypeError) as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during cleaning: {e}")
        raise

def _check_balanced_brackets(text: str) -> bool:
    brackets = {'(': ')', '[': ']', '{': '}', '"': '"', "'": "'"}
    stack = []
    
    for char in text:
        if char in brackets:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
    
    return len(stack) == 0


def _normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text