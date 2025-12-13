import numpy as np
from typing import List, Tuple


def pad_and_create_mask(id_list: List[List[int]], pad_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    try:
        if not id_list:
            raise ValueError("id_list cannot be empty")
        
        max_len = max(len(ids) for ids in id_list)
        
        padded = np.array([ids + [pad_id] * (max_len - len(ids)) for ids in id_list], dtype=np.int64)
        mask = np.array([[1] * len(ids) + [0] * (max_len - len(ids)) for ids in id_list], dtype=np.int64)
        
        print(f"Padded {len(id_list)} sequences to max length {max_len}")
        return padded, mask
        
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error in pad_and_create_mask: {e}")
        raise