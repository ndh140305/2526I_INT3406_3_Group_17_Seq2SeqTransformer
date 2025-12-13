import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict


class TranslationDataset(Dataset):
    def __init__(self, source_ids: np.ndarray, target_ids: np.ndarray, 
                 source_mask: np.ndarray, target_mask: np.ndarray):
        try:
            if not all(isinstance(x, np.ndarray) for x in [source_ids, target_ids, source_mask, target_mask]):
                raise TypeError("All inputs must be numpy arrays")
            
            if len(source_ids) != len(target_ids):
                raise ValueError("source_ids and target_ids must have same length")
            
            self.source_ids = torch.from_numpy(source_ids).long()
            self.target_ids = torch.from_numpy(target_ids).long()
            self.source_mask = torch.from_numpy(source_mask).long()
            self.target_mask = torch.from_numpy(target_mask).long()
            
        except Exception as e:
            print(f"Error initializing TranslationDataset: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.source_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'source_ids': self.source_ids[idx],
            'target_ids': self.target_ids[idx],
            'source_mask': self.source_mask[idx],
            'target_mask': self.target_mask[idx]
        }


def create_dataloader(source_ids: np.ndarray, target_ids: np.ndarray,
                     source_mask: np.ndarray, target_mask: np.ndarray,
                     batch_size: int = 32, shuffle: bool = True, 
                     num_workers: int = 0) -> DataLoader:
    try:
        dataset = TranslationDataset(source_ids, target_ids, source_mask, target_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                               num_workers=num_workers, pin_memory=True)
        
        print(f"Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
        return dataloader
        
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        raise
