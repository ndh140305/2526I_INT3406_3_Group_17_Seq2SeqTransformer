import numpy as np
from typing import List, Dict

def report_data_statistics(source_list: List[str], target_list: List[str], source_vocab: Dict, target_vocab: Dict) -> None:
    try:
        if not source_list or not target_list:
            raise ValueError("Source and target lists cannot be empty")
        
        source_lens = [len(s.split()) for s in source_list]
        target_lens = [len(t.split()) for t in target_list]
        
        print(f"\n{'='*50}")
        print(f"DATA STATISTICS")
        print(f"{'='*50}")
        print(f"Train sentences: {len(source_list)}")
        print(f"Source avg length: {np.mean(source_lens):.1f}, max length: {np.max(source_lens)}")
        print(f"Target avg length: {np.mean(target_lens):.1f}, max length: {np.max(target_lens)}")
        print(f"Source vocab size: {len(source_vocab)}")
        print(f"Target vocab size: {len(target_vocab)}")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"Error in report_data_statistics: {e}")
        raise