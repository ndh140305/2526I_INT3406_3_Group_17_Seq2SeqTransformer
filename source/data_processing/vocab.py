from typing import Dict

import sentencepiece as spm

def build_vocab(sp: spm.SentencePieceProcessor) -> Dict[str, int]:
    if not isinstance(sp, spm.SentencePieceProcessor):
        raise TypeError("sp must be a SentencePieceProcessor instance")

    size = sp.GetPieceSize()
    if size <= 0:
        raise ValueError("SentencePiece model has no vocabulary")

    return {sp.IdToPiece(i): i for i in range(size)}