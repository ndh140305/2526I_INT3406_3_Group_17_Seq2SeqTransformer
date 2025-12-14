from typing import List, Tuple

import sentencepiece as spm

def convert_to_ids(
    tokenized_source: List[List[str]],
    tokenized_target: List[List[str]],
    source_sp: spm.SentencePieceProcessor,
    target_sp: spm.SentencePieceProcessor,
) -> Tuple[List[List[int]], List[List[int]]]:
    if not isinstance(source_sp, spm.SentencePieceProcessor) or not isinstance(target_sp, spm.SentencePieceProcessor):
        raise TypeError("source_sp and target_sp must be SentencePieceProcessor instances")
    if not isinstance(tokenized_source, list) or not isinstance(tokenized_target, list):
        raise TypeError("tokenized_source and tokenized_target must be lists")

    source_ids = [
        source_sp.EncodeAsIds(tokens) if isinstance(tokens, str) else source_sp.EncodeAsIds(" ".join(tokens))
        for tokens in tokenized_source
    ]

    target_ids = [
        [target_sp.PieceToId("<s>")] + target_sp.EncodeAsIds(" ".join(tokens)) + [target_sp.PieceToId("</s>")]
        for tokens in tokenized_target
    ]

    return source_ids, target_ids