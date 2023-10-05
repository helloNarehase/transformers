# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List
import numpy as np_
from sentencepiece import SentencePieceProcessor


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str , f"arg '{s}' is Type Not str : {type(s)}"
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
def bigtokenizing(tok:Tokenizer, x, maxlength = 200, eAt = True):
    xc = []
    for i in x:
        pad = np_.zeros(maxlength)
        a = tok.encode(i, eAt, not eAt)
        pad[maxlength-len(a):] = a
        xc.append(pad)
    return np_.array(xc)
if "__main__" == __name__:
    tok = Tokenizer("tokenizer.model")
    print(tok.encode("안녕 나는 나래야", False, False))
    print(tok.n_words)
    print(tok.decode([29871, 31734, 238, 136, 152, 29871, 31207, 31081, 29871, 31207, 238, 161, 155, 239, 152, 191]))