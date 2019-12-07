from typing import List
from typing import Tuple
from typing import Union

import librosa
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class ScatterTransform(torch.nn.Module):
    def __init__(self,
                 # filter options,
                 ):
        super().__init__()

    def forward(self, feat: torch.Tensor, ilens: torch.LongTensor) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)

        logmel_feat = (mel_feat + 1e-20).log()
        # Zero padding
        logmel_feat = logmel_feat.masked_fill(
            make_pad_mask(ilens, logmel_feat, 1), 0.0)
        return logmel_feat, ilens
