from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.
    1 for non-padded part and 0 for padded part.
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention."""
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e+10
    return mask


def padding(data: List[torch.Tensor]):
    """ Padding the data into batch data

    Parameters
    ----------
        data: List[Tensor], shape of Tensor (n_mels, T) e.g. (128, T)

    Returns:
    -------
        feats (Tensor): (B, n_mels, T_max)
        feats_lengths (Tensor): (B,)
    """
    sample = data
    assert isinstance(sample, list)
    if not sample: # Handle empty list case
        return torch.empty(0), torch.empty(0)
        
    feats_lengths = torch.tensor([s.size(1) for s in sample],
                                 dtype=torch.int32)
    # Transpose from (n_mels, T) to (T, n_mels) for pad_sequence
    feats = [s.t() for s in sample] 
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)

    # Transpose back from (B, T_max, n_mels) to (B, n_mels, T_max)
    return padded_feats.transpose(1, 2), feats_lengths
