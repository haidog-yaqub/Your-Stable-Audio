import torch
import numpy as np


def scale_shift(x, scale, shift):
    return (x+shift) * scale


def scale_shift_re(x, scale, shift):
    return (x/scale) - shift


def align_seq(source, target_length, mapping_method='hard'):
    source_len = source.shape[1]
    if mapping_method == 'hard':
        mapping_idx = np.round(np.arange(target_length) * source_len / target_length)
        output = source[:, mapping_idx]
    else:
        # TBD
        raise NotImplementedError

    return output


if __name__ == "__main__":

    a = torch.rand(2, 10)
    target_len = 15

    b = align_seq(a, target_len)