from .dist import Randn, Zeros
from typing import Union
from collections.abc import Iterable

class Conv2d:
    def __init__(self, in_channels: int, out_channels, kernel_size: Iterable, stride: Union[int, Iterable] = 1, pad: [int, Iterable] = 0, pad_val = 0, need = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.weight = Randn((out_channels, in_channels, *self.kernel_size), need=need)      # initialization will come from different distribution (Kaiming and others?)
        self.bias = Randn((out_channels,), need=need)                                       # ...
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.pad = (pad, pad) if isinstance(pad, int) else tuple(pad)
        self.pad_val = pad_val

    def __call__(self, x):
        assert x.shape[1] + self.pad[0] >= self.kernel_size[0] and x.shape[2] + self.pad[1] >= self.kernel_size[1], f"Kernel size exceeds effective input size"

        h = (x.shape[1] - self.kernel_size[0] + 2*self.pad[0]) // self.stride[0] + 1
        w = (x.shape[2] - self.kernel_size[1] + 2*self.pad[1]) // self.stride[1] + 1
        out = Zeros((self.out_channels, h, w))
        x_ = Zeros((x.shape[0], x.shape[1]+2*self.pad[0], x.shape[2]+2*self.pad[1]))
        x_ = x_ + self.pad_val
        x_[:, self.pad[0]:self.pad[0]+x.shape[1], self.pad[1]:self.pad[1]+x.shape[2]] = x

        for c in range(self.out_channels):
            for i in range(0, x.shape[1] - self.kernel_size[0] + 2*self.pad[0] + self.stride[0], self.stride[0]):
                idx_i = i // self.stride[0]
                for j in range(0, x.shape[2] - self.kernel_size[1] + 2*self.pad[1] + self.stride[1], self.stride[1]):
                    idx_j = j // self.stride[1]
                    out[c, idx_i, idx_j] = (self.weight[c,...] * x_[:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]]).sum()
            out[c,...] = out[c,...] + self.bias[c]

        return out

