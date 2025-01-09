import numpy as np
from .dist import Randn, Zeros, Uniform
from typing import Union
from collections.abc import Iterable
from .tensor import Tensor

class Linear:
    def __init__(self, in_size: int, out_size: int, need = False) -> None:
        t = np.sqrt(1/in_size)
        self.weight = Uniform(low=-t, high=t, shape=(out_size, in_size), need=need)
        self.bias = Uniform(low=-t, high=t, shape=(out_size,), need=need)
        self.params = {"weight" : self.weight, "bias" : self.bias}

    def __call__(self, x: Tensor) -> Union[Tensor, AssertionError]:
        assert x.shape[-1] == self.weight.shape[1], f"Expected input tensor shape: (..., {self.weight.shape[1]}), given: (..., {x.shape[-1]})"
        return self.bias + x @ self.weight.T

class Conv2d:
    def __init__(self, in_channels: int, out_channels, kernel_size: Iterable, stride: Union[int, Iterable] = 1, pad: Union[int, Iterable] = 0, pad_val = 0, dilation: Union[int, Iterable] = 1, groups: int = 1, need = False) -> Union[None, AssertionError]:
        assert out_channels % groups == 0 and in_channels % groups == 0, "groups must be a common factor for both in_channels and out_channels"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.pad = (pad, pad) if isinstance(pad, int) else tuple(pad)
        self.pad_val = pad_val
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.groups = groups

        self.in_channels_eff = self.in_channels // self.groups
        self.out_channels_eff = self.out_channels // self.groups

        t = np.sqrt(self.groups/(self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = Uniform(low=-t, high=t, shape=(self.out_channels, self.in_channels_eff, *self.kernel_size), need=need)
        self.bias = Uniform(low=-t, high=t, shape=(self.out_channels,), need=need)
        self.params = {"weight" : self.weight, "bias" : self.bias}

    def __call__(self, x: Tensor) -> Union[Tensor, AssertionError]:
        assert len(x.shape) >= 3, "Convolution permitted only for 3D+ tensors"
        assert x.shape[-2] + self.pad[0] >= self.kernel_size[0] and x.shape[-1] + self.pad[1] >= self.kernel_size[1], "Kernel size exceeds effective input size"

        h = (x.shape[-2] - (self.kernel_size[0] - 1) * self.dilation[0] + 2 * self.pad[0] - 1) // self.stride[0] + 1
        w = (x.shape[-1] - (self.kernel_size[1] - 1) * self.dilation[1] + 2 * self.pad[1] - 1) // self.stride[1] + 1
        out_shape = (*x.shape[:-3], self.out_channels, h, w) if len(x.shape) > 3 else (self.out_channels, h, w)

        num_batch_flat = 1
        for n in x.shape[:-3]: num_batch_flat *= n
        out_shape_batch_flat = (num_batch_flat, *out_shape[-3:])

        out = Zeros(out_shape_batch_flat)
        x_ = Zeros((num_batch_flat, x.shape[-3], x.shape[-2]+2*self.pad[0], x.shape[-1]+2*self.pad[1])) + self.pad_val
        x_[..., self.pad[0]:self.pad[0]+x.shape[-2], self.pad[1]:self.pad[1]+x.shape[-1]] = x

        for b in range(out.shape[0]):   # batch
            for c in range(self.out_channels):
                g = c // self.out_channels_eff
                for i in range(0, x.shape[-2] - (self.kernel_size[0]-1)*self.dilation[0] + 2*self.pad[0] + self.stride[0] - 1, self.stride[0]):
                    idx_i = i // self.stride[0]
                    for j in range(0, x.shape[-1] - (self.kernel_size[1]-1)*self.dilation[1] + 2*self.pad[1] + self.stride[1] - 1, self.stride[1]):
                        idx_j = j // self.stride[1]
                        out[b, c, idx_i, idx_j] = (self.weight[c,...] * x_[b, g*self.in_channels_eff:(g+1)*self.in_channels_eff, i:i+self.kernel_size[0]*self.dilation[0]:self.dilation[0], j:j+self.kernel_size[1]*self.dilation[1]:self.dilation[1]]).sum()
                out[b, c,...] = out[b, c,...] + self.bias[c]


        # circular-fft for conv?

        return out.reshape(out_shape)
