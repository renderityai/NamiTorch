import numpy as np

from .backends import namespace, readonly, same_device

from ._convolution import col2im, im2col


@same_device
def window_counts(input_shape, kernel_size, stride, padding, dilation, output_spatial, device=None):
    xp = namespace(device)
    valid = xp.ones((1, 1, *input_shape[-2:]), dtype=np.int64)
    columns = im2col(valid, kernel_size, stride, padding, dilation, output_spatial)
    return columns, columns.sum(axis=(2, 3), dtype=np.int64)


@same_device
def max_pool2d_forward(input, kernel_size, stride, padding, dilation, output_spatial):
    xp = namespace(input)
    batch, channels, _, width = input.shape
    kh, kw = kernel_size
    oh, ow = output_spatial
    valid, counts = window_counts(input.shape, kernel_size, stride, padding, dilation, output_spatial, input)
    if xp.any(counts == 0):
        raise ValueError("MaxPool windows must contain at least one input element.")
    valid = valid.reshape(1, 1, kh * kw, oh, ow).astype(bool)
    columns = im2col(input, kernel_size, stride, padding, dilation, output_spatial).reshape(batch, channels, kh * kw, oh, ow)
    columns = xp.where(valid, columns, -np.inf)
    offsets = xp.argmax(columns, axis=2)
    selected_valid = xp.take_along_axis(xp.broadcast_to(valid, columns.shape), offsets[:, :, None], axis=2)[:, :, 0]
    offsets = xp.where(selected_valid, offsets, xp.argmax(valid, axis=2))
    output = xp.take_along_axis(columns, offsets[:, :, None], axis=2)[:, :, 0]
    rows = xp.arange(oh).reshape(1, 1, oh, 1) * stride[0] - padding[0] + (offsets // kw) * dilation[0]
    cols = xp.arange(ow).reshape(1, 1, 1, ow) * stride[1] - padding[1] + (offsets % kw) * dilation[1]
    indices = rows * width + cols
    readonly(indices)
    return output, indices


@same_device
def max_pool2d_backward(gradient, input_shape, indices):
    xp = namespace(gradient)
    batch, channels, height, width = input_shape
    result = xp.zeros((batch, channels, height * width), dtype=gradient.dtype)
    batch_indices = xp.arange(batch).reshape(batch, 1, 1, 1)
    channel_indices = xp.arange(channels).reshape(1, channels, 1, 1)
    xp.add.at(result, (batch_indices, channel_indices, indices), gradient)
    return result.reshape(input_shape)


@same_device
def avg_pool2d_divisor(input_shape, kernel_size, stride, padding, output_spatial, count_include_pad, dtype, device=None):
    xp = namespace(device)
    if count_include_pad:
        return dtype.type(kernel_size[0] * kernel_size[1])
    _, counts = window_counts(input_shape, kernel_size, stride, padding, (1, 1), output_spatial, device)
    if xp.any(counts == 0):
        raise ValueError("AvgPool windows must contain at least one input element when count_include_pad=False.")
    return counts.astype(dtype)


@same_device
def avg_pool2d_forward(input, kernel_size, stride, padding, output_spatial, count_include_pad):
    columns = im2col(input, kernel_size, stride, padding, (1, 1), output_spatial)
    divisor = avg_pool2d_divisor(input.shape, kernel_size, stride, padding, output_spatial, count_include_pad, input.dtype, input)
    return columns.sum(axis=(2, 3), dtype=input.dtype) / divisor


@same_device
def avg_pool2d_backward(gradient, input_shape, kernel_size, stride, padding, count_include_pad):
    xp = namespace(gradient)
    divisor = avg_pool2d_divisor(input_shape, kernel_size, stride, padding, gradient.shape[-2:], count_include_pad, gradient.dtype, gradient)
    scaled = gradient / divisor
    columns = xp.broadcast_to(scaled[:, :, None, None], (*input_shape[:2], *kernel_size, *gradient.shape[-2:]))
    return col2im(columns, input_shape, stride, padding, (1, 1))
