import numpy as np

from ._convolution import col2im, im2col


def window_counts(input_shape, kernel_size, stride, padding, dilation, output_spatial):
    valid = np.ones((1, 1, *input_shape[-2:]), dtype=np.int64)
    columns = im2col(valid, kernel_size, stride, padding, dilation, output_spatial)
    return columns, columns.sum(axis=(2, 3), dtype=np.int64)


def max_pool2d_forward(input, kernel_size, stride, padding, dilation, output_spatial):
    batch, channels, _, width = input.shape
    kh, kw = kernel_size
    oh, ow = output_spatial
    valid, counts = window_counts(input.shape, kernel_size, stride, padding, dilation, output_spatial)
    if np.any(counts == 0):
        raise ValueError("MaxPool windows must contain at least one input element.")
    valid = valid.reshape(1, 1, kh * kw, oh, ow).astype(bool)
    columns = im2col(input, kernel_size, stride, padding, dilation, output_spatial).reshape(batch, channels, kh * kw, oh, ow)
    columns = np.where(valid, columns, -np.inf)
    offsets = np.argmax(columns, axis=2)
    selected_valid = np.take_along_axis(np.broadcast_to(valid, columns.shape), offsets[:, :, None], axis=2)[:, :, 0]
    offsets = np.where(selected_valid, offsets, np.argmax(valid, axis=2))
    output = np.take_along_axis(columns, offsets[:, :, None], axis=2)[:, :, 0]
    rows = np.arange(oh).reshape(1, 1, oh, 1) * stride[0] - padding[0] + (offsets // kw) * dilation[0]
    cols = np.arange(ow).reshape(1, 1, 1, ow) * stride[1] - padding[1] + (offsets % kw) * dilation[1]
    indices = rows * width + cols
    indices.flags.writeable = False
    return output, indices


def max_pool2d_backward(gradient, input_shape, indices):
    batch, channels, height, width = input_shape
    result = np.zeros((batch, channels, height * width), dtype=gradient.dtype)
    batch_indices = np.arange(batch).reshape(batch, 1, 1, 1)
    channel_indices = np.arange(channels).reshape(1, channels, 1, 1)
    np.add.at(result, (batch_indices, channel_indices, indices), gradient)
    return result.reshape(input_shape)


def avg_pool2d_divisor(input_shape, kernel_size, stride, padding, output_spatial, count_include_pad, dtype):
    if count_include_pad:
        return dtype.type(kernel_size[0] * kernel_size[1])
    _, counts = window_counts(input_shape, kernel_size, stride, padding, (1, 1), output_spatial)
    if np.any(counts == 0):
        raise ValueError("AvgPool windows must contain at least one input element when count_include_pad=False.")
    return counts.astype(dtype)


def avg_pool2d_forward(input, kernel_size, stride, padding, output_spatial, count_include_pad):
    columns = im2col(input, kernel_size, stride, padding, (1, 1), output_spatial)
    divisor = avg_pool2d_divisor(input.shape, kernel_size, stride, padding, output_spatial, count_include_pad, input.dtype)
    return columns.sum(axis=(2, 3), dtype=input.dtype) / divisor


def avg_pool2d_backward(gradient, input_shape, kernel_size, stride, padding, count_include_pad):
    divisor = avg_pool2d_divisor(input_shape, kernel_size, stride, padding, gradient.shape[-2:], count_include_pad, gradient.dtype)
    scaled = gradient / divisor
    columns = np.broadcast_to(scaled[:, :, None, None], (*input_shape[:2], *kernel_size, *gradient.shape[-2:]))
    return col2im(columns, input_shape, stride, padding, (1, 1))
