import numpy as np


def im2col(input, kernel_size, stride, padding, dilation, output_spatial):
    batch, channels, _, _ = input.shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    oh, ow = output_spatial
    padded = np.pad(input, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if ph or pw else input
    columns = np.empty((batch, channels, kh, kw, oh, ow), dtype=input.dtype)
    for row in range(kh):
        for column in range(kw):
            start_h, start_w = row * dh, column * dw
            columns[:, :, row, column] = padded[:, :, start_h:start_h + oh * sh:sh, start_w:start_w + ow * sw:sw]
    return columns


def col2im(columns, input_shape, stride, padding, dilation):
    batch, channels, height, width = input_shape
    _, _, kh, kw, oh, ow = columns.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    padded = np.zeros((batch, channels, height + 2 * ph, width + 2 * pw), dtype=columns.dtype)
    for row in range(kh):
        for column in range(kw):
            start_h, start_w = row * dh, column * dw
            padded[:, :, start_h:start_h + oh * sh:sh, start_w:start_w + ow * sw:sw] += columns[:, :, row, column]
    return padded[:, :, ph:ph + height, pw:pw + width]


def conv2d_forward(input, weight, bias, stride, padding, dilation, groups, output_spatial):
    batch, channels, _, _ = input.shape
    out_channels, _, kh, kw = weight.shape
    oh, ow = output_spatial
    inner = channels // groups * kh * kw
    columns = im2col(input, (kh, kw), stride, padding, dilation, output_spatial)
    columns = columns.reshape(batch, groups, inner, oh * ow)
    kernels = weight.reshape(groups, out_channels // groups, inner)
    output = np.matmul(kernels, columns).reshape(batch, out_channels, oh, ow)
    if bias is not None:
        output += bias.reshape(1, out_channels, 1, 1)
    return output
