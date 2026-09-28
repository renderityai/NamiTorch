import math

import numpy as np

from .kernels import CUDAKernels, _KernelKey, _OPTIONS


_MAX_WIDTH = 8192
_ROWS_PER_TILE = 32
_REDUCTIONS = '''
__device__ float norm_sum(float value, float* shared) {
    unsigned int lane = threadIdx.x;
    shared[lane] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}
__device__ void norm_welford(float count, float mean, float m2, float* shared, float* result_mean, float* result_m2) {
    unsigned int lane = threadIdx.x;
    float* counts = shared;
    float* means = shared + blockDim.x;
    float* moments = shared + 2 * blockDim.x;
    counts[lane] = count;
    means[lane] = mean;
    moments[lane] = m2;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride && counts[lane + stride] > 0.0f) {
            float right_count = counts[lane + stride];
            if (counts[lane] == 0.0f) {
                counts[lane] = right_count;
                means[lane] = means[lane + stride];
                moments[lane] = moments[lane + stride];
            } else {
                float total = counts[lane] + right_count;
                float delta = means[lane + stride] - means[lane];
                moments[lane] += moments[lane + stride] + delta * delta * (counts[lane] * right_count / total);
                means[lane] += delta * (right_count / total);
                counts[lane] = total;
            }
        }
        __syncthreads();
    }
    *result_mean = means[0];
    *result_m2 = moments[0];
    __syncthreads();
}
'''


def _normalization_key(operation, stage, capability, device_index):
    if operation not in ('rms_norm', 'layer_norm') or stage not in ('forward', 'input_backward', 'affine_backward'):
        raise ValueError('Unsupported CUDA normalization kernel.')
    layer = operation == 'layer_norm'
    name = f'namitorch_{operation}_{stage}_float32'
    centered = '(input[base + column] - input[base]) - mean' if layer else 'input[base + column]'
    if stage == 'forward':
        statistics = '''
        float count = 0.0f, local_mean = 0.0f, local_m2 = 0.0f;
        float anchor = input[base];
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x) {
            float value = input[base + column] - anchor;
            count += 1.0f;
            float delta = value - local_mean;
            local_mean += delta / count;
            local_m2 += delta * (value - local_mean);
        }
        float mean, moment;
        norm_welford(count, local_mean, local_m2, shared, &mean, &moment);
        float inverse = rsqrtf(moment / static_cast<float>(width) + eps);
''' if layer else '''
        float local_sum = 0.0f;
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x) {
            float value = input[base + column];
            local_sum += value * value;
        }
        float mean = 0.0f;
        float inverse = rsqrtf(norm_sum(local_sum, shared) / static_cast<float>(width) + eps);
'''
        source = _REDUCTIONS + f'''
extern "C" __global__ void {name}(const float* input, const float* weight, const float* bias,
    float* output, float* statistics, unsigned long long rows, unsigned long long width, float eps) {{
    extern __shared__ float shared[];
    for (unsigned long long row = blockIdx.x; row < rows; row += gridDim.x) {{
        unsigned long long base = row * width;
        {statistics}
        if (threadIdx.x == 0 && statistics != nullptr) {{ statistics[2 * row] = mean; statistics[2 * row + 1] = inverse; }}
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x) {{
            float normalized = ({centered}) * inverse;
            float value = weight == nullptr ? normalized : normalized * weight[column];
            output[base + column] = bias == nullptr ? value : value + bias[column];
        }}
        __syncthreads();
    }}
}}
'''
    elif stage == 'input_backward':
        mean_gradient = 'float mean_gradient = norm_sum(local_gradient, shared) / static_cast<float>(width);' if layer else 'float mean_gradient = 0.0f;'
        source = _REDUCTIONS + f'''
extern "C" __global__ void {name}(const float* input, const float* weight, const float* gradient,
    const float* statistics, float* output, unsigned long long rows, unsigned long long width) {{
    extern __shared__ float shared[];
    for (unsigned long long row = blockIdx.x; row < rows; row += gridDim.x) {{
        unsigned long long base = row * width;
        float mean = statistics[2 * row], inverse = statistics[2 * row + 1];
        float local_gradient = 0.0f, local_product = 0.0f;
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x) {{
            float normalized = ({centered}) * inverse;
            float weighted = gradient[base + column] * (weight == nullptr ? 1.0f : weight[column]);
            local_gradient += weighted;
            local_product += weighted * normalized;
        }}
        {mean_gradient}
        float mean_product = norm_sum(local_product, shared) / static_cast<float>(width);
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x) {{
            float normalized = ({centered}) * inverse;
            float weighted = gradient[base + column] * (weight == nullptr ? 1.0f : weight[column]);
            output[base + column] = inverse * (weighted - mean_gradient - normalized * mean_product);
        }}
        __syncthreads();
    }}
}}
'''
    else:
        source = f'''
extern "C" __global__ void {name}(const float* input, const float* gradient, const float* statistics,
    float* weight_partial, float* bias_partial, unsigned long long rows, unsigned long long width,
    unsigned long long row_tiles, unsigned long long column_tiles, unsigned long long tile_rows) {{
    for (unsigned long long tile = blockIdx.x; tile < row_tiles * column_tiles; tile += gridDim.x) {{
        unsigned long long row_tile = tile / column_tiles;
        unsigned long long column = (tile % column_tiles) * blockDim.x + threadIdx.x;
        if (column < width) {{
            float weight_sum = 0.0f, bias_sum = 0.0f;
            unsigned long long end = (row_tile + 1) * tile_rows;
            if (end > rows) end = rows;
            for (unsigned long long row = row_tile * tile_rows; row < end; ++row) {{
                unsigned long long base = row * width;
                float upstream = gradient[base + column];
                if (weight_partial != nullptr) {{
                    float mean = statistics[2 * row], inverse = statistics[2 * row + 1];
                    weight_sum += upstream * (({centered}) * inverse);
                }}
                if (bias_partial != nullptr) bias_sum += upstream;
            }}
            unsigned long long destination = row_tile * width + column;
            if (weight_partial != nullptr) weight_partial[destination] = weight_sum;
            if (bias_partial != nullptr) bias_partial[destination] = bias_sum;
        }}
    }}
}}
'''
    return _KernelKey(source, name, np.dtype('float32').str, str(capability), _OPTIONS, device_index)


class NormalizationCUDAKernels(CUDAKernels):
    def _array_supported(self, array):
        return isinstance(array, self._backend.array_type) and array.device.id == self._backend.device.index and array.dtype == np.dtype('float32') and array.flags.c_contiguous and array.data.ptr % 4 == 0

    def supports(self, operation, arrays):
        backward = operation.endswith('_backward')
        name = operation.removesuffix('_backward')
        if name not in ('rms_norm', 'layer_norm') or len(arrays) != (4 if backward else 3):
            return False
        value, weight = arrays[:2]
        if not self._array_supported(value) or value.ndim < 1 or not 1 <= value.shape[-1] <= _MAX_WIDTH:
            return False
        if weight is not None and (not self._array_supported(weight) or weight.shape != (value.shape[-1],)):
            return False
        if backward:
            gradient, statistics = arrays[2:]
            return self._array_supported(gradient) and gradient.shape == value.shape and self._array_supported(statistics) and statistics.shape == (value.size // value.shape[-1], 2)
        bias = arrays[2]
        return bias is None or (name == 'layer_norm' and self._array_supported(bias) and bias.shape == (value.shape[-1],))

    def _kernel(self, name, stage):
        with self._lock:
            self._configure_device()
            return self._compile_key(_normalization_key(name, stage, self._capability, self._backend.device.index))

    def _launch(self, kernel, grid, block, arguments, shared_mem=0):
        self._backend.execution.hold(kernel)
        kernel((min(grid, self._max_grid_size),), (block,), arguments, shared_mem=shared_mem, stream=self._backend.module.cuda.get_current_stream())

    def run(self, operation, arrays, *, eps=None, save_stats=True, needs=(True, True, True)):
        if not self.supports(operation, arrays):
            return None
        backward = operation.endswith('_backward')
        name = operation.removesuffix('_backward')
        if backward:
            if not isinstance(needs, tuple) or len(needs) != 3 or any(type(flag) is not bool for flag in needs):
                raise TypeError('Normalization gradient requirements must be three Python bools.')
            return self._backward(name, arrays, needs)
        if type(save_stats) is not bool:
            raise TypeError('save_stats must be a Python bool.')
        if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, (float, int, np.floating)) or not math.isfinite(eps) or not 0 < eps <= float(np.finfo(np.float32).max):
            raise ValueError('Normalization eps must be positive and finite in float32.')
        with np.errstate(under='ignore'):
            epsilon = np.float32(eps)
        if epsilon == 0:
            raise ValueError('Normalization eps must remain positive in float32.')
        value, weight, bias = arrays
        rows, width = value.size // value.shape[-1], value.shape[-1]
        backend = self._backend
        with backend.context(arrays):
            output = backend.module.empty(value.shape, dtype=value.dtype)
            statistics = backend.module.empty((rows, 2), dtype=value.dtype) if save_stats else None
            if rows:
                kernel = self._kernel(name, 'forward')
                block = 1 << (min(self._block_size, width).bit_length() - 1)
                self._launch(kernel, rows, block, (value, weight, bias, output, statistics, np.uint64(rows), np.uint64(width), epsilon), block * 4 * (3 if name == 'layer_norm' else 1))
            return backend.result((output, statistics))

    def _backward(self, name, arrays, needs):
        value, weight, gradient, statistics = arrays
        rows, width = value.size // value.shape[-1], value.shape[-1]
        backend = self._backend
        with backend.context(arrays):
            dx = backend.module.empty(value.shape, dtype=value.dtype) if needs[0] else None
            dw = db = None
            if not rows:
                dw = backend.module.zeros((width,), dtype=value.dtype) if needs[1] else None
                db = backend.module.zeros((width,), dtype=value.dtype) if needs[2] else None
                return backend.result((dx, dw, db))
            if needs[0]:
                kernel = self._kernel(name, 'input_backward')
                block = 1 << (min(self._block_size, width).bit_length() - 1)
                self._launch(kernel, rows, block, (value, weight, gradient, statistics, dx, np.uint64(rows), np.uint64(width)), block * 4)
            if needs[1] or needs[2]:
                kernel = self._kernel(name, 'affine_backward')
                block = self._block_size
                row_tiles = (rows + _ROWS_PER_TILE - 1) // _ROWS_PER_TILE
                column_tiles = (width + block - 1) // block
                partial_shape = (row_tiles, width)
                partial_weight = backend.module.empty(partial_shape, dtype=value.dtype) if needs[1] else None
                partial_bias = backend.module.empty(partial_shape, dtype=value.dtype) if needs[2] else None
                self._launch(kernel, row_tiles * column_tiles, block, (value, gradient, statistics, partial_weight, partial_bias, np.uint64(rows), np.uint64(width), np.uint64(row_tiles), np.uint64(column_tiles), np.uint64(_ROWS_PER_TILE)))
                if needs[1]:
                    dw = backend.module.sum(partial_weight, axis=0, dtype=value.dtype)
                if needs[2]:
                    db = backend.module.sum(partial_bias, axis=0, dtype=value.dtype)
            return backend.result((dx, dw, db))
