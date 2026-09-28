import numpy as np

from .kernels import CUDAKernels, _KernelKey, _OPTIONS


_MAX_ROW_WIDTH = 4096
_OPERATIONS = frozenset(('softmax', 'masked_softmax', 'causal_softmax', 'softmax_backward', 'masked_softmax_backward', 'causal_softmax_backward'))
_REDUCTIONS = '''
__device__ float merge_max(float a, float b) {
    return isnan(a) ? a : (isnan(b) ? b : fmaxf(a, b));
}
__device__ float reduce_max(float value, float* shared) {
    unsigned int lane = threadIdx.x;
    shared[lane] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) shared[lane] = merge_max(shared[lane], shared[lane + stride]);
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}
__device__ float reduce_sum(float value, float* shared) {
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
'''


def _softmax_key(operation, capability, device_index):
    if operation not in _OPERATIONS:
        raise ValueError(f'Unsupported CUDA row softmax kernel: {operation!r}.')
    name = 'namitorch_row_' + operation + '_float32'
    causal = 'column <= offset + row % queries' if operation.startswith('causal') else 'true'
    allowed = f'({causal}) && (mask == nullptr || mask[mask_rows == 0 ? 0 : (row % mask_rows) * width + column])'
    if operation.endswith('_backward'):
        body = f'''
        float local_sum = 0.0f;
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x)
            local_sum += gradient[base + column] * input[base + column];
        float dot = reduce_sum(local_sum, shared);
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x)
            output[base + column] = {allowed} ? input[base + column] * (gradient[base + column] - dot) : 0.0f;
'''
    else:
        body = f'''
        float local_max = -__int_as_float(0x7f800000);
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x) {{
            float value = {allowed} ? input[base + column] : -__int_as_float(0x7f800000);
            local_max = merge_max(local_max, value);
        }}
        float maximum = reduce_max(local_max, shared);
        bool empty = isinf(maximum) && maximum < 0.0f;
        bool invalid = isnan(maximum) || (isinf(maximum) && maximum > 0.0f);
        if (threadIdx.x == 0 && status != nullptr) status[row] = empty ? 1 : (invalid ? 2 : 0);
        float local_sum = 0.0f;
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x) {{
            float value = {allowed} ? input[base + column] : -__int_as_float(0x7f800000);
            float exponent = empty ? 0.0f : (invalid ? __int_as_float(0x7fffffff) : expf(value - maximum));
            output[base + column] = exponent;
            local_sum += exponent;
        }}
        float denominator = reduce_sum(local_sum, shared);
        for (unsigned long long column = threadIdx.x; column < width; column += blockDim.x)
            output[base + column] = empty ? 0.0f : output[base + column] / denominator;
'''
    source = _REDUCTIONS + f'''
extern "C" __global__ void {name}(
    const float* input, const float* gradient, const bool* mask, float* output, int* status,
    unsigned long long rows, unsigned long long width, unsigned long long mask_rows,
    unsigned long long queries, unsigned long long offset) {{
    extern __shared__ float shared[];
    for (unsigned long long row = blockIdx.x; row < rows; row += gridDim.x) {{
        unsigned long long base = row * width;
        {body}
        __syncthreads();
    }}
}}
'''
    return _KernelKey(source, name, np.dtype('float32').str, str(capability), _OPTIONS, device_index)


def _mask_rows(mask_shape, shape):
    if len(mask_shape) > len(shape):
        return None
    padded = (1,) * (len(shape) - len(mask_shape)) + mask_shape
    if any(left != 1 and left != right for left, right in zip(padded, shape)):
        return None
    if all(size == 1 for size in padded):
        return 0
    if padded[-1] != shape[-1]:
        return None
    first = next(index for index, size in enumerate(padded) if size != 1)
    if padded[first:] != shape[first:]:
        return None
    rows = 1
    for size in padded[:-1]:
        rows *= size
    return rows


class SoftmaxCUDAKernels(CUDAKernels):
    def supports(self, operation, arrays):
        if operation not in _OPERATIONS or not arrays:
            return False
        backward = operation.endswith('_backward')
        masked = operation.startswith(('masked', 'causal'))
        if len(arrays) != 1 + int(backward) + int(masked):
            return False
        value = arrays[0]
        for array in arrays[:2 if backward else 1]:
            if not isinstance(array, self._backend.array_type) or array.device.id != self._backend.device.index:
                return False
            if array.dtype != np.dtype('float32') or not array.flags.c_contiguous or array.data.ptr % 4:
                return False
            if array.shape != value.shape:
                return False
        if value.ndim == 0 or value.shape[-1] > _MAX_ROW_WIDTH:
            return False
        if operation.startswith('causal') and value.ndim < 2:
            return False
        if masked:
            mask = arrays[-1]
            if mask is None:
                return operation.startswith('causal')
            if not isinstance(mask, self._backend.array_type) or mask.device.id != self._backend.device.index:
                return False
            if mask.dtype != np.dtype('bool') or not mask.flags.c_contiguous or _mask_rows(mask.shape, value.shape) is None:
                return False
        return True

    def run(self, operation, arrays, *, query_position_offset=0, require_nonempty=False):
        if not self.supports(operation, arrays):
            return None
        if type(query_position_offset) is not int or query_position_offset < 0:
            raise ValueError('query_position_offset must be a nonnegative Python integer.')
        if type(require_nonempty) is not bool:
            raise TypeError('require_nonempty must be a Python bool.')
        backend = self._backend
        value = arrays[0]
        backward = operation.endswith('_backward')
        masked = operation.startswith(('masked', 'causal'))
        mask = arrays[-1] if masked else None
        width = value.shape[-1]
        rows = value.size // width if width else 0
        with backend.context(arrays):
            output = backend.module.empty(value.shape, dtype=value.dtype)
            if not rows:
                return backend.result(output)
            with self._lock:
                self._configure_device()
                kernel = self._compile_key(_softmax_key(operation, self._capability, backend.device.index))
                maximum_block = min(self._block_size, 256)
                block = 1 << (min(maximum_block, max(1, width)).bit_length() - 1)
            status = backend.module.empty((rows,), dtype=np.int32) if require_nonempty and not backward else None
            arguments = (
                value, arrays[1] if backward else None, mask, output, status,
                np.uint64(rows), np.uint64(width), np.uint64(0 if mask is None else _mask_rows(mask.shape, value.shape)),
                np.uint64(max(1, value.shape[-2]) if value.ndim > 1 else 1), np.uint64(min(query_position_offset, width)),
            )
            backend.execution.hold(kernel)
            kernel((min(rows, self._max_grid_size),), (block,), arguments, shared_mem=block * 4, stream=backend.module.cuda.get_current_stream())
            if status is not None and bool(backend.module.any(status != 0).item()):
                if bool(backend.module.any(status == 2).item()):
                    raise ValueError('Masked attention scores must be finite or -inf.')
                row = int(backend.module.argmax(status != 0).item())
                coordinate = tuple(int(index) for index in np.unravel_index(row, value.shape[:-1]))
                raise ValueError(f'Attention row {coordinate} has no allowed keys.')
            return backend.result(output)
