from types import MappingProxyType

import numpy as np

from .kernels import CUDAKernels, _KernelKey, _OPTIONS, launch_dimensions


_HELPERS = '''
__device__ float stable_sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}
__device__ float silu_value(float x) {
    return isinf(x) && x < 0.0f ? 0.0f : x * stable_sigmoid(x);
}
__device__ float silu_derivative(float x) {
    float s = stable_sigmoid(x);
    return isfinite(x) ? s + x * (s * (1.0f - s)) : s;
}
__device__ float sigmoid_derivative(float x) {
    float s = stable_sigmoid(x);
    return s * (1.0f - s);
}
__device__ float gelu_value(float x) {
    if (fabsf(x) > 20.0f) return x > 0.0f ? x : -0.0f;
    float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
    return (0.5f * x) * (1.0f + t);
}
__device__ float gelu_derivative(float x) {
    if (fabsf(x) > 20.0f) return x > 0.0f ? 1.0f : 0.0f;
    float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
    float d = 0.7978845608028654f * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * d;
}
__device__ float relu_value(float x) {
    return x <= 0.0f ? 0.0f : x;
}
__device__ float relu_derivative(float x) {
    return isnan(x) ? x : (x > 0.0f ? 1.0f : 0.0f);
}
'''

_bodies = {
    'sigmoid': ('unary', 'output[index] = stable_sigmoid(a[index]);'),
    'silu': ('unary', 'output[index] = silu_value(a[index]);'),
    'gelu_tanh': ('unary', 'output[index] = gelu_value(a[index]);'),
    'sigmoid_backward': ('unary_backward', 'float y = a[index]; output[index] = b[index] * (y * (1.0f - y));'),
    'silu_backward': ('unary_backward', 'output[index] = b[index] * silu_derivative(a[index]);'),
    'gelu_tanh_backward': ('unary_backward', 'output[index] = b[index] * gelu_derivative(a[index]);'),
    'swiglu': ('pair', 'output[index] = silu_value(a[index]) * b[index];'),
    'swiglu_backward': ('pair_backward', 'output[index] = (c[index] * b[index]) * silu_derivative(a[index]); second[index] = c[index] * silu_value(a[index]);'),
    'clamp': ('clamp', '''float x = a[index];
        float lo = has_min ? b[0] : 0.0f;
        float hi = has_max ? c[0] : 0.0f;
        if (isnan(x) || isnan(lo) || isnan(hi)) output[index] = x + lo + hi;
        else { if (has_min && x <= lo) x = lo; if (has_max && x >= hi) x = hi; output[index] = x; }'''),
    'clamp_backward': ('clamp_backward', '''float x = a[index];
        float lo = has_min ? b[0] : 0.0f;
        float hi = has_max ? c[0] : 0.0f;
        if (isnan(x) || isnan(lo) || isnan(hi)) output[index] = x + lo + hi;
        else output[index] = (!has_min || x > lo) && (!has_max || x < hi) ? d[index] : 0.0f;'''),
}
for _name, _forward, _derivative in (
    ('sigmoid', 'stable_sigmoid', 'sigmoid_derivative'), ('silu', 'silu_value', 'silu_derivative'),
    ('gelu_tanh', 'gelu_value', 'gelu_derivative'), ('relu', 'relu_value', 'relu_derivative'),
):
    _bodies[f'bias_{_name}'] = ('bias', f'output[index] = {_forward}(a[index] + b[index % width]);')
    _bodies[f'bias_{_name}_backward'] = ('bias_backward', f'output[index] = c[index] * {_derivative}(a[index] + b[index % width]);')
_BODIES = MappingProxyType(_bodies)
del _bodies, _name, _forward, _derivative


def _fused_key(operation, capability, device_index):
    if operation not in _BODIES:
        raise ValueError(f"Unsupported fused CUDA operation: {operation!r}.")
    name = f'namitorch_fused_{operation}_float32'
    source = _HELPERS + (
        f'extern "C" __global__ void {name}('
        'const float* a, const float* b, const float* c, const float* d, '
        'float* output, float* second, unsigned long long count, unsigned long long width, int has_min, int has_max) {\n'
        '    const unsigned long long stride = static_cast<unsigned long long>(blockDim.x) * gridDim.x;\n'
        '    for (unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n'
        '         index < count; index += stride) {\n'
        f'        {_BODIES[operation][1]}\n'
        '    }\n}\n'
    )
    return _KernelKey(source, name, np.dtype('float32').str, str(capability), _OPTIONS, device_index)


class FusedCUDAKernels(CUDAKernels):
    def supports(self, operation, arrays):
        if operation not in _BODIES:
            return False
        layout = _BODIES[operation][0]
        arity = {'unary': 1, 'unary_backward': 2, 'pair': 2, 'pair_backward': 3, 'bias': 2, 'bias_backward': 3, 'clamp': 3, 'clamp_backward': 4}[layout]
        if len(arrays) != arity or arrays[0] is None:
            return False
        for position, array in enumerate(arrays):
            if array is None:
                if not layout.startswith('clamp') or position not in (1, 2):
                    return False
                continue
            if not isinstance(array, self._backend.array_type) or array.device.id != self._backend.device.index:
                return False
            if array.dtype != np.dtype('float32') or not array.flags.c_contiguous or array.data.ptr % 4:
                return False
        first = arrays[0]
        if layout.startswith('bias'):
            return first.ndim >= 1 and arrays[1].shape == (first.shape[-1],) and (len(arrays) == 2 or arrays[2].shape == first.shape)
        if layout.startswith('clamp'):
            return any(array is not None for array in arrays[1:3]) and all(array is None or array.ndim == 0 for array in arrays[1:3]) and (len(arrays) == 3 or arrays[3].shape == first.shape)
        return all(array.shape == first.shape for array in arrays[1:])

    def run(self, operation, arrays):
        if not self.supports(operation, arrays):
            return None
        backend = self._backend
        first = arrays[0]
        with backend.context(arrays):
            count = int(first.size)
            outputs = tuple(backend.module.empty(first.shape, dtype=first.dtype) for _ in range(2 if operation == 'swiglu_backward' else 1))
            if count:
                with self._lock:
                    self._configure_device()
                    kernel = self._compile_key(_fused_key(operation, self._capability, backend.device.index))
                pointers = [first if array is None else array for array in arrays]
                pointers.extend([first] * (4 - len(pointers)))
                clamp = operation.startswith('clamp')
                arguments = (*pointers, outputs[0], outputs[-1], np.uint64(count), np.uint64(first.shape[-1] if first.ndim else 1), np.int32(clamp and arrays[1] is not None), np.int32(clamp and arrays[2] is not None))
                grid, block = launch_dimensions(count, self._block_size, self._max_grid_size)
                backend.execution.hold(kernel)
                kernel(grid, block, arguments, stream=backend.module.cuda.get_current_stream())
            return backend.result(outputs if len(outputs) == 2 else outputs[0])
