import numpy as np

from .kernels import CUDAKernels, _KernelKey, _OPTIONS, launch_dimensions


_SOURCE = '''
extern "C" __global__ void namitorch_adamw_float32(
    float* parameter, const float* gradient, float* first, float* second,
    float lr, float beta1, float beta2, float complement1, float complement2,
    float eps, float weight_decay, float decay_factor, float correction1, float correction2,
    int maximize, int update_parameter, unsigned long long count) {
    unsigned long long stride = static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    for (unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        float grad = maximize ? -gradient[index] : gradient[index];
        float moment = beta1 * first[index] + complement1 * grad;
        float variance = beta2 * second[index] + (complement2 * grad) * grad;
        first[index] = moment;
        second[index] = variance;
        if (update_parameter) {
            float value = parameter[index];
            if (weight_decay != 0.0f) value *= decay_factor;
            float corrected_moment = moment / correction1;
            float denominator = sqrtf(variance / correction2) + eps;
            parameter[index] = value - lr * (corrected_moment / denominator);
        }
    }
}
'''


class AdamWCUDAKernel(CUDAKernels):
    def supports(self, parameter, gradient, first=None, second=None):
        arrays = (parameter, gradient, first, second)
        if parameter is None or gradient is None:
            return False
        present = tuple(array for array in arrays if array is not None)
        for array in present:
            if not isinstance(array, self._backend.array_type):
                return False
            if array.device.id != self._backend.device.index or array.dtype != np.dtype('float32'):
                return False
            if not array.flags.c_contiguous or array.data.ptr % 4 or array.shape != parameter.shape:
                return False
        ranges = [(array.data.ptr, array.data.ptr + array.nbytes) for array in present if array.nbytes]
        return not any(max(start, other_start) < min(end, other_end) for index, (start, end) in enumerate(ranges) for other_start, other_end in ranges[index + 1:])

    def _kernel(self):
        with self._lock:
            self._configure_device()
            key = _KernelKey(_SOURCE, 'namitorch_adamw_float32', np.dtype('float32').str, str(self._capability), _OPTIONS + ('--fmad=false',), self._backend.device.index)
            return self._compile_key(key)

    def run(self, parameter, gradient, first, second, *, coefficients, maximize, update_parameter):
        if first is None or second is None or not self.supports(parameter, gradient, first, second):
            raise RuntimeError('Fused AdamW requires non-overlapping contiguous float32 CUDA arrays with identical shapes and device.')
        if len(coefficients) != 10 or any(not isinstance(value, np.float32) or not np.isfinite(value) for value in coefficients):
            raise ValueError('Fused AdamW requires ten finite float32 coefficients.')
        if coefficients[5] <= 0 or coefficients[8] <= 0 or coefficients[9] <= 0:
            raise ValueError('Fused AdamW eps and bias corrections must be positive.')
        if type(maximize) is not bool or type(update_parameter) is not bool:
            raise TypeError('Fused AdamW flags must be Python bools.')
        if parameter.size == 0:
            return False
        backend = self._backend
        with backend.context(parameter, gradient, first, second):
            kernel = self._kernel()
            grid, block = launch_dimensions(parameter.size, self._block_size, self._max_grid_size)
            backend.execution.hold(kernel)
            arguments = (parameter, gradient, first, second, *coefficients, np.int32(maximize), np.int32(update_parameter), np.uint64(parameter.size))
            kernel(grid, block, arguments, stream=backend.module.cuda.get_current_stream())
            backend.result((parameter, first, second))
        return True
