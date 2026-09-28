_fused_kernels_enabled = True


def enable_fused_kernels(mode):
    global _fused_kernels_enabled
    if type(mode) is not bool:
        raise TypeError("Fused kernel mode must be a Python bool.")
    _fused_kernels_enabled = mode


def fused_kernels_enabled():
    return _fused_kernels_enabled
