import math
import torch


# Keep atan2/sin/cos in fp32 under AMP: in fp16 the phase wrap at ±pi can underflow
# sin/cos enough to make atan2 output ±pi exactly, which in turn produces 0 gradient
# across a non-trivial neighborhood of inputs. Matches the fp32 policy in so3.py.
def _fp32(op):
    def wrapper(*args, **kwargs):
        src_dtype = None
        device_type = 'cpu'
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                src_dtype = a.dtype
                device_type = 'cuda' if a.is_cuda else 'cpu'
                break
        with torch.autocast(device_type=device_type, enabled=False):
            f32_args = tuple(a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args)
            f32_kwargs = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in kwargs.items()}
            out = op(*f32_args, **f32_kwargs)
        if src_dtype is not None and src_dtype != torch.float32 and isinstance(out, torch.Tensor):
            return out.to(src_dtype)
        return out
    wrapper.__name__ = getattr(op, '__name__', 'wrapped')
    return wrapper


@_fp32
def tor_expmap(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (x + u) % (2 * math.pi)


@_fp32
def tor_logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # shortest signed difference on circle
    return torch.atan2(torch.sin(y - x), torch.cos(y - x))


def tor_random_uniform(*size, dtype=None, device=None) -> torch.Tensor:
    z = torch.rand(*size, dtype=dtype, device=device)
    return z * 2 * math.pi


@_fp32
def tor_geodesic_t(t: torch.Tensor, angles_1: torch.Tensor, angles_0: torch.Tensor) -> torch.Tensor:
    tangent_vec = t * tor_logmap(angles_0, angles_1)
    return tor_expmap(angles_0, tangent_vec)
