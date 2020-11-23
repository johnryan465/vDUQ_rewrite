from typing import Optional, TypeVar
import torch
from torch.nn.functional import normalize
from torch.nn.utils.spectral_norm import SpectralNorm, SpectralNormLoadStateDictPreHook, SpectralNormStateDictHook
from torch import nn

T_module = TypeVar('T_module', bound=nn.Module)

class SoftSpectralNorm(SpectralNorm):
    # To use soft constraints instead of hard constraints we can't
    # directly use the inbuilt spectral norm from pytorch
    # So we subclass the spectral norm class and override the required functions.

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12, coeff : float = 0.9) -> None:
        super(SoftSpectralNorm, self).__init__(name, n_power_iterations, dim, eps)
        self.coeff = coeff

    # overriding parent class function
    def compute_weight(self, module: T_module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.mv(weight_mat.t(), u),
                                  dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v),
                                  dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # scaling with the coefficent
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        
        return weight

    @staticmethod
    def apply(module: T_module, name: str, n_power_iterations: int, dim: int, eps: float, coeffs : float) -> 'SoftSpectralNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SoftSpectralNorm(name, n_power_iterations, dim, eps, coeffs)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(
            SpectralNormLoadStateDictPreHook(fn))
        return fn




def soft_spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None,
                  coeff : float = 0.9) -> T_module:
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SoftSpectralNorm.apply(module, name, n_power_iterations, dim, eps, coeff)
    return module
