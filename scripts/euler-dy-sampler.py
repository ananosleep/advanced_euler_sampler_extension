import torch
import tqdm
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers
from tqdm.auto import trange, tqdm
from k_diffusion import utils
from k_diffusion.sampling import to_d
import math


NAME = 'Euler_Dy'
ALIAS = 'euler_dy'



@torch.no_grad()
def dy_sampling_step(x, model, dt, sigma_hat, **extra_args):

    original_shape = x.shape
    batch_size, m, n = original_shape[0], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, 4, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, 4, m, n)

    denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **extra_args)
    d = to_d(c, sigma_hat, denoised)
    c = c + d * dt

    d_list = c.view(batch_size, 4, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = a_list.view(batch_size, 4, m, n, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(batch_size, 4, 2 * m, 2 * n)

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, :2 * m, :2 * n] = x
        if extra_row:
            x_expanded[:, :, -1:, :2 * n + 1] = extra_row_content
        if extra_col:
            x_expanded[:, :, :2 * m, -1:] = extra_col_content
        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]
        x = x_expanded

    return x


@torch.no_grad()
def sample_euler_dy(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                               s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        # print(i)
        # i第一步为0
        gamma = max(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        # print(sigma_hat)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if sigmas[i + 1] > 0:
            if i // 2 == 1:
                x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        # Euler method
        x = x + d * dt
    return x



if not NAME in [x.name for x in sd_samplers.all_samplers]:
    euler_max_samplers = [(NAME, sample_euler_dy, [ALIAS], {})]
    samplers_data_euler_max_samplers = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in euler_max_samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers.all_samplers += samplers_data_euler_max_samplers
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
