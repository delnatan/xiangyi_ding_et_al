"""
Following Weisong Zhao's paper on two-step sparsity-enhanced deconvolution.
This module provides the parts and function to solve the following problem:

I wrote this assuming that the machine has PyTorch setup with CUDA for GPU usage

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from wavelet_bg_est import atrous_filters
import tifffile

GPU = torch.device("cuda")


def torch_estimate_background_idwt(img, max_k=6, niter=100, tol=1e-2):
    psi = atrous_filters(img.shape, levels=max_k)
    psi = torch.from_numpy(psi.astype(np.complex64)).to(GPU)
    estbg = torch.from_numpy(img.astype(np.float32)).to(GPU)

    for n in range(niter):
        ft_img = torch.fft.rfft2(estbg)
        # compute wavelet coefficients
        c = torch.fft.irfft2(ft_img[None, :, :] * psi, s=img.shape)
        clipmask = estbg > c[-1]
        prev = 1.0 * estbg
        estbg = clipmask * c[-1] + ~clipmask * estbg
        dx = torch.norm(estbg - prev) / torch.norm(prev)

        if dx < tol:
            break

    ft_est = torch.fft.rfft2(estbg)
    estbg = torch.fft.irfft2(ft_est * psi[-1], s=img.shape)
    return estbg.cpu().numpy()


def compute_2nd_order_filters_2d(n):
    i2pi = 2.0 * np.pi * 1j
    fy = np.fft.fftfreq(n[0])
    fx = np.fft.rfftfreq(n[1])
    ky, kx = np.meshgrid(fy, fx, indexing="ij")

    Dxx = np.exp(-i2pi * kx) - 2 + np.exp(i2pi * kx)
    Dyy = np.exp(-i2pi * ky) - 2 + np.exp(i2pi * ky)
    Dyx = (
        1 - np.exp(-i2pi * ky) - np.exp(-i2pi * kx) + np.exp(-i2pi * (ky + kx))
    )
    Dxy = (
        1 - np.exp(-i2pi * kx) - np.exp(-i2pi * ky) + np.exp(-i2pi * (ky + kx))
    )

    # compute Gram matrix D^t.D
    DtD = np.conj(Dxx) * Dxx + np.conj(Dyy) * Dyy + 2 * np.conj(Dyx) * Dyx

    return [Dyy, Dxx, Dyx, Dxy], DtD


def shrink(z, alpha):
    if z.ndim == 2:
        return torch.sign(z) * torch.maximum(
            torch.abs(z) - alpha, torch.tensor(0.0)
        )
    else:
        out = torch.zeros_like(z)
        for i in range(z.shape[0]):
            out[i] = torch.sign(z[i]) * torch.maximum(
                torch.abs(z[i]) - alpha, torch.tensor(0.0)
            )
        return out


def shrink2(z, alpha):
    # proximal operator to non-negative L1-norm
    return torch.maximum(torch.tensor(0), z - alpha)


def block_shrink(z, alpha):
    d = torch.norm(z)
    out = torch.zeros_like(z)
    for i in range(z.shape[0]):
        out[i] = torch.maximum(1 - (z[i] / d), torch.tensor(0.0)) * z[i]
    return out


def sparsify_denoise2(
    img,
    lam1,
    lam2,
    rho=10,
    releps=1e-3,
    max_iter=100,
    pad=20,
):
    padshape = tuple([s + pad * 2 for s in img.shape])
    device = img.device
    dfilters, HtH = compute_2nd_order_filters_2d(padshape)

    # convert to pytorch tensors
    dfilters = [
        torch.from_numpy(f.astype(np.complex64)).to(device) for f in dfilters
    ]
    HtH = torch.from_numpy(HtH.astype(np.complex64)).to(device)

    # 1 - hessian (Hx, continuity)
    u1 = torch.zeros((len(dfilters),) + padshape).to(device)
    z1 = torch.zeros((len(dfilters),) + padshape).to(device)

    # 2 - sparsity (x)
    u2 = torch.zeros(padshape).to(device)
    z2 = torch.zeros(padshape).to(device)

    x = torch.zeros(padshape).to(device)
    b = torch.zeros(padshape).to(device)

    b[pad:-pad, pad:-pad] = img

    mask_in = torch.zeros(padshape, dtype=bool).to(device)
    if pad != 0:
        inside_slices = (slice(pad, -pad), slice(pad, -pad))
    else:
        inside_slices = (slice(None, None), slice(None, None))
    mask_in[inside_slices] = True
    mask_out = ~mask_in
    lastreltol = float("inf")

    for k in range(max_iter):
        # x-update
        # use work variables (in Fourier space)
        denominator = 1.0 + rho * HtH
        wrk1 = torch.fft.rfft2(z1 - u1)
        wrk2 = torch.fft.rfft2(z2 - u2)

        # H' * (z1 - u1)
        adj1 = sum([torch.conj(f) * wrk1[i] for i, f in enumerate(dfilters)])

        numerator = torch.fft.rfft2(b) + rho * adj1 + rho * wrk2

        x_old = torch.clone(x)
        x = torch.fft.irfft2(numerator / denominator)
        x = torch.maximum(x, torch.tensor(0.0))

        tol = torch.norm(x - x_old)

        reltol = tol / max(torch.norm(x_old), 1e-7)

        if reltol > lastreltol:
            break

        iterstr = "\riter = {iter:3d}, |dx| = {reltol:.3E}"

        print(iterstr.format(iter=k, reltol=reltol), end="")

        if reltol < releps:
            break

        # z1 - update, continuity
        ft_x = torch.fft.rfft2(x)
        Hx = torch.stack([torch.fft.irfft2(f * ft_x) for f in dfilters])

        z1 = mask_in * shrink(Hx + u1, lam1 / rho) + mask_out * (Hx + u1)

        # z2 - update, sparsity
        z2 = mask_in * shrink(x + u2, lam2 / rho) + mask_out * (x + u2)

        u1 = u1 + (Hx - z1)
        u2 = u2 + (x - z2)

    print("")

    return x[pad:-pad, pad:-pad].cpu().numpy()
