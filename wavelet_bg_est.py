import numpy as np


def fourier_meshgrid(*N, d=1.0):
    """computes meshgrid of frequency spacing

    Note: only half coordinates are considered (for real-valued input)

    Args:
        N1,N2,N3,... : int
            Size of sample N for computing Fourier coordinate grids
        d1,d2,d3... : float, optional
            Real-space spacing for each corresponding sample size N1,N2,...
            default is 1.0

    """

    Ndim = len(N)
    dimrange = range(Ndim)

    if d == 1.0:
        dvec = [1.0 for n in dimrange]

    else:
        dvec = d

    Nd = len(dvec)

    assert (
        Ndim == Nd
    ), "Number of spacing d ({:d}) must be equal to number of \
    sample sizes ({:d})".format(
        Nd, Ndim
    )

    # last dimension uses rfft by convention (so will have half the samples)

    spacings = []
    lastdim = dimrange[-1]

    for n in dimrange:
        if n == lastdim:
            spacings.append(np.fft.rfftfreq(N[n], d=dvec[n]))
        else:
            spacings.append(np.fft.fftfreq(N[n], d=dvec[n]))

    return np.meshgrid(*spacings, indexing="ij")


def atrous_filters(shape, levels=4):
    """compute discrete wavelet transform (Fourier) filters

    Note that the last filter is the for computing last smoothed image,
    required for reconstruction. Coefficients use cubic b-splines.

    F^(0) = F^(k_max) + sum(W^(k))

    k_max is the number of decompisition level + 1
    W are the wavelet coefficients
    F^(k_max) is the last blurred image

    Args:
        shape : list of ints
            Each integer is the shape of input dimensions

    Returns:
        Fourier filters with the shape (levels, dim[0], dim[1], ...)

    Example ::
        img = imread("roi_0014.png")
        x̃ = np.fft.rfft2(img.astype(float))
        ψ = atrous_filters(img.shape, levels=4)
        # use broadcasting to do the convolution
        coefs = np.fft.irfft2(ψ * x̃[None, :, :])

        # reconstruction is done by summing the coefficients
        recon = coefs.sum(axis=0)

    """
    kspace = fourier_meshgrid(*shape)

    fourier_shape = kspace[0].shape
    psi = np.zeros((levels + 1, *fourier_shape), dtype=np.complex128)
    twopi_i = 2.0 * np.pi * 1j

    wrk = np.ones(fourier_shape, dtype=np.complex128)
    for s in kspace:
        wrk *= (
            0.0625 * np.exp(-twopi_i * -2 * s)
            + 0.25 * np.exp(-twopi_i * -1 * s)
            + 0.375
            + 0.25 * np.exp(-twopi_i * 1 * s)
            + 0.0625 * np.exp(-twopi_i * 2 * s)
        )

    psi[0, ...] = wrk.copy()

    # compute the "holey" spline filters
    for k in range(levels):
        wrk = np.ones(fourier_shape, dtype=np.complex128)
        for s in kspace:
            wrk *= (
                0.0625 * np.exp(-twopi_i * -(2**k * 2 + 2) * s)
                + 0.25 * np.exp(-twopi_i * -(2**k + 1) * s)
                + 0.375
                + 0.25 * np.exp(-twopi_i * (2**k + 1) * s)
                + 0.0625 * np.exp(-twopi_i * (2**k * 2 + 2) * s)
            )
        psi[k + 1, ...] = wrk.copy()

    # form the wavelet filters
    WF = np.zeros((levels + 1, *fourier_shape), dtype=np.complex128)

    for k in range(levels):
        if k == 0:
            WF[k, ...] = 1.0 - psi[k]
        else:
            WF[k, ...] = psi[0:k, ...].prod(axis=0) - psi[0 : k + 1, ...].prod(
                axis=0
            )

    # last filter is a product of all coefficients (except the last one)
    WF[-1, ...] = psi[:-1].prod(axis=0)

    return WF


def estimate_background_idwt(img, max_k=5, niter=25, tol=1e-3):
    psi = atrous_filters(img.shape, levels=max_k)
    estbg = img.copy()

    for n in range(niter):
        ft_img = np.fft.rfft2(estbg)
        # compute wavelet coefficients
        c = np.fft.irfft2(ft_img[None, :, :] * psi, s=img.shape)
        clipmask = estbg > c[-1]
        prev = estbg.copy()
        estbg = clipmask * c[-1] + ~clipmask * estbg
        dx = np.linalg.norm(estbg - prev) / np.linalg.norm(prev)

        if dx < tol:
            break

    ft_est = np.fft.rfft2(estbg)
    estbg = np.fft.irfft2(ft_est * psi[-1], s=img.shape)
    return estbg
