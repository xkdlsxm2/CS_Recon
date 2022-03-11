import pathlib
import numpy as np
import pickle
import torch
from typing import Tuple
import cupy as cp
import matplotlib.pyplot as plt
import torch.fft
import matplotlib.colors as colors
#from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_files(directory: pathlib.Path):
    files = pickle.load(open(directory, 'rb'))
    keys = list(files.keys())

    return np.array(files[keys[-1]])

def kdata_torch2numpy(kdata_torch: torch.Tensor):
    kdata_np = kdata_torch.detach().cpu().numpy()

    return (kdata_np[..., 0] + 1j * kdata_np[..., 1])


def center_crop(data: np, shape: Tuple[int, int]) -> np:
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def save_sens_map(fname, sens_map, sub_folder):

    sens_map_pkl_path = pathlib.Path(f"Results/{sub_folder}/Sens_maps")
    sens_map_pkl_path.mkdir(exist_ok=True, parents=True)
    sens_map_pkl_path = sens_map_pkl_path / f"{fname}_sens_map.pkl"

    pickle.dump(sens_map, open(sens_map_pkl_path, 'wb'))


def save_recon(fname, CG_SENSE, ground_truth, sub_folder, recon):
    recon_pkl_path = pathlib.Path(f"{sub_folder}/Recon/pkl")
    recon_png_path = pathlib.Path(f"{sub_folder}/Recon/png")
    recon_pkl_path.mkdir(exist_ok=True, parents=True)
    recon_png_path.mkdir(exist_ok=True, parents=True)

    pkl_path = recon_pkl_path / f"{fname}_{recon}.pkl"
    png_path = recon_png_path / f"{fname}_{recon}.png"
    pickle.dump(CG_SENSE, open(pkl_path, 'wb'))
    imsave(CG_SENSE, png_path)

    GT_path = pathlib.Path(f"Recon/{sub_folder}/GT")
    GT_path.mkdir(exist_ok=True, parents=True)
    gt_path = GT_path / f"{fname}_GT.pkl"
    pickle.dump(ground_truth['ground_truth'], open(gt_path, 'wb'))


def imsave(obj, path):
    obj = cp.asnumpy(obj)
    f, a = plt.subplots(1, 1)
    a.imshow(abs(obj), cmap='gray')
    a.axis('off')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(28, 14)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data

def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def fftshift(x: torch.Tensor, dim=None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim=None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def roll(
        x: torch.Tensor,
        shift,
        dim,
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def imshow1row(imgs, titles=None, isMag=True, filename=None, log=False, suptitle=None, norm=1):
    f, a = plt.subplots(1, len(imgs))
    titles = [None] * len(imgs) if titles is None else titles

    for i, (img, title) in enumerate(zip(imgs, titles)):
        if torch.is_tensor(img):
            img = img[0, 0, :, :, 0] + img[0, 0, :, :, 1] * 1j
            img = img.cpu().numpy()
        ax = a[i] if len(imgs) >= 2 else a
        img = abs(img) if isMag else img
        img = np.log(img) if log else img
        ax.imshow(img, cmap='gray', norm=colors.PowerNorm(gamma=norm))
        ax.axis('off')
        ax.set_title(title)

    f.suptitle(suptitle) if suptitle is not None else f.suptitle("")
    if filename is None:
        plt.show()
    elif filename is not None:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(28, 14)
        plt.savefig(filename, bbox_inches='tight')
    plt.close(f)

'''
def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval=None) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]
'''