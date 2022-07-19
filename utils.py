import pickle, plot, h5py
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from cs import CS
from grappa import GRAPPA


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    data = cp.asnumpy(data)
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def save_sens_maps(sens_maps, sub_folder):
    savePath = sub_folder.parent / 'SensMaps'
    savePath.mkdir(exist_ok=True, parents=True)
    h5f = h5py.File(savePath / f"{sub_folder.stem}_sens_maps.h5", 'w')
    h5f.create_dataset('sens_maps', data=sens_maps)
    h5f.close()

    recon_png_path = sub_folder.parent / f'pngs' / sub_folder.stem / "Sens_maps"
    recon_png_path.mkdir(exist_ok=True, parents=True)

    for i, sens_map in enumerate(sens_maps):
        png_path = recon_png_path / f"{sub_folder.stem}_{i}"
        sens_map = sens_map[..., 0] + sens_map[..., 1] * 1j
        sens_map = sens_map.detach().cpu().numpy()
        plot.ImagePlot(sens_map, z=0, save_path=png_path, save_basename='png', hide_axes=True)


def save_result(fname, result, sub_folder, recon):
    recon_npy_path = sub_folder.parent / f'npys' / sub_folder.stem
    recon_png_path = sub_folder.parent / f'pngs' / sub_folder.stem / recon
    recon_npy_path.mkdir(exist_ok=True, parents=True)
    recon_png_path.mkdir(exist_ok=True, parents=True)

    npy_path = recon_npy_path / f"{fname}_{recon}.npy"
    png_path = recon_png_path / f"{fname}_{recon}"

    _, h, w = result.shape
    result = cp.rot90(result[:, :, w // 4 * 1:w // 4 * 3 + 30], axes=(1,2))
    with open(npy_path, 'wb') as f:
        np.save(f, result)
    for i, img in enumerate(result):
        png_path = png_path.parent / (png_path.stem+f"_{i}")
        imsave(img, png_path)


def imsave(obj, path):
    obj = cp.asnumpy(obj)
    f, a = plt.subplots(1, 1)
    a.imshow(abs(obj), cmap='gray', vmin=0, vmax=1)
    a.axis('off')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(28, 14)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm="ortho"
        )
    )
    data = fftshift(data, dim=(-3, -2))
    return data


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def create_mask(kspace):
    """

    :param kspace: shape of (slice, coil, readout, PE, dim)
    :return: mask => shape of (1,1,PE,1)
    """
    mask = kspace[0, 0, 0, :, 0].numpy().astype(bool)[None, None, :, None]
    return to_tensor(mask)


def choose_method(args):
    if args.method == "cs":
        return CS
    elif args.method == "grappa":
        return GRAPPA
    else:
        raise "Method should be either 'cs' or 'grappa'!"


def undersample(kspace, rate=3):
    '''

    :param kspace: kspace to be undersampled
    :param rate: Rate of undersample (MRCP data is already PAT3, so default value for the rate is 3)
    :return:
        kspace_us: Undersampled kspace
    '''

    was_torch = False
    if torch.is_tensor(kspace):
        kspace = torch.view_as_complex(kspace)
        kspace = kspace.detach().cpu().numpy()
        was_torch = True

    mask = kspace[0, 0, 0, :].astype(bool)
    acs_start, acs_end = get_acs_index(mask)
    acs = kspace[:, :, :, acs_start:acs_end + 1]

    kspace_us = undersample_(kspace, rate)
    kspace_us[:, :, :, acs_start:acs_end + 1] = acs

    if was_torch:
        kspace_us = to_tensor(kspace_us)

    return kspace_us


def get_acs_index(mask):
    if torch.is_tensor(mask):
        mask = np.array(mask.cpu().detach())
    slices = np.ma.clump_masked(np.ma.masked_where(mask, mask))
    acs_ind = [(s.start, s.stop - 1) for s in slices if s.start < (s.stop - 1)]
    assert (acs_ind != [] or len(
        acs_ind) > 1), "Couldn't extract center lines mask from k-space - is there pat2 undersampling?"
    acs_start = acs_ind[0][0]
    acs_end = acs_ind[0][1]

    return acs_start, acs_end


def undersample_(kspace, rate):
    # Since mrcp is already PAT3, so just divide the rate by 3
    idx = np.where(kspace[0, 0, 0, :].astype(bool))[0][::rate//3]

    kspace_us = np.zeros_like(kspace)
    kspace_us[:, :, :, idx] = kspace[:, :, :, idx]

    return kspace_us
