import plot, h5py, os, pathlib
import cupy as cp
import numpy as np
import torch
from cs import CS
from grappa import GRAPPA
from argparse import ArgumentParser
from numpy.fft import fft2, fftshift


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


def save_result(fname, result, sub_folder, recon, rate=3, is_k=None):
    recon_npy_path = sub_folder.parent / f'npys' / sub_folder.stem
    recon_npy_path.mkdir(exist_ok=True, parents=True)

    if is_k is None:
        npy_path = recon_npy_path / f"{fname}_{recon}_PAT{rate}.npy"
        _, h, w = result.shape
        result = cp.rot90(result[:, :, w // 4 * 1:w // 4 * 3 + 30], axes=(1,2))
    else:
        npy_path = recon_npy_path / f"{fname}_{recon}_PAT{rate}_k.npy"

    with open(npy_path, 'wb') as f:
        np.save(f, result)


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
    data = ifftshift_(data, dim=(-3, -2))
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm="ortho"
        )
    )
    data = fftshift_(data, dim=(-3, -2))
    return data


def fftshift_(x, dim=None):
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


def ifftshift_(x, dim=None):
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


def choose_method(method):
    if method == "cs":
        return CS
    elif method == "grappa":
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



def build_args(config_json):
    parser = ArgumentParser()

    config = config_json['path']
    parser.add_argument(
        '--save_path',
        type=str,
        default=pathlib.Path(config["save_path"]),
        help='path to save reconstruction images and pickles')

    parser.add_argument(
        '--data_path',
        type=str,
        default=pathlib.Path(config["data_path"]),
        help='path to data')

    parser.add_argument(
        '--data_name',
        default=config["data_name"],
        help='data name to be reconstructed (None: all data in the path')

    config = config_json['recon']
    parser.add_argument(
        '--method',
        type=str,
        default=config["method"],
        help='Reconstruction method')

    parser.add_argument(
        '--rates',
        type=list,
        default=config["undersamping_rates"],
        help='Undersamping_rates')

    config = config_json['cs']
    parser.add_argument(
        '--ESPIRiT_threshold',
        type=float,
        default=config["ESPIRiT_threshold"],
        help='ESPIRiT_threshold')

    parser.add_argument(
        '--CS_lambda',
        type=float,
        default=config["CS_lambda"],
        help='CS_lambda')

    config = config_json['grappa']
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=config["kernel_size"],
        help='kernel_size for weigh estimation')

    args = parser.parse_args()
    data_path = args.data_path
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    slurm_job_id = "." if slurm_job_id == None else f"{slurm_job_id}.tinygpu"

    data_path = data_path.parent / slurm_job_id / data_path.name
    args.data_path = data_path

    return args

def save_recon_k_cs(recon_k, img, sens_map, smoothing_factor=8):
    img = cp.asnumpy(img)
    sens_map = cp.asnumpy(sens_map)
    coil_img = img[None, ...] * sens_map
    coil_k = apply_fft(coil_img[0])
    coil_k = cp.asnumpy(coil_k)
    coil_k *= np.expm1(smoothing_factor) / coil_k.max()
    coil_k = np.log1p(coil_k)
    coil_k /= coil_k.max()
    recon_k.append(coil_k)

def apply_fft(img):
    return fftshift(fft2(fftshift(img)))


def grappa_k4save(recon_k, smoothing_factor=8):
    recon_k = recon_k[..., 0] + 1j * recon_k[..., -1]
    recon_k = recon_k.cpu().numpy()[:, 0]
    recon_k *= np.expm1(smoothing_factor) / np.max(recon_k, axis=(1,2))[:, None, None]
    recon_k = np.log1p(recon_k)
    recon_k /= np.max(recon_k, axis=(1,2))[:, None, None]
    return recon_k