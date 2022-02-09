"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy

import numpy as np
import torch
from utils import *
from subsample import MaskFunc


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
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def gaussian_fn(M: int, std: float) -> numpy.ndarray:
    """
    Creating 1d gaussian distribution

    Args:
        M: Kernel length
        std: Standard deviation

    Returns:
        1d gaussian distribution
    """
    n = np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)
    return w


def reshape_mask(mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
    """Reshape mask to desired output shape."""
    num_rows, num_cols = shape[-3:-1]
    mask_shape = [1 for _ in shape]
    mask_shape[-3:-1] = [num_rows, num_cols]

    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))


def mask2D(kdata_torch: torch.Tensor, subsampling_mask: torch.Tensor):
    '''

    Args:
        kdata_torch: shape of (#coil x row x col x 2)
        subsampling_mask: shape of (1 x 1 x col x 1)

    Returns: mask: shape of (row x col)

    '''
    mask_shape = kdata_torch.shape[-3:-1]
    shape = (kdata_torch.shape[-3], 1)

    return subsampling_mask.reshape(-1, mask_shape[1]).repeat(shape).numpy()


def gaussian_mask(kspace: torch.Tensor, mask: torch.Tensor, std: float, small_acs_block: tuple,
                  acq_start: int, acq_end: int):
    """
    Returns a 2D Gaussian mas array.
    Args:
        kspace: Masked kspace
        mask: Subsampling mask
        std: Standard deviation
        small_acs_block: A block for low-freq
        acq_start: A indice for acquisition start
        acq_end: A indice for acquisition end

    Returns:
        2d gaussian kernel map
    """
    mask = mask2D(kspace, mask)
    nrow, ncol = mask[mask != 0].reshape(mask.shape[0], -1).shape
    gkern1d_h = gaussian_fn(nrow, std=(nrow - 1) / std)
    gkern1d_w = gaussian_fn(ncol, std=(ncol - 1) / std)
    gkern2d_distrbt = np.outer(gkern1d_h, gkern1d_w)
    gkern_map = np.random.random((nrow, ncol))
    gkern_map = gkern_map < gkern2d_distrbt

    loss_mask = mask.copy()
    loss_mask[loss_mask == 1] = gkern_map.flatten()

    center_x, center_y = [i // 2 for i in mask.shape]
    pad_x_l, pad_x_r = small_acs_block[0] // 2, small_acs_block[0] // 2 + small_acs_block[0] % 2
    pad_y_l, pad_y_r = small_acs_block[1] // 2, small_acs_block[1] // 2 + small_acs_block[0] % 2
    loss_mask[center_x - pad_x_l:center_x + pad_x_r, center_y - pad_y_l:center_y + pad_y_r] = 0

    trn_mask = mask - loss_mask

    trn_mask[:, :acq_start] = 1
    trn_mask[:, acq_end:] = 1
    loss_mask[:, :acq_start] = 1
    loss_mask[:, acq_end:] = 1

    return reshape_mask(trn_mask, kspace.shape), reshape_mask(loss_mask, kspace.shape), \
           torch.from_numpy(gkern2d_distrbt).unsqueeze(2).unsqueeze(0).to(dtype=torch.float32)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


def apply_mask(
        data: torch.Tensor,
        mask_func: MaskFunc,
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Optional[Sequence[int]] = None,
):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1]:] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
        x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
                not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(
        x: torch.Tensor, y: torch.Tensor, target_size: tuple
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the target_size.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.
        target_size: The size to crop

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """

    x = center_crop(x, target_size)
    y = center_crop(y, target_size)

    return x, y


def normalize(
        data: torch.Tensor,
        mean: Union[float, torch.Tensor],
        stddev: Union[float, torch.Tensor],
        eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
        data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        input: k-space after applying theta mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
        training_mask: Mask_theta for training.
        loss_mask: Mask_gamma for loss.
        ground_truch: A ground truth image.
        masked_kspace: Subampled kdata
    """

    input: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: list
    max_value: float
    crop_size: Tuple[int, int]
    training_mask: torch.Tensor
    loss_mask: torch.Tensor
    ground_truth: torch.Tensor
    ground_truch_k: torch.Tensor
    kspace_us: torch.Tensor
    pdf: torch.Tensor
    mask: torch.Tensor


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True, train_manner: str = "self_sv"):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            k_split_rate: The rate for being taken from the kdata for DC and loss
                k_split_rate[0] => for DC / k_split_rate[1] => for loss
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.train_manner = train_manner

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: Optional[np.ndarray],
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> VarNetSample:

        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))

        kspace_us, mask_torch, num_low_frequencies = apply_mask(kspace_torch, self.mask_func, seed=seed)

        return kspace_us
