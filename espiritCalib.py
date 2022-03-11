import pathlib, os
import sigpy as sp
import sigpy.mri as mr
import torch, pickle
import numpy as np
import h5py
import sigpy.plot as pl
from transform import apply_mask, to_tensor
from subsample import create_mask_for_mask_type

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1"

DATA_CACHE_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\CS_recon\dataset_cache_local.pkl")
# DATA_CACHE_PATH = pathlib.Path(r"/home/hpc/iwbi/iwbi002h/fastMRI/fastMRI/dataset_cache_server.pkl")
DATASET = "test"  # "train", "val", "test"
DATA_PATH = pathlib.Path(rf"C:\Users\z0048drc\Desktop\data_fm\knee_from_val_test\multicoil_{DATASET}")
SENS_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\CS_recon\Results\sens_maps_local\TH0")
MASK_TYPE = "equispaced"
ACCELERATION = 4
NUM_LOW_FREQUENCIES = 24


def apply_mask(data, mask_func, offset=None, seed=None, padding=None, ):
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


def kdata_torch2numpy(kdata_torch: torch.Tensor):
    kdata_np = kdata_torch.detach().cpu().numpy()

    return (kdata_np[..., 0] + 1j * kdata_np[..., 1])


def get_files(directory: pathlib.Path):
    if os.name == 'nt':  # for Windows, there is no PosixPath.
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            with open(directory, "rb") as f:
                files = pickle.load(f)
        finally:
            pathlib.PosixPath = posix_backup
    else:
        with open(directory, "rb") as f:
            files = pickle.load(f)

    key = [key for key in list(files.keys()) if DATASET in key.stem][0]

    return np.array(files[key])


def save_sens_map(fname, sens_map, path):
    sens_map_pkl_path = pathlib.Path(f"{path}")
    sens_map_pkl_path.mkdir(exist_ok=True, parents=True)
    sens_map_pkl_path = sens_map_pkl_path / f"{fname}.pkl"

    pickle.dump(sens_map, open(sens_map_pkl_path, 'wb'))


if __name__ == "__main__":
    files = get_files(DATA_CACHE_PATH)
    SENS_PATH = SENS_PATH / DATASET
    for fname, dataslice, metadata in files:
        fname = DATA_PATH / fname.name
        name = f"{fname.stem}_{str(dataslice)}_sens_map"

        print(f"{name} start!")

        mask = create_mask_for_mask_type(MASK_TYPE, ACCELERATION, NUM_LOW_FREQUENCIES)
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            target = hf["reconstruction_rss"][dataslice]

            attrs = dict(hf.attrs)
            attrs.update(metadata)

            kspace_torch = to_tensor(kspace)

            seed = tuple(map(ord, fname.name))

            kspace_us = apply_mask(kspace_torch, mask, seed=seed)

            gt = {"ground_truth": to_tensor(target),
                  "max_val": attrs["max"]}
        input_k = kdata_torch2numpy(kspace_us)
        
        sens_map_pkl_path = pathlib.Path(f"{SENS_PATH}/{name}.pkl")
        if sens_map_pkl_path.exists():
            print(f"{name} is already processed!")
            sens_map = pickle.load(open(sens_map_pkl_path, 'rb'))
        else:
            print("Start to estimate sens_map...")
            sens_map = mr.app.EspiritCalib(input_k, device=sp.Device(0), thresh=0.0).run()
            sens_map_tensor = to_tensor(sens_map.get())
            save_sens_map(name, sens_map_tensor, path=SENS_PATH)
            print("Estimate sens_map done...")
