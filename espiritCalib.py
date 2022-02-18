import pathlib, os
import sigpy as sp
import sigpy.mri as mr
import torch, pickle
import numpy as np
import h5py
from transform import apply_mask, to_tensor
from subsample import create_mask_for_mask_type

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1"

DATA_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\fastmri\dataset_cache.pkl")
# DATA_PATH = pathlib.Path(r"/home/hpc/iwbi/iwbi002h/fastMRI/fastMRI/dataset_cache.pkl")
SENS_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\CS_recon\Results\Sens_Maps")
DATASET = "test"  # "train", "val", "test"
MASK_TYPE = "equispaced"
ACCELERATION = 4
NUM_LOW_FREQUENCIES = 24


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
    files = pickle.load(open(directory, 'rb'))
    key = [key for key in list(files.keys()) if DATASET in key.stem][0]

    return np.array(files[key])


def save_sens_map(fname, sens_map, path):
    sens_map_pkl_path = pathlib.Path(f"{path}")
    sens_map_pkl_path.mkdir(exist_ok=True, parents=True)
    sens_map_pkl_path = sens_map_pkl_path / f"{fname}.pkl"

    pickle.dump(sens_map, open(sens_map_pkl_path, 'wb'))


if __name__ == "__main__":
    files = get_files(DATA_PATH)
    SENS_PATH = SENS_PATH / DATASET

    for fname, dataslice, metadata in files:
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
            sens_map = mr.app.EspiritCalib(input_k, device=sp.Device(0)).run()
            sens_map_tensor = to_tensor(sens_map.get())
            save_sens_map(name, sens_map_tensor, path=SENS_PATH)
            print("Estimate sens_map done...")
