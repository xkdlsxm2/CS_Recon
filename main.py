import pathlib, os
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
from utils import *
import h5py
from transform import apply_mask, to_tensor
from subsample import create_mask_for_mask_type

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1"

DATA_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\fastmri\dataset_cache.pkl")
# DATA_PATH = pathlib.Path(r"/home/hpc/iwbi/iwbi002h/fastMRI/fastMRI/dataset_cache_server.pkl")
CROP_SIZE = (320, 320)
NUM_LOG_IMAGES = 16
MASK_TYPE = "equispaced"
ACCELERATION = 4
NUM_LOW_FREQUENCIES = 24
MAX_ITER = 10

if __name__ == "__main__":
    files = get_files(DATA_PATH)

    # [:-1] => just make the same as training method, which was mistake.
    indices = list(np.linspace(0, len(files), NUM_LOG_IMAGES).astype(int))[:-1]
    for index in indices:
        fname, dataslice, metadata = files[index]
        name = f"{str(index)}_{fname.stem}_R{ACCELERATION}_C{NUM_LOW_FREQUENCIES}"

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

        sub_folder = f"R{ACCELERATION}_C{NUM_LOW_FREQUENCIES}"
        sens_map_pkl_path = pathlib.Path(f"Results/{sub_folder}/Sens_maps/{name}_sens_map.pkl")
        if sens_map_pkl_path.exists():
            print(f"{name}_sens_map is already processed!")
            sens_map = pickle.load(open(sens_map_pkl_path, 'rb'))
        else:
            print("Start to estimate sens_map...")
            sens_map = mr.app.EspiritCalib(input_k, device=sp.Device(0)).run()
            save_sens_map(name, sens_map, sub_folder=sub_folder)
            print("Estimate sens_map done...")

        recon_pkl_path = pathlib.Path(f"Results/{sub_folder}/Recon/pkl/{name}_CG-SENSE.pkl")
        if recon_pkl_path.exists():
            print(f"{name}_CG-SENSE is already processed!\n\n")

        else:
            print("Start to reconstruction...")
            sense_recon = mr.app.SenseRecon(input_k.copy(), sens_map.copy(), lamda=0.01, device=sp.Device(0)).run()
            sense_recon = center_crop(sense_recon, CROP_SIZE)
            save_recon(name, sense_recon, gt, sub_folder=sub_folder)
            print("Recon done...")
            print(f"{name} done!\n\n")