import pathlib, os
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
from utils import *
import h5py
from transform import apply_mask, to_tensor
from subsample import create_mask_for_mask_type

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1"

DATA_CACHE_PATH = pathlib.Path(r"dataset_cache_local.pkl")
SENS_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\CS_recon\Results\sens_maps_local\TH0")
RECON_SAVE_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\CS_recon\Results")
CROP_SIZE = (320, 320)
NUM_LOG_IMAGES = 16
MASK_TYPE = "equispaced"
ACCELERATION = 1
NUM_LOW_FREQUENCIES = 24
MAX_ITER = 10
RECON = ["CS-SENSE"]
DATASET = "train"

def save_recon(fname, CG_SENSE, sub_folder, recon):
    recon_pkl_path = pathlib.Path(f"{sub_folder}/Recon/pkl")
    recon_png_path = pathlib.Path(f"{sub_folder}/Recon/png")
    recon_pkl_path.mkdir(exist_ok=True, parents=True)
    recon_png_path.mkdir(exist_ok=True, parents=True)

    pkl_path = recon_pkl_path / f"{fname}_{recon}.pkl"
    png_path = recon_png_path / f"{fname}_{recon}.png"
    pickle.dump(CG_SENSE, open(pkl_path, 'wb'))
    imsave(CG_SENSE, png_path)

def get_files(directory: pathlib.Path, dataset):
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

    key = [key for key in list(files.keys()) if dataset in key.stem][0]

    return np.array(files[key])


if __name__ == "__main__":
    for recon in RECON:
        files = get_files(DATA_CACHE_PATH, DATASET)
        SENS_PATH = SENS_PATH / DATASET

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

            input_k = kdata_torch2numpy(kspace_us)

            sub_folder = f"R{ACCELERATION}C{NUM_LOW_FREQUENCIES}"
            sens_map_pkl_path = pathlib.Path(f"{SENS_PATH}/{name}.pkl")
            if sens_map_pkl_path.exists():
                print(f"{name}_sens_map is already processed!")
                sens_map = pickle.load(open(sens_map_pkl_path, 'rb'))
            else:
                print("Start to estimate sens_map...")
                sens_map = mr.app.EspiritCalib(input_k, device=sp.Device(0), thresh=0.0).run()
                save_sens_map(name, sens_map, sub_folder=sub_folder)
                print("Estimate sens_map done...")

            sub_folder = RECON_SAVE_PATH / sub_folder
            recon_pkl_path = sub_folder / f"Recon/pkl/{name}_{recon}.pkl"
            if recon_pkl_path.exists():
                print(f"{name}_{recon} is already processed!\n\n")

            else:
                print("Start to reconstruction...")
                if recon == "CS-SENSE":
                    sense_recon = mr.app.L1WaveletRecon(input_k.copy(), sens_map.copy(), lamda=3E-6,
                                                        device=sp.Device(0)).run()
                else:
                    sense_recon = mr.app.SenseRecon(input_k.copy(), sens_map.copy(), lamda=0.01,
                                                    device=sp.Device(0)).run()
                sense_recon = center_crop(sense_recon, CROP_SIZE)
                save_recon(name, sense_recon, sub_folder=sub_folder, recon=recon)
                print("Recon done...")
                print(f"{name} done!\n\n")
