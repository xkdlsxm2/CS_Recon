import pathlib, os
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
from utils import *
import h5py
from transform import apply_mask, to_tensor
from subsample import create_mask_for_mask_type

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1"
RECON_SAVE_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\fastmri\results\Benchmark\CS_CG_FM_SSL\test")
DATA_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\CS_recon\Indiv_recon\data")
CROP_SIZE = (320, 320)
NUM_LOG_IMAGES = 16
MASK_TYPE = "equispaced"
ACCELERATION = 4
NUM_LOW_FREQUENCIES = 24
MAX_ITER = 30
RECON = ["CS-SENSE"]
DATASET = "test"


def save_recon(fname, CG_SENSE, sub_folder, recon):
    recon_pkl_path = pathlib.Path(f"{sub_folder}/Recon/pkl")
    recon_png_path = pathlib.Path(f"{sub_folder}/Recon/png")
    recon_pkl_path.mkdir(exist_ok=True, parents=True)
    recon_png_path.mkdir(exist_ok=True, parents=True)

    pkl_path = recon_pkl_path / f"{fname}_{recon}.pkl"
    png_path = recon_png_path / f"{fname}_{recon}.png"
    pickle.dump(CG_SENSE, open(pkl_path, 'wb'))
    imsave(CG_SENSE, png_path)


def getfiles(path):
    file_list = []
    for x in path.iterdir():
        if x.is_file():
            file_list.append(x)

    return file_list


if __name__ == "__main__":
    for recon in RECON:
        files = getfiles(DATA_PATH)
        for fname in files:
            dataslice = int(fname.stem.split('-')[1])
            name = f"{fname.stem}_R{ACCELERATION}_C{NUM_LOW_FREQUENCIES}"
            print(f"{name} start!")

            mask = create_mask_for_mask_type(MASK_TYPE, ACCELERATION, NUM_LOW_FREQUENCIES)
            with h5py.File(fname, "r") as hf:
                kspace = hf["kspace"][dataslice]

                target = hf["reconstruction_rss"][dataslice]

                kspace_torch = to_tensor(kspace)

                seed = tuple(map(ord, fname.name))

                kspace_us = apply_mask(kspace_torch, mask, seed=seed)

            input_k = kdata_torch2numpy(kspace_us)

            sub_folder = f"R{ACCELERATION}C{NUM_LOW_FREQUENCIES}"

            sens_map = mr.app.EspiritCalib(input_k, device=sp.Device(0)).run()
            print("Estimate sens_map done...")

            sub_folder = RECON_SAVE_PATH / sub_folder
            if recon == "CS-SENSE":
                sense_recon = mr.app.L1WaveletRecon(input_k.copy(), sens_map.copy(), lamda=3E-6,
                                                    device=sp.Device(0), max_iter=MAX_ITER).run()
            else:
                sense_recon = mr.app.SenseRecon(input_k.copy(), sens_map.copy(), lamda=0.01,
                                                device=sp.Device(0), max_iter=MAX_ITER).run()
            sense_recon = center_crop(sense_recon, CROP_SIZE)
            save_recon(name, sense_recon, sub_folder=sub_folder, recon=recon)
            print("Recon done...")
            print(f"{name} done!\n\n")
