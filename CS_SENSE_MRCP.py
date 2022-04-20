import pathlib, os
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
from utils import *
import h5py
from transform import apply_mask, to_tensor
from subsample import create_mask_for_mask_type

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1"
RECON_SAVE_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\CS_recon\Results\MRCP")
DPATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\data_fm\MRCP")
DNAME = [pathlib.Path(r"meas_MID00062_FID07837_t2_space_cor_p3_trig_384_iso-RO_half_FTz_center.h5"),
         pathlib.Path(r"meas_MID00062_FID07837_t2_space_cor_p3_trig_384_iso-RO_half_FTz.h5")]



CROP_SIZE = (704, 300)
RECON = ["CS-SENSE"]


def save_recon(fname, recon_img, sub_folder, recon_type="CS"):
    recon_pkl_path = sub_folder / 'pkl'
    recon_png_path = sub_folder / 'png'
    recon_pkl_path.mkdir(exist_ok=True, parents=True)
    recon_png_path.mkdir(exist_ok=True, parents=True)

    pkl_path = recon_pkl_path / f"{fname}_{recon_type}.pkl"
    png_path = recon_png_path / f"{fname}_{recon_type}.png"
    pickle.dump(recon_img, open(pkl_path, 'wb'))
    imsave(recon_img, png_path)


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
    for dname in DNAME:
        fname = DPATH / dname
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][:]
        ncoil, kx, ky, nz = kspace.shape
        for dataslice in range(nz):
            folderName = RECON_SAVE_PATH / dname.stem
            folderName.mkdir(exist_ok=True, parents=True)

            name = f"{fname.stem}_{dataslice}"
            print(f"{name} start!")
            kspace_z = kspace[:, :, :, dataslice]

            sub_folder = folderName / "Sens_maps"
            sens_map_pkl_path = sub_folder / f"{name}_sens_map.pkl"
            if sens_map_pkl_path.exists():
                print(f"{name}_sens_map is already processed!")
                sens_map = pickle.load(open(sens_map_pkl_path, 'rb'))
            else:
                print("Start to estimate sens_map...")
                sens_map = mr.app.EspiritCalib(kspace_z, device=sp.Device(0)).run()
                save_sens_map(name, sens_map, sub_folder=sub_folder)
                print("Estimate sens_map done...")

            sub_folder = folderName / "Recon"
            print("Start to reconstruction...")
            sense_recon = mr.app.L1WaveletRecon(kspace_z.copy(), sens_map.copy(), lamda=3E-6, device=sp.Device(0)).run()
            # sense_recon = center_crop(sense_recon, (kx, ky))
            save_recon(name, sense_recon, sub_folder=sub_folder)
            print("Recon done...")
            print(f"{name} done!\n\n")
