import h5py
import torch
import sigpy as sp
import sigpy.mri as mr
import numpy as np
import utils

def CS(dname, args):
    SENS_EXIST = False
    with h5py.File(dname, "r") as hf:
        kspace = hf["kspace"][:]

    kspace = kspace.astype(np.complex64)
    ncoil, kx, ky, nz = kspace.shape
    sens_maps = torch.zeros((nz, ncoil, kx, ky, 2))  # to store all sens_maps
    sub_folder = args.save_path / dname.stem

    for dataslice in range(nz):

        name = f"{dname.stem}_{dataslice}"
        print(f"{name} start!")
        kspace_z = kspace[:, :, :, dataslice]

        sens_map_pkl_path = sub_folder.parent / 'SensMaps' / f"{sub_folder.stem}_sens_maps.h5"
        if sens_map_pkl_path.exists():
            print(f"{sub_folder.stem}_sens_map.h5 is already exist!")
            SENS_EXIST = True
            with h5py.File(sens_map_pkl_path, "r") as hf:
                sens_map = hf["sens_maps"][dataslice]
            sens_map = sens_map[..., 0] + sens_map[..., 1] * 1j
        else:
            print("Start to estimate sens_map...")
            sens_map = mr.app.EspiritCalib(kspace_z, device=sp.Device(0), thresh=args.ESPIRiT_threshold).run()
            sens_map_tensor = utils.to_tensor(sens_map)
            sens_maps[dataslice] = sens_map_tensor
            print("Estimate sens_map done...")

        recon_pkl_path = sub_folder.parent / f'CS_pkl' / sub_folder.stem / f"{name}_CS.pkl"
        if recon_pkl_path.exists():
            print(f"{name}_CS is already processed!")
        else:
            print("Start to reconstruction...")
            sense_recon = mr.app.L1WaveletRecon(kspace_z.copy(), sens_map.copy(), lamda=args.CS_lambda,
                                                device=sp.Device(0)).run()
            utils.save_result(name, sense_recon, sub_folder=sub_folder, recon="CS")
            print("Recon done...")
        print(f"{name} done!\n\n")

    else:
        if not SENS_EXIST:
            utils.save_sens_maps(sens_maps, sub_folder)
