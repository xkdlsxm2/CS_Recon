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
        kspace = np.transpose(kspace, axes=(3, 0, 1, 2))

    kspace = utils.undersample(kspace, args.rate)
    kspace = kspace.astype(np.complex64)

    sens_maps = torch.zeros((*kspace.shape, 2))  # to store all sens_maps
    recons = list()
    sub_folder = args.save_path / dname.stem
    for dataslice, kspace_z in enumerate(kspace):
        name = f"{dname.stem}_{dataslice}"
        print(f"    - {name} start!")
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
        print("Start to reconstruction...")
        sense_recon = mr.app.L1WaveletRecon(kspace_z.copy(), sens_map.copy(), lamda=args.CS_lambda,
                                            device=sp.Device(0)).run()
        recons.append(sense_recon)
        print("Recon done...")
        print(f"    - {name} done!\n\n")
    else:
        recons = abs(np.stack(recons))
        utils.save_result(dname.stem, recons, sub_folder=sub_folder, recon="CS")
        if not SENS_EXIST:
            utils.save_sens_maps(sens_maps, sub_folder)
