import os, json, h5py, pathlib

import torch
import sigpy as sp
import sigpy.mri as mr
import numpy as np
from argparse import ArgumentParser

import utils


def cs(args):
    for dname in args.data_name:
        fname = args.data_path / dname
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][:]

        kspace = kspace.astype(np.complex64)
        ncoil, kx, ky, nz = kspace.shape
        sens_maps = torch.zeros((nz, ncoil, kx, ky, 2)) # to store all sens_maps

        for dataslice in range(nz):
            folderName = args.save_path / dname.stem

            name = f"{fname.stem}_{dataslice}"
            print(f"{name} start!")
            kspace_z = kspace[:, :, :, dataslice]

            sub_folder = folderName
            sens_map_pkl_path = sub_folder.parent / 'SensMaps' / f"{sub_folder.stem}_sens_maps.h5"
            if sens_map_pkl_path.exists():
                print(f"{sub_folder.stem}_sens_map.h5 is already exist!")
                with h5py.File(sens_map_pkl_path, "r") as hf:
                    sens_map = hf["sens_maps"][dataslice]
                sens_map = sens_map[...,0] + sens_map[...,1]*1j
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
                utils.save_result(name, sense_recon, sub_folder=sub_folder)
                print("Recon done...")
            print(f"{name} done!\n\n")

        else:
            utils.save_sens_maps(sens_maps, sub_folder)


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
        '--ESPIRiT_threshold',
        type=float,
        default=config["ESPIRiT_threshold"],
        help='ESPIRiT_threshold')

    parser.add_argument(
        '--CS_lambda',
        type=float,
        default=config["CS_lambda"],
        help='CS_lambda')

    args = parser.parse_args()
    data_path = args.data_path
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    slurm_job_id = "." if slurm_job_id == None else f"{slurm_job_id}.tinygpu"

    data_path = data_path.parent / slurm_job_id / data_path.name

    args.data_path = data_path
    args.data_name = [pathlib.Path(i) for i in os.listdir(data_path)] if args.data_name is None \
        else [pathlib.Path(args.data_name)]

    return args


def run_cs():
    config = json.load(open(pathlib.Path(__file__).parent / "config.json"))
    args = build_args(config)
    cs(args)


if __name__ == "__main__":
    run_cs()
