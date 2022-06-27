import os, pickle, json, h5py, pathlib
import sigpy as sp
import sigpy.mri as mr
import numpy as np
from argparse import ArgumentParser
from utils import save_result


def cs(args):
    for dname in args.data_name:
        fname = args.data_path / dname
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][:]

        kspace = kspace.astype(np.complex64)
        ncoil, kx, ky, nz = kspace.shape
        for dataslice in range(nz):
            folderName = args.save_path / dname.stem
            folderName.mkdir(exist_ok=True, parents=True)

            name = f"{fname.stem}_{dataslice}"
            print(f"{name} start!")
            kspace_z = kspace[:, :, :, dataslice]

            sub_folder = folderName
            sens_map_pkl_path = sub_folder / "Sens_maps" / f"{name}_sens_map.pkl"
            if sens_map_pkl_path.exists():
                print(f"{name}_sens_map is already processed!")
                sens_map = pickle.load(open(sens_map_pkl_path, 'rb'))
            else:
                print("Start to estimate sens_map...")
                sens_map = mr.app.EspiritCalib(kspace_z, device=sp.Device(0), thresh=args.ESPIRiT_threshold).run()
                save_result(name, sens_map, sub_folder=sub_folder, type="Sens_maps")
                print("Estimate sens_map done...")

            recon_pkl_path = sub_folder / "CS" / f"{name}_CS.pkl"
            if recon_pkl_path.exists():
                print(f"{name}_CS is already processed!")
            else:
                print("Start to reconstruction...")
                sense_recon = mr.app.L1WaveletRecon(kspace_z.copy(), sens_map.copy(), lamda=args.CS_lambda,
                                                    device=sp.Device(0)).run()
                save_result(name, sense_recon, sub_folder=sub_folder)
                print("Recon done...")
            print(f"{name} done!\n\n")


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

    config = config_json['data']
    parser.add_argument(
        '--reduce_size',
        type=bool,
        default=config["reduce_size"],
        help='whether to reduce the data size fo running the code on the local laptop')

    args = parser.parse_args()
    args.data_name = [pathlib.Path(i) for i in os.listdir(args.data_path)] if args.data_name is None \
        else [pathlib.Path(args.data_name)]

    return args


def run_cs():
    config = json.load(open(pathlib.Path(__file__).parent / "config.json"))
    args = build_args(config)
    cs(args)


if __name__ == "__main__":
    run_cs()
