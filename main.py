import pathlib
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
from utils import *

DATA_PATH = pathlib.Path(r"C:\Users\z0048drc\Desktop\fastmri\Input_data")
CROP_SIZE = (320, 320)

if __name__ == "__main__":
    files = searching_all_files(DATA_PATH)

    for file in files:
        print(f"{file.stem} start!")
        input_k = CPU_Unpickler(open(file, 'rb')).load()
        input_k = kdata_torch2numpy(input_k)

        sens_map_pkl_path = pathlib.Path(f"Sens_maps/{file.stem}_sens_map.pkl")
        if sens_map_pkl_path.exists():
            print(f"{file.stem} is already processed!\n\n")
            continue
        else:
            print("Start to estimate sems_map...")
            sens_map = mr.app.EspiritCalib(input_k).run()
            print("Estimate sens_map done...")

            print("Start to recon...")
            sense_recon = mr.app.SenseRecon(input_k.copy(), sens_map.copy(), lamda=0.01).run()
            sense_recon = center_crop(sense_recon, CROP_SIZE)
            save(file, [sens_map, sense_recon])
            print("Recon done...")
            print(f"{file.stem} done!\n\n")
