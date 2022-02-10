import pickle, torch, os
from pathlib import Path
from utils import *
from transform import to_tensor
import numpy as np

HDC_ResUNet_PATH = Path(
    r"C:\Users\z0048drc\Desktop\fastmri\logs\varnet\knee_leaderboard\hard_dc_resunet\lightning_logs\version_5\Test_pickle")
SDC_ResUNet_PATH = Path(r"")
HDC_UNet_PATH = Path(r"")
SDC_UNet_PATH = Path(r"")
SV_PATH = Path(r"")
CGSENCE_PATH = Path(r"C:\Users\z0048drc\Desktop\CS_recon\Recon\pkl")
GT_PATH = Path(r"C:\Users\z0048drc\Desktop\CS_recon\Recon\GT")

if __name__ == "__main__":
    target_idx = 16
    path_list = [CGSENCE_PATH, HDC_ResUNet_PATH, SDC_ResUNet_PATH, HDC_UNet_PATH, SDC_UNet_PATH, SV_PATH]
    names = ["CG-SENSE", "HDC_ResUNet", "SDC_ResUNet", "HDC_Unet", "SDC_Unet", "Supervised"]

    gt_path = [i for i in os.listdir(GT_PATH) if int(i.split('_')[0]) == target_idx][0]
    gt = pickle.load(open(GT_PATH/gt_path, 'rb'))
    ground_truth = gt['ground_truth'].detach().cpu().numpy()
    max_val = gt['max_val']

    outputs, ssims = [], []
    for i, (path, name) in enumerate(zip(path_list, names)):
        if len(path.stem) == 0: continue
        file = [i for i in os.listdir(path) if int(i.split('_')[0]) == target_idx][0]
        with open(path/file, 'rb') as p:
            output = abs(pickle.load(p))

        if type(output) == torch.Tensor:
            output = output.detach().cpu().numpy()

        if output.ndim == 3:
            output = output.reshape(output.shape[-2:])

        ssim_ = ssim(ground_truth[None, ...], output[None, ...], maxval=max_val)
        outputs.append(output)
        ssims.append(ssim_)

pass