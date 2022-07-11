import pickle, plot, h5py
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    data = cp.asnumpy(data)
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def save_sens_maps(sens_maps, sub_folder):
    savePath = sub_folder.parent / 'SensMaps'
    savePath.mkdir(exist_ok=True, parents=True)
    h5f = h5py.File(savePath / f"{sub_folder.stem}_sens_maps.h5", 'w')
    h5f.create_dataset('sens_maps', data=sens_maps)
    h5f.close()

    recon_png_path = sub_folder.parent / f'pngs' / sub_folder.stem / "Sens_maps"
    recon_png_path.mkdir(exist_ok=True, parents=True)

    for i, sens_map in enumerate(sens_maps):
        png_path = recon_png_path / f"{sub_folder.stem}_{i}"
        sens_map = sens_map[..., 0] + sens_map[..., 1] * 1j
        sens_map = sens_map.detach().cpu().numpy()
        plot.ImagePlot(sens_map, z=0, save_path=png_path, save_basename='png', hide_axes=True)

def save_result(fname, result, sub_folder):
    recon_pkl_path = sub_folder.parent / f'CS_pkl' / sub_folder.stem
    recon_png_path = sub_folder.parent / f'pngs' / sub_folder.stem / "CS"
    recon_pkl_path.mkdir(exist_ok=True, parents=True)
    recon_png_path.mkdir(exist_ok=True, parents=True)

    pkl_path = recon_pkl_path / f"{fname}_CS.pkl"
    png_path = recon_png_path / f"{fname}_CS"

    h, w = result.shape
    result = cp.rot90(result[:, w // 4 * 1:w // 4 * 3 + 30])
    pickle.dump(result, open(pkl_path, 'wb'))
    imsave(result, png_path)


def imsave(obj, path):
    obj = cp.asnumpy(obj)
    f, a = plt.subplots(1, 1)
    a.imshow(abs(obj), cmap='gray')
    a.axis('off')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(28, 14)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()