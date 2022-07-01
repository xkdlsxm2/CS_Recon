import pickle, plot
import cupy as cp
import matplotlib.pyplot as plt


def save_result(fname, result, sub_folder, type="CS"):
    if type == "Sens_maps":
        recon_pkl_path = sub_folder.parent / 'SensMaps_pkl' / sub_folder.stem
    else:
        recon_pkl_path = sub_folder.parent / f'{type}_pkl' / sub_folder.stem

    recon_png_path = sub_folder.parent / f'pngs' / sub_folder.stem / type
    recon_pkl_path.mkdir(exist_ok=True, parents=True)
    recon_png_path.mkdir(exist_ok=True, parents=True)

    pkl_path = recon_pkl_path / f"{fname}_{type}.pkl"
    png_path = recon_png_path / f"{fname}_{type}"
    pickle.dump(result, open(pkl_path, 'wb'))

    if type == "Sens_maps":
        plot.ImagePlot(result, z=0, save_path=png_path, save_basename='png', hide_axes=True)
    else:
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