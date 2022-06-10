import os, pickle
import matplotlib.pyplot as plt
import pathlib
PATH = pathlib.Path(r'C:\Users\z0048drc\Desktop\CS_recon\Results\MRCP\meas_MID00062_FID07837_t2_space_cor_p3_trig_384_iso-RO_half_center_FTz\Sens_maps')

for sm_name in os.listdir(PATH):
    sm_name = pathlib.Path(sm_name)
    sens_map = pickle.load(open(PATH/sm_name, 'rb'))
    fig = plt.figure()

    for idx, sens_map_coil in enumerate(sens_map, start=1):
        ax = fig.add_subplot(6, 8, idx)
        ax.imshow(abs(sens_map_coil), cmap='gray', vmin=0.25, vmax=1.25)
        ax.axis('off')
    plt.savefig(f'{PATH.parent/"Sens_maps_png"/sm_name.stem}.png', bbox_inches='tight', pad_inches=0)
    plt.close()