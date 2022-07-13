import numpy as np
import torch, h5py
from torch import nn
from tqdm import tqdm
from torch.nn.functional import mse_loss
import utils


class Grappa:
    def __init__(self, kernel_size, num_coils=15, device="cpu"):
        self.kernel_size = kernel_size
        self.pad = nn.ZeroPad2d(padding=kernel_size // 2)
        self.conv = nn.Conv2d(num_coils * 2, num_coils * 2, kernel_size=kernel_size, bias=False)
        self.device = device
        self.conv.to(self.device)

    def reset(self):
        self.conv.reset_parameters()

    def fit_weights(self, masked_kspace, mask, num_iters=500, lr=1e-3, weight_decay=0., l1_penalty=0.):
        optim = torch.optim.RMSprop(self.conv.parameters(), lr, weight_decay=weight_decay)

        chans, rows, cols, dims = masked_kspace.shape

        mask_ = np.array(mask.cpu().detach())
        slices = np.ma.clump_masked(np.ma.masked_where(mask_, mask_))
        acs_ind = [(s.start, s.stop - 1) for s in slices if s.start < (s.stop - 1)]
        assert (acs_ind != [] or len(
            acs_ind) > 1), "Couldn't extract center lines mask from k-space - is there pat2 undersampling?"
        acs_start = acs_ind[0][0]
        acs_end = acs_ind[0][1]

        acs = masked_kspace[:, :, acs_start:acs_end + 1, :]
        acs_masked1 = acs.detach().clone()
        acs_masked2 = acs.detach().clone()
        acs_masked3 = acs.detach().clone()
        acs_masked1[:, :, 1::3, :] = 0
        acs_masked2[:, :, 2::3, :] = 0
        acs_masked3[:, :, ::3, :] = 0

        acs_ = acs.permute(0, 3, 1, 2).contiguous().view(1, chans * dims, rows, -1).to(self.device)
        acs_masked1_ = acs_masked1.permute(0, 3, 1, 2).contiguous().view(1, chans * dims, rows, -1).to(self.device)
        acs_masked2_ = acs_masked2.permute(0, 3, 1, 2).contiguous().view(1, chans * dims, rows, -1).to(self.device)
        acs_masked3_ = acs_masked3.permute(0, 3, 1, 2).contiguous().view(1, chans * dims, rows, -1).to(self.device)

        for i in range(num_iters):
            acs_convo1 = self.conv(acs_masked1_)
            acs_convo2 = self.conv(acs_masked2_)
            acs_convo3 = self.conv(acs_masked3_)
            conv_cut = self.kernel_size // 2
            acs_loss = (mse_loss(acs_convo1, acs_[:, :, conv_cut:-conv_cut, conv_cut:-conv_cut]) +
                        mse_loss(acs_convo2, acs_[:, :, conv_cut:-conv_cut, conv_cut:-conv_cut]) +
                        mse_loss(acs_convo3, acs_[:, :, conv_cut:-conv_cut, conv_cut:-conv_cut])) * 1e9
            if l1_penalty > 0:
                acs_loss += l1_penalty * self.conv.weight.abs().sum()
            optim.zero_grad()
            acs_loss.backward()
            optim.step()

        return None

    def reconstruct(self, masked_kspace, mask):
        chans, rows, cols, dims = masked_kspace.shape
        masked_kspace_ = masked_kspace.permute(0, 3, 1, 2).contiguous().view(1, chans * dims, rows, -1).to(self.device)
        recons_ = self.pad(self.conv(masked_kspace_))
        recons = recons_.squeeze(0).view([chans, dims, rows, cols]).permute(0, 2, 3, 1)
        # Data Consistency
        recons = (~mask) * recons + mask * masked_kspace
        # Remove margin lines
        mask_ = np.squeeze(np.array(mask.cpu()))
        assert len(mask_.shape) == 1, "Expecting 1D mask"
        ind_first_line = np.nonzero(mask_)[0][0]
        ind_last_line = np.nonzero(mask_)[0][-1]
        if ind_first_line > 0:
            recons[:, :, :ind_first_line, :] = 0
        if ind_first_line < cols - 1:
            recons[:, :, ind_last_line + 1:, :] = 0

        return recons

    def get_kernel(self):
        return self.conv.weight

    def set_kernel(self, weight):
        self.conv.weight = nn.Parameter(weight.clone())


def compute_rss(kspace, dim=0):
    image = utils.root_sum_of_squares(utils.complex_abs(utils.ifft2(kspace)), dim)
    return image


def GRAPPA(dname, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub_folder = args.save_path / dname.stem

    with h5py.File(dname, 'r') as data:
        kspace = utils.to_tensor(data['kspace'][()]).permute(-2, 0, 1, 2, -1)

    mask = utils.create_mask(kspace)
    print(f'    - {dname.stem}')

    mask = mask.to(device)
    recons_kspace = []
    num_coils = kspace.shape[1]
    grappa = Grappa(5, num_coils=num_coils, device=device)

    for slice_i in tqdm(range(kspace.shape[0])):
        grappa.reset()
        grappa.fit_weights(kspace[slice_i], mask)
        recons_kspace_slice = grappa.reconstruct(kspace[slice_i].clone().to(device), mask).detach().cpu()
        recons_kspace.append(recons_kspace_slice)

    recons_kspace = torch.stack(recons_kspace)
    recons_img = compute_rss(recons_kspace.detach().cpu(), dim=1)

    recons_img = recons_img.detach().cpu().numpy()
    recons_img /= np.max(recons_img, axis=(1, 2))[:, None, None]
    mip = np.max(recons_img, axis=0)

    for dataslice, recon_img in enumerate(recons_img):
        name = f"{dname.stem}_{dataslice}"
        utils.save_result(name, recon_img, sub_folder=sub_folder, recon="GRAPPA")

    # Save MIP
    h, w = mip.shape
    mip = np.rot90(mip[:, w // 4 * 1:w // 4 * 3 + 30])
    path = sub_folder.parent / f'pngs' / sub_folder.stem / "GRAPPA" / f"MIP_GRAPPA_{dname.stem}"
    utils.imsave(mip, path)