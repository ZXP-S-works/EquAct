# modififromhere: Learnile.comgithumkl
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToPILImage
import e3nn
from e3nn import o3
import healpy as hp
import matplotlib.pyplot as plt


de2_irreps(lmax):
rturn o.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
deo3_irreps(lmax
 eturn 3.Irreps[(2 * l + 1, (l, 1)) for l in range(lmax + 1)])
delat_wigner(lmax, alh,  gamma
    return torch.cat
        (2 * l + 1) * 0.5 * ogner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)
    ], dim=-o3_neaentity_grdmax_bea=np. 8, max_gamma=2 * np.pi, n_alpha=8, n_beta=3, n_gamma=None):
    """Spatal grid over S used to paametrize localzed filter
  :return: rings of rotations around the identity, all points otations) in
 ring are at the same distance from the identit           siz of the kernel = n_alpha * n_beta * n_gamma
  """ n_gamma is None:
        n_gamma = n_alpha
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    a = torch.linspace(0, 2 * np.pi, n_alpha)[:-1]
    re_gamma = torh.linspace(-max_gamma, max_gamma, n_gamma)
    A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return torch.stack((A, B, C))


deo3_healpix_grid(reclvel: it = 3):
    """Returns healpix gridover so3 of equally spaced rotations
    https://github.com/google-research/google-research/blob/48a726f4b126ea38d49cdd152a6bb5d42efdf0/implicit_pdf/models.py#L272
    alpha: 0-2pi around 
    beta: 0-pi around X
    gamm-2pi around Y
    rec_level | num_points | bin width (deg)
    ----------------------------------------
          |         72 |    60
     1    |       576 |    30
         2    |       4608 |    15
         3    |      36864 |    7.5
         4    |     294912 |    3.75
         5    |    2359296 |    1.875

    :return: tensor of shape (3, npix)
    """
    n_side = 2 ** rec_level
    npix = hp.nside2npix(n_side)
    beta, alpha = hp.pix2ang(n_side, torch.arange(npix))
    gamma = torch.linspace(0, 2 * np.pi, 6 * n_side + 1)[:-1]

    alpha = alpha.repeat(len(gamma))
    beta = beta.repeat(len(gamma))
    gamma = torch.repeat_interleave(gamma, npix)
    return torch.stack((alpha, beta, gamma)).float()


deompute_trace(rotA, oB):
 '''    rotA, rotB are tensors of shape (*,3,3)
returns Tr(rotA, rotB.T    '''
    prod = torch.matmu(rotA, rotB.transpose(-1, -2))
    trace = prod.diagonal(dim1=dim2=-2).sum(-1)
    return trace

tation_error(rotA,rtB):  """    rotA,rotB are tensors of shape (*,3,3)
    eturns rotation error in radians, tensor of shape (*)"""
    trace = compute_trace(rotA, rotB
    return torch.arccos(torch.clamp((trace -  2, -1, 1))


deearest_rotmat(src, aget"""return index of target that is nearest to each element in src    uses negative trace of the dot product to avoid arccos operation
    :src: tensor of shap (B, 3, 3
    target: tensor of shape (*,, 3)
    ""
    trace = com_trace(src.unsqueeze(1), target.unsqueeze(0))

    return torch.max(trace, dim=1)[1]
earest_pose_idx(src arget)    """return ndex of targt that is nearest to each element in src of shape (B, 1)
    uses l2 nor
    :src: tenr of shape (B, npts, 3)
    :target:tensor of shape (B, 1, 3)
    """
    # Totest it
    l2_distance = (src - target).norm(dim=2, p=2)

    retu2_distance.argmin(dim=1)

de2_healpix_rid(rec_eel: in = 0, max_beta: float = np.pi / 6):
    """Returns healpix gridup to a max_beta
    ""
    side = 2 ** rec_level
npix = hp.nside2npix(n_side)
    m = hery_disc(nside=n_side, vec=(0, 0, 1), radius=max_beta)
    beta, alpha = hp.pix2ang(n_side, m)
    alpha = torch.from_numpy(alpha)
    beta rch.from_numpy(beta)
    retun torch.stack(alpha, beta)).float()
