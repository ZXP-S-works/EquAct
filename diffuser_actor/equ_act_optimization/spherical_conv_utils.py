# modified from Image to Sphere: Learning Equivariant Features for Efficient Pose Prediction
# https://colab.research.google.com/github/dmklee/image2sphere/blob/main/model_walkthrough.ipynb
import einops
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToPILImage
import e3nn
from e3nn import o3
import healpy as hp
import matplotlib.pyplot as plt

from utils import pytorch3d_transforms


def s2_irreps(lmax):
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax):
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def flat_wigner(lmax, alpha, beta, gamma):
    return torch.cat([
        (2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)
    ], dim=-1)


def so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=2 * np.pi, n_alpha=8, n_beta=3, n_gamma=None):
    """Spatial grid over SO3 used to parametrize localized filter

  :return: rings of rotations around the identity, all points (rotations) in
           a ring are at the same distance from the identity
           size of the kernel = n_alpha * n_beta * n_gamma
  """
    if n_gamma is None:
        n_gamma = n_alpha
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * np.pi, n_alpha)[:-1]
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return torch.stack((A, B, C))


class S2Conv(nn.Module):
    '''S2 group convolution which outputs signal over SO(3) irreps

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass.
  '''

    def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
        super().__init__()
        # filter weight parametrized over spatial grid on S2
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_s2_pts]

        # linear projection to convert filter weights to fourier domain
        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
        )  # [n_s2_pts, (2*lmax+1)**2]

        # defines group convolution using appropriate irreps
        # note, we set internal_weights to False since we defined our own filter above
        self.lin = o3.Linear(s2_irreps(lmax), so3_irreps(lmax),
                             f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        """Perform S2 group convolution to produce signal over irreps of SO(3).
        First project filter into fourier domain then perform convolution

        :x: tensor of shape (B, f_in, (lmax+1)**2), signal over S2 irreps
        :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
        """
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        psi = einops.rearrange(psi, 'in out irrep -> out in irrep')
        x = einops.rearrange(x, 'b in irrep -> in b irrep')
        return self.lin(psi, weight=x)


class SO3Conv(nn.Module):
    '''SO3 group convolution

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas, gammas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass
  '''

    def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
        super().__init__()

        # filter weight parametrized over spatial grid on SO3
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]

        # wigner D matrices used to project spatial signal to irreps of SO(3)
        self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, sum_l^L (2*l+1)**2]

        # defines group convolution using appropriate irreps
        self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax),
                             f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        '''Perform SO3 group convolution to produce signal over irreps of SO(3).
    First project filter into fourier domain then perform convolution

    :x: tensor of shape (B, f_in, sum_l^L (2*l+1)**2), signal over SO3 irreps
    :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
    '''
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)


def so3_healpix_grid(rec_level: int = 3):
    """Returns healpix grid over so3 of equally spaced rotations

    https://github.com/google-research/google-research/blob/4808a726f4b126ea38d49cdd152a6bb5d42efdf0/implicit_pdf/models.py#L272
    alpha: 0-2pi around Y
    beta: 0-pi around X
    gamma: 0-2pi around Y
    rec_level | num_points | bin width (deg)
    ----------------------------------------
         0    |         72 |    60
         1    |        576 |    30
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


def compute_trace(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns Tr(rotA, rotB.T)
    '''
    prod = torch.matmul(rotA.unsqueeze(1), rotB.transpose(-1, -2).unsqueeze(0))
    trace = prod.diagonal(dim1=-1, dim2=-2).sum(-1)
    return trace


def rotation_error(rotA, rotB):
    """
    rotA, rotB are tensors of shape (*,3,3)
    returns rotation error in radians, tensor of shape (*)
    """
    trace = compute_trace(rotA, rotB)
    return torch.arccos(torch.clamp((trace - 1) / 2, -1, 1))


def nearest_rotmat(src, target):
    """return index of target that is nearest to each element in src
    uses negative trace of the dot product to avoid arccos operation
    :src: tensor of shape (B, 3, 3)
    :target: tensor of shape (*, 3, 3)
    """
    assert src.dim() == target.dim() == 3
    trace = compute_trace(src.unsqueeze(1), target.unsqueeze(0))

    return torch.max(trace, dim=1)[1]


def nearest_pose_idx(src, target):
    """return index of target that is nearest to each element in src of shape (B, 1)
    uses l2 norm
    :src: tensor of shape (B, npts, 3)
    :target: tensor of shape (B, 1, 3)
    """
    # ToDo: test it
    l2_distance = (src - target).norm(dim=2, p=2)

    return l2_distance.argmin(dim=1)


def s2_healpix_grid(rec_level: int = 0, max_beta: float = np.pi / 6):
    """Returns healpix grid up to a max_beta
    """
    n_side = 2 ** rec_level
    npix = hp.nside2npix(n_side)
    m = hp.query_disc(nside=n_side, vec=(0, 0, 1), radius=max_beta)
    beta, alpha = hp.pix2ang(n_side, m)
    alpha = torch.from_numpy(alpha)
    beta = torch.from_numpy(beta)
    return torch.stack((alpha, beta)).float()

def plot_so3_distribution(probs: torch.Tensor,
                          rots: torch.Tensor,
                          gt_rot_quat=None,
                          fig=None,
                          ax=None,
                          display_threshold_probability=0.000005,
                          show_color_wheel: bool=True,
                          canonical_rotation=torch.eye(3),
                          ):
    '''
    Taken from https://github.com/google-research/google-research/blob/master/implicit_pdf/evaluation.py
    '''
    cmap = plt.cm.hsv

    def _show_single_marker(ax, rotation, marker):
        # alpha, beta, gamma = o3.matrix_to_angles(rotation)
        # euler = Rotation.from_matrix(rotation).as_euler('YXY')
        euler = pytorch3d_transforms.matrix_to_euler_angles(rotation, 'YXY')
        alpha, beta, gamma = euler[0], euler[1], euler[2]
        color = cmap(0.5 + gamma.repeat(2) / 2. / np.pi)[-1]
        ax.scatter(alpha, beta-np.pi/2, s=2000, edgecolors=color, facecolors='none', marker=marker, linewidth=5)
        ax.scatter(alpha, beta-np.pi/2, s=1500, edgecolors='k', facecolors='none', marker=marker, linewidth=2)
        ax.scatter(alpha, beta-np.pi/2, s=2500, edgecolors='k', facecolors='none', marker=marker, linewidth=2)

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=100)
        fig.subplots_adjust(0.01, 0.08, 0.90, 0.95)
        ax = fig.add_subplot(111, projection='mollweide')

    rots = rots @ canonical_rotation
    scatterpoint_scaling = 3e3
    # alpha, beta, gamma = o3.matrix_to_angles(rots)
    # euler = Rotation.from_matrix(rots).as_euler('YXY')
    euler = pytorch3d_transforms.matrix_to_euler_angles(rots, 'YXY')
    alpha, beta, gamma = euler[:, 0], euler[:, 1], euler[:, 2]

    # offset alpha and beta so different gammas are visible
    R = 0.02
    alpha += R * np.cos(gamma)
    beta += R * np.sin(gamma)

    which_to_display = (probs > display_threshold_probability)

    # Display the distribution
    ax.scatter(alpha[which_to_display],
               beta[which_to_display]-np.pi/2,
               s=scatterpoint_scaling * probs[which_to_display],
               c=cmap(0.5 + gamma[which_to_display] / 2. / np.pi))
    if gt_rot_quat is not None:
        mat = pytorch3d_transforms.quaternion_to_matrix(gt_rot_quat)
        gt_rotation = torch.tensor(mat).float()
        gt_rotation = gt_rotation @ canonical_rotation
        _show_single_marker(ax, gt_rotation, 'o')
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticklabels([r'90$\degree$', None,
                            r'180$\degree$', None,
                            r'270$\degree$', None,
                            r'0$\degree$'], fontsize=12)
        ax.spines['polar'].set_visible(False)
        plt.text(0.5, 0.5, 'Tilt', fontsize=12,
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)
    plt.show()
    # img = plot_to_image(fig)
    # plt.close(fig)
    # return img
