import time
import numpy as np
import torch
import escnn
import e3nn
from einops import rearrange
from scipy.spatial.transform import Rotation


def flat_wigner(lmax, alpha, beta, gamma):
    return torch.cat([(2 * l + 1) ** 0.5 * e3nn.o3.wigner_D(l, alpha, beta, gamma).flatten(-2)
                      for l in range(lmax + 1)], dim=-1).float()


def so3_irreps(lmax):
    return e3nn.o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def so3_near_identity_grid(max_beta=np.pi, max_gamma=2 * np.pi, n_alpha=8, n_beta=3, n_gamma=None):
    """
    :return: rings of rotations around the identity, all points (rotations) in
    a ring are at the same distance from the identity
    size of the kernel = n_alpha * n_beta * n_gamma
    ??? ZXP conv by using a grid? Equally-spaced grid?
    """
    if n_gamma is None:
        n_gamma = n_alpha  # similar to regular representations
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * np.pi, n_alpha)[:-1]
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return torch.stack((A, B, C))


class SO3Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax_in, lmax_out=None, kernel_grid=None):
        super().__init__()
        if kernel_grid is None:
            kernel_grid = so3_near_identity_grid()
        if lmax_out is None:
            lmax_out = lmax_in

        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]
        self.register_buffer("D", flat_wigner(lmax_in, *kernel_grid))  # [n_so3_pts, psi]
        self.lin = e3nn.o3.Linear(so3_irreps(lmax_in), so3_irreps(lmax_out), f_in=f_in,
                                  f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)


if __name__ == "__main__":
    # ToDo: WARNING: ESCNN and E3NN has different coordinate systems for the tensor, double check their documents!
    B = 8
    C_in = 1
    C_out = 1
    l_in = 6
    l_out = 6

    oct_act = escnn.gspaces.octaOnR3()

    in_type = escnn.nn.FieldType(oct_act, C_in * [oct_act.trivial_repr])
    out_type = escnn.nn.FieldType(oct_act, C_out * [oct_act.regular_repr])
    oct_conv = escnn.nn.R3Conv(in_type, out_type, 3)

    # create local region
    x = torch.randn((B, C_in, 3, 3, 3))
    x = escnn.nn.GeometricTensor(x, in_type)

    # for converting to harmonics, need to know Octahedral group elements in SO3
    oct_quats = torch.from_numpy(np.array([g._element for g in oct_act.testing_elements]))
    oct_on_so3 = torch.stack(e3nn.o3.quaternion_to_angles(oct_quats))
    oct_xyz = e3nn.o3.angles_to_xyz(*oct_on_so3[:2])  # ZXP ??? what is xyz
    # import matplotlib.pyplot as plt
    # f = plt.figure()
    # ax = f.add_subplot(111, projection='3d')
    # ax.scatter(*oct_on_so3)
    # plt.show()
    # exit()

    # WignerD matrix defines linear projection from spatial to harmonics
    D_oct = flat_wigner(l_in, *oct_on_so3)
    D_oct /= D_oct.shape[0] ** 0.5
    print(D_oct.shape)

    # converting back to spatial at arbitrary resolution given by alpha, beta, gamma
    # for this demo, i will convert back using octa_angles to make testing equiv error easier
    D_out = flat_wigner(l_out, *oct_on_so3)
    D_out /= D_out.shape[1] ** 0.5

    # define so3_conv in fourier domain
    so3_conv = SO3Convolution(C_out, C_out, l_in, l_out)

    # example of nonlinearity
    so3_activation = e3nn.nn.SO3Activation(l_out, l_out, torch.relu, 10)


    def function(x: escnn.nn.GeometricTensor) -> escnn.nn.GeometricTensor:
        x = oct_conv(x)

        # remove spatial elements and arrange so each group element has features
        x = rearrange(x.tensor, 'b (c g) () () () -> b c g',
                      c=C_out, g=oct_act.fibergroup.order())

        x_harmonic = x @ D_oct  # b c (l)
        x_harmonic = so3_conv(x_harmonic)
        # x_harmonic = so3_activation(x_harmonic)
        # l could be chosen to be 2 in 4 out
        # convert back to escnn oct type for calc equiv error
        x = x_harmonic @ D_out.transpose(0, 1)
        x = rearrange(x, 'b c g -> b (c g) 1 1 1')
        x = escnn.nn.GeometricTensor(x, out_type)

        return x, x_harmonic

    so3irreps = so3_irreps(l_out)

    y, harmonic = function(x)
    for i, g in enumerate(oct_act.testing_elements):
        x_tfm = x.transform(g)
        rot_mtx = so3irreps.D_from_angles(*oct_on_so3[:, i]).float()
        harmonic_tfm = torch.einsum("ij,...j->...i", rot_mtx, harmonic)
        y_tfm_before, harmonic_tfm_before = function(x_tfm)
        y_tfm_after = y.transform(g)

        equiv_error = torch.linalg.norm(y_tfm_after.tensor - y_tfm_before.tensor)
        print(f'eerror {i} : {equiv_error:.2e}')
        hequiv_error = torch.linalg.norm(harmonic_tfm - harmonic_tfm_before)
        print(f'harmonic error {i} : {hequiv_error:.2e}')

    print('variance ', y_tfm_after.tensor.var())
