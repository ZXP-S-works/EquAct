import math
import einops
import torch
import torch.nn as nn
from torch_cluster import radius, knn
import utils.pytorch3d_transforms as torch3d_tf
from typing import List, Optional, Union, Tuple, Iterable, Dict
from equiformer_v2.gaussian_rbf import GaussianRadialBasisLayer, \
    GaussianRadialBasisLayerFiniteCutoff
from equiformer_v2.edge_rot_mat import init_edge_rot_mat2
from equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from equiformer_v2.module_list import ModuleListInfo
from equiformer_v2.radial_function import RadialFunction
from equiformer_v2.layer_norm import (
    get_normalization_layer
)
from equiformer_v2.equiformerv2_block import (
    TransBlock,
)
from equiformer_v2.connectivity import RadiusGraph, FpsPool, AdaptiveOriginPool, FpsKnnPool
from e3nn.o3 import spherical_harmonics_alpha_beta
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torchvision import models as vision_models
import itertools
import numpy as np
import einops

debug = False


# def compute_spherical_harmonics(vectors, lmax=3):
#     r, beta, alpha = vector_to_spherical(vectors)
#     harmonics_list = []
#     for l in range(lmax + 1):
#         harmonics = spherical_harmonics_alpha_beta(l, alpha, beta, normalization='component')
#         harmonics_list.append(harmonics)
#     harmonics = torch.cat(harmonics_list, dim=-1)
#     return harmonics.detach()


class EquiformerEnc(nn.Module):

    def __init__(
            self,
            max_neighbors=(16, 16, 16, 16, 1024),
            max_radius=(0.06, 0.125, 0.25, 0.5, 2),
            # 1024, 256, 64, 16, 4, 1
            pool_ratio=(1, 0.25, 0.25, 0.25, 0.25),
            # pool_ratio=(1,) * 5,
            sphere_channels=(16, 32, 64, 128, 128),
            attn_hidden_channels=(16, 32, 64, 128, 256),
            num_heads=4,
            attn_alpha_channels=(8, 8, 16, 16, 32),
            attn_value_channels=(4, 4, 8, 8, 16),
            ffn_hidden_channels=(64, 64, 128, 256, 512),

            norm_type='rms_norm_sh',

            lmax_list=[3],
            mmax_list=[2],
            grid_resolution=30,

            edge_channels=(32, 32, 64, 64, 128),
            use_m_share_rad=False,
            distance_function="gaussian_soft",
            num_distance_basis=(64, 128, 128, 256, 512),

            use_attn_renorm=True,
            use_grid_mlp=False,
            use_sep_s2_act=True,

            alpha_drop=0.,
            drop_path_rate=0.,
            proj_drop=0.,

            weight_init='normal',
    ):
        super().__init__()
        # -----------------------------------EquiformerV2 GNN Enc--------------------------------
        self.max_neighbors = max_neighbors
        self.pool_ratio = pool_ratio
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution
        self.edge_channels = edge_channels
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis
        self.use_attn_renorm = use_attn_renorm
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']
        self.max_radius = max_radius
        print('GNN graph radius', self.max_radius)

        self.deterministic = True
        self.device = torch.cuda.current_device()
        self.n_scales = len(self.max_radius)
        self.num_resolutions = len(self.lmax_list)
        self.pcd_channels = 3
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels[0]

        assert self.distance_function in [
            'gaussian', 'gaussian_soft'
        ]

        # Weights for message initialization
        self.type0_linear = nn.Linear(self.pcd_channels, self.sphere_channels_all, bias=True)

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l,
                        m,
                        resolution=self.grid_resolution,
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        ## Down Blocks
        self.down_blocks = torch.nn.ModuleList()
        for n in range(len(self.max_neighbors)):
            # Initialize the sizes of radial functions (input channels and 2 hidden channels)
            edge_channels_list = [int(self.num_distance_basis[n])] + [self.edge_channels[n]] * 2

            block = torch.nn.ModuleDict()
            if n != len(self.max_neighbors) - 1:
                block['pool'] = FpsPool(ratio=self.pool_ratio[n], random_start=not self.deterministic,
                                        r=self.max_radius[n], max_num_neighbors=self.max_neighbors[n])
            else:
                block['pool'] = AdaptiveOriginPool(ratio=self.pool_ratio[n], random_start=not self.deterministic,
                                                   r=self.max_radius[n], max_num_neighbors=self.max_neighbors[n])

            # Initialize the function used to measure the distances between atoms
            if self.distance_function == 'gaussian':
                block['distance_expansion'] = GaussianRadialBasisLayer(num_basis=self.num_distance_basis[n],
                                                                       cutoff=self.max_radius[n])
            elif self.distance_function == 'gaussian_soft':
                block['distance_expansion'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.num_distance_basis[n],
                                                                                   cutoff=self.max_radius[n] * 0.99)
            else:
                raise ValueError

            scale_out_sphere_channels = self.sphere_channels[min(n + 1, self.n_scales - 1)]
            if debug:
                print('down block {}, {}->{} channels'.format(n, self.sphere_channels[n], scale_out_sphere_channels))
            block['transblock'] = TransBlock(
                sphere_channels=self.sphere_channels[n],
                attn_hidden_channels=self.attn_hidden_channels[n],
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels[n],
                attn_value_channels=self.attn_value_channels[n],
                ffn_hidden_channels=self.ffn_hidden_channels[n],
                output_channels=scale_out_sphere_channels,
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                edge_channels_list=edge_channels_list,
                use_m_share_rad=self.use_m_share_rad,
                use_attn_renorm=self.use_attn_renorm,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop,
                drop_path_rate=self.drop_path_rate,
                proj_drop=self.proj_drop
            )

            self.down_blocks.append(block)

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)
        ####### ToDo:@Boce implement s2 -> c4 feature convertion by wingler D matrix ############

    def forward(self, pcd):
        """
        Arguments:
            pcd: (batch, npts, 6) in [-0.5, 0.5]
        -> embedded SH graph, where features are in shape (b (lmax+1)**2 c), signal over S2 irreps
        """
        xyz, rgb = pcd[:, :, :3], pcd[:, :, 3:]
        self.dtype = xyz.dtype
        self.device = xyz.device

        node_coord = xyz.reshape(-1, 3)  # (b npts) 3
        num_points, batch_size = node_coord.shape[0], xyz.shape[0]
        node_feature = rgb.reshape(-1, 3)  # (b npts) 3
        batch = torch.arange(0, xyz.shape[0]).repeat_interleave(xyz.shape[1]).to(self.device)  # (b npts)

        node_src = None
        node_dst = None
        ########### Downstream Block #############
        for n, block in enumerate(self.down_blocks):
            #### Downsampling ####
            pool_graph = block['pool'](node_coord_src=node_coord, batch_src=batch)
            node_coord_dst, edge_src, edge_dst, degree, batch_dst, node_idx = pool_graph

            edge_vec = node_coord.index_select(0, edge_src) - node_coord_dst.index_select(0, edge_dst)
            edge_vec = edge_vec.detach()
            edge_length = torch.norm(edge_vec, dim=-1).detach()

            # Compute 3x3 rotation matrix per edge
            edge_rot_mat = self._init_edge_rot_mat(edge_vec)

            # Initialize the WignerD matrices and other values for spherical harmonic calculations
            for i in range(self.num_resolutions):
                self.SO3_rotation[i].set_wigner(edge_rot_mat)

            if node_src is None:
                node_src = SO3_Embedding(
                    num_points,
                    self.lmax_list,
                    self.sphere_channels[n],
                    self.device,
                    self.dtype,
                )

                offset_res = 0
                offset = 0
                # Initialize the l = 0, m = 0 coefficients for each resolution
                for i in range(self.num_resolutions):
                    if self.num_resolutions == 1:
                        node_src.embedding[:, offset_res, :] = self.type0_linear(node_feature)
                    else:
                        node_src.embedding[:, offset_res, :] = self.type0_linear(node_feature)[:,
                                                               offset: offset + self.sphere_channels[0]]
                    offset = offset + self.sphere_channels[0]
                    offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

            # Edge encoding (distance and atom edge)
            edge_attr = block['distance_expansion'](edge_length)
            node_dst = SO3_Embedding(
                batch_size,
                self.lmax_list,
                self.sphere_channels[n],
                self.device,
                self.dtype,
            )
            if n != len(self.max_neighbors) - 1:
                node_dst.set_embedding(node_src.embedding[node_idx])
            node_dst.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())
            node_dst = block['transblock'](node_src,
                                           node_dst,
                                           edge_attr,
                                           edge_src,
                                           edge_dst,
                                           batch=batch)
            node_src = node_dst
            node_coord_src = node_coord.clone()
            node_coord = node_coord_dst
            batch = batch_dst
            if debug:
                print('down embedding', node_src.embedding.shape)

        ########### ToDo: Extract C4 Regular Representation from Irrps ##########

        ####### Currently outputs irrp1 (vector) ###########
        forces = node_dst.embedding.narrow(1, 1, 3)
        forces = einops.rearrange(forces, 'b d c -> b c d', b=batch_size)

        return forces

    def _init_edge_rot_mat(self, edge_length_vec):
        # return init_edge_rot_mat(edge_length_vec)
        return init_edge_rot_mat2(edge_length_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
                or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)


def rot_pcd(pcd, euler_XYZ):
    pcd = einops.rearrange(pcd, 'b n d -> b d n')
    rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(euler_XYZ, "XYZ").unsqueeze(0).repeat(pcd.shape[0], 1, 1)
    rotated_pcd = torch.bmm(rot_shift_3x3, pcd)
    rotated_pcd = einops.rearrange(rotated_pcd, 'b d n -> b n d')
    return rotated_pcd


if __name__ == "__main__":
    device = 'cuda:1'
    atol = 1e-2
    bs = 32
    xyz = torch.rand(bs, 1024, 3) - 0.5
    feats = torch.rand(bs, 1024, 3) - 0.5
    xyz, feats = xyz.to(device), feats.to(device)
    model = EquiformerEnc().to(device)
    c4_rots = torch.zeros((4, 3)).to(device)
    c4_rots[:, -1] = torch.arange(4) * np.pi / 2
    print('total #param: ', model.num_params)

    with torch.no_grad():
        out = model(torch.cat([xyz, feats], dim=-1))

    success = True
    for i in range(c4_rots.shape[0]):
        xyz_tfm = rot_pcd(xyz.clone(), c4_rots[i])
        out_feats_tfm_after = rot_pcd(out.clone(), c4_rots[i])

        with torch.no_grad():
            out_feats_tfm_before = model(torch.cat([xyz_tfm, feats], dim=-1))

        eerr = torch.linalg.norm(out_feats_tfm_before - out_feats_tfm_after, dim=1).max()
        err = torch.linalg.norm(out_feats_tfm_after - out, dim=1).max()
        if not torch.allclose(out_feats_tfm_before, out_feats_tfm_after, atol=atol):
            print(f"FAILED on {c4_rots[i]}: {eerr:.1E} > {atol}, {err}")
            success = False
        else:
            print(f"PASSED on {c4_rots[i]}: {eerr:.1E} < {atol}, {err}")

    # import matplotlib.pyplot as plt
    # f = plt.figure(figsize=(16, 4))
    # ax = [f.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
    # plt.show()

    if success:
        print('PASSED')
    print(1)
