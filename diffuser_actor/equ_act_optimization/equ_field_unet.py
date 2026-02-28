import copy
import math

import e3nn.o3
import einops
import torch
import torch.nn as nn
from torch_cluster import radius, knn, fps
from typing import List, Optional, Union, Tuple, Iterable, Dict
import utils.pytorch3d_transforms as torch3d_tf
from diffuser_actor.equ_act_optimization.equiformer_v2.up_down_sampling import CoefficientMaxPool, interpolation
from diffuser_actor.utils.layers import FFWRelativeCrossAttentionModule
from diffuser_actor.utils.position_encodings import RotaryPositionEncoding3D

try:
    from e3nn import o3
except ImportError:
    pass

from diffuser_actor.utils.utils import (
    sample_ghost_points_uniform_cube,
    sample_ghost_points_uniform_sphere
)
from diffuser_actor.equ_act_optimization.equiformer_v2.gaussian_rbf import GaussianRadialBasisLayer, \
    GaussianRadialBasisLayerFiniteCutoff
from diffuser_actor.equ_act_optimization.equiformer_v2.edge_rot_mat import init_edge_rot_mat2
from diffuser_actor.equ_act_optimization.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from diffuser_actor.equ_act_optimization.equiformer_v2.module_list import ModuleListInfo
from diffuser_actor.equ_act_optimization.equiformer_v2.radial_function import RadialFunction
from diffuser_actor.equ_act_optimization.equiformer_v2.layer_norm import (
    get_normalization_layer
)
from diffuser_actor.equ_act_optimization.equiformer_v2.equiformerv2_block import (
    SO2EquivariantGraphAttention,
    TransBlock,
)
from diffuser_actor.equ_act_optimization.transformer_enc import Transformer
from diffuser_actor.equ_act_optimization.equiformer_v2.connectivity import RadiusGraph, FpsPool, FpsKnnPool, \
    URandKnnPool, KnnGraph
from diffuser_actor.equ_act_optimization.spherical_conv_utils import *

debug = False


class EquiformerV2Block(nn.Module):
    def __init__(self, c_in, c_out, scale, num_resolutions, num_distance_basis, distance_function, edge_channels,
                 max_neighbors, max_radius,
                 attn_hidden_channels, num_heads, attn_alpha_channels, attn_value_channels, ffn_hidden_channels,
                 lmax_list, mmax_list, SO3_rotation, mappingReduced, SO3_grid, use_m_share_rad, use_attn_renorm,
                 use_grid_mlp, use_sep_s2_act, norm_type, alpha_drop, drop_path_rate, proj_drop, film_dim, film_type):
        super().__init__()
        assert film_type in ['iFiLM', 'FiLM']
        self.film_type = film_type
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        edge_channels_list = [int(num_distance_basis)] + [edge_channels] * 2
        self.num_resolutions = num_resolutions
        self.SO3_rotation = SO3_rotation
        self.block = torch.nn.ModuleDict()
        self.block['graph'] = KnnGraph(r=3, max_num_neighbors=max_neighbors)

        # Initialize the function used to measure the distances between atoms
        if distance_function == 'gaussian':
            self.block['distance_expansion'] = GaussianRadialBasisLayer(num_basis=num_distance_basis,
                                                                        cutoff=max_radius)
        elif distance_function == 'gaussian_soft':
            self.block['distance_expansion'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=num_distance_basis,
                                                                                    cutoff=max_radius * 0.99)
        else:
            raise ValueError

        if debug:
            print('{}->{} channels'.format(c_in, c_out))
        self.block['transblock'] = TransBlock(
            sphere_channels=c_in,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,
            output_channels=c_out,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            edge_channels_list=edge_channels_list,
            use_m_share_rad=use_m_share_rad,
            use_attn_renorm=use_attn_renorm,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            norm_type=norm_type,
            alpha_drop=alpha_drop,
            drop_path_rate=drop_path_rate,
            proj_drop=proj_drop,
            film_dim=film_dim if film_type == 'iFiLM' else 0
        )
        self.scale = scale
        if self.scale < 1:
            self.pool = CoefficientMaxPool(lmax=lmax_list[0])
        if film_type == 'FiLM':
            self.weight_dim = c_out * (lmax_list[0] + 1) ** 2
            self.c_out = c_out
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(film_dim, 2 * self.weight_dim),
            )

    def forward(self, node_coord, node_dst, batch, condition=None, up_coord=None, up_batch=None):
        node_coord, node_dst, batch = node_coord.clone(), node_dst.clone(), batch.clone()
        if self.scale < 1:
            node_dst_idx = fps(src=node_coord, batch=batch, ratio=self.scale)
            node_dst_idx = torch.unique(node_dst_idx)
            node_coord_dst, batch_dst = node_coord[node_dst_idx], batch[node_dst_idx]
            edge_dst, edge_src = knn(node_coord, node_coord_dst, 16, batch_x=batch, batch_y=batch_dst)
            knn_node_src = copy.copy(node_dst)
            knn_node_src.embedding = knn_node_src.embedding[edge_src]
            knn_node_src.embedding = einops.rearrange(knn_node_src.embedding, '(b n) i c -> b n i c', n=16)  # [b n i c]
            node_dst.embedding = self.pool(knn_node_src.embedding)  # in shape [b i c]
            node_coord = node_coord_dst
            batch = batch_dst
        elif self.scale > 1:
            node_dst.embedding = interpolation(node_coord.contiguous(), up_coord.contiguous(),
                                               node_dst.embedding.contiguous(), batch, up_batch)
            node_coord = up_coord
            batch = up_batch

        node_dst, node_coord_dst, edge_src, edge_dst, degree, batch_dst = \
            self.block['graph'](node_coord_src=node_coord,
                                node_feature_src=node_dst,
                                batch_src=batch)

        edge_vec = node_coord.index_select(0, edge_src) - node_coord_dst.index_select(0, edge_dst)
        edge_vec = edge_vec.detach()
        edge_length = torch.norm(edge_vec, dim=-1).detach()

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = init_edge_rot_mat2(edge_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        edge_attr = self.block['distance_expansion'](edge_length)
        node_dst = self.block['transblock'](node_dst,
                                            node_dst,
                                            edge_attr,
                                            edge_src,
                                            edge_dst,
                                            batch=batch,
                                            condition=condition)
        if self.film_type == 'FiLM':
            film = self.cond_encoder(condition)
            weight, bias = film[:, :self.weight_dim], film[:, self.weight_dim:]
            c = node_dst.embedding  # (b npts) irrep c_out
            c = einops.rearrange(c, '(b n) i c -> b n (i c)', b=condition.shape[0])
            c = c * weight[:, None, :] + bias[:, None, :]
            node_dst.embedding = einops.rearrange(c, 'b n (i c) -> (b n) i c', c=self.c_out)
        if debug:
            print('down embedding', node_dst.embedding.shape)
        return node_coord, node_dst, batch


class EquFieldUnet(nn.Module):

    def __init__(
            self,
            max_neighbors=(16, 16, 16, 16),
            max_radius=(0.2, 0.4, 0.8, 1.6),  # for knn
            pool_ratio=(1, 0.25, 0.25, 0.25),  # 2048, 512, 128, 32
            sphere_channels=(32, 32, 64, 128, 256),
            attn_hidden_channels=(16, 16, 32, 64),
            num_heads=4,
            attn_alpha_channels=(8, 8, 16, 32),
            attn_value_channels=(4, 4, 8, 16),
            ffn_hidden_channels=(64, 64, 128, 256),
            edge_channels=(16, 16, 32, 64),
            num_distance_basis=(128, 128, 128, 128),
            alpha_drop=(0.1, 0.1, 0.1, 0.1),
            drop_path_rate=(0., 0., 0., 0.),
            proj_drop=(0.1, 0.1, 0.1, 0.1),
            norm_type='rms_norm_sh',
            deterministic=False,
            lmax_list=[3],
            mmax_list=[2],
            grid_resolution=None,
            use_m_share_rad=False,
            distance_function="gaussian_soft",
            use_attn_renorm=True,
            use_grid_mlp=False,
            use_sep_s2_act=True,
            weight_init='normal',
            healpix_grid=None,  # SO(3) equal distance grid
            rot_mats=None,  # rotation matrix for the grid
            rot_lmax=[4],
            rot_mmax=[3],
            field_net='equiformer',  # 'equiformer' 'roformer'
            include_proprio=True,
            n_gripper_pos=10,
            film_dim=128,
            film_type='iFiLM',
            field_nn_c=64
    ):
        super().__init__()
        # -----------------------------------EquiformerV2 GNN Unet--------------------------------
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

        self.deterministic = deterministic
        self.device = torch.cuda.current_device()
        self.n_scales = len(self.max_radius)
        self.num_resolutions = len(self.lmax_list)
        self.pcd_channels = 3
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels[0]
        self.include_proprio = include_proprio
        self.n_gripper_pos = n_gripper_pos
        self.film_dim = film_dim
        self.field_nn_c = field_nn_c

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

        # Down Blocks
        self.down_blocks = torch.nn.ModuleList()
        for n in range(self.n_scales):
            block = EquiformerV2Block(c_in=self.sphere_channels[n],
                                      c_out=self.sphere_channels[n + 1],
                                      scale=self.pool_ratio[n],
                                      num_resolutions=self.num_resolutions,
                                      num_distance_basis=self.num_distance_basis[n],
                                      distance_function=self.distance_function,
                                      edge_channels=self.edge_channels[n],
                                      max_neighbors=self.max_neighbors[n],
                                      max_radius=self.max_radius[n],
                                      attn_hidden_channels=self.attn_hidden_channels[n],
                                      num_heads=self.num_heads,
                                      attn_alpha_channels=self.attn_alpha_channels[n],
                                      attn_value_channels=self.attn_value_channels[n],
                                      ffn_hidden_channels=self.ffn_hidden_channels[n],
                                      lmax_list=self.lmax_list,
                                      mmax_list=self.mmax_list,
                                      SO3_rotation=self.SO3_rotation,
                                      mappingReduced=self.mappingReduced,
                                      SO3_grid=self.SO3_grid,
                                      use_m_share_rad=self.use_m_share_rad,
                                      use_attn_renorm=self.use_attn_renorm,
                                      use_grid_mlp=self.use_grid_mlp,
                                      use_sep_s2_act=self.use_sep_s2_act,
                                      norm_type=self.norm_type,
                                      alpha_drop=self.alpha_drop,
                                      drop_path_rate=self.drop_path_rate,
                                      proj_drop=self.proj_drop,
                                      film_dim=self.film_dim,
                                      film_type=film_type
                                      )
            self.down_blocks.append(block)

        ## Middle Blocks
        self.middle_block = EquiformerV2Block(c_in=self.sphere_channels[-1],
                                              c_out=self.sphere_channels[-1],
                                              scale=self.pool_ratio[-1],
                                              num_resolutions=self.num_resolutions,
                                              num_distance_basis=self.num_distance_basis[-1],
                                              distance_function=self.distance_function,
                                              edge_channels=self.edge_channels[-1],
                                              max_neighbors=self.max_neighbors[-1],
                                              max_radius=self.max_radius[-1],
                                              attn_hidden_channels=self.attn_hidden_channels[-1],
                                              num_heads=self.num_heads,
                                              attn_alpha_channels=self.attn_alpha_channels[-1],
                                              attn_value_channels=self.attn_value_channels[-1],
                                              ffn_hidden_channels=self.ffn_hidden_channels[-1],
                                              lmax_list=self.lmax_list,
                                              mmax_list=self.mmax_list,
                                              SO3_rotation=self.SO3_rotation,
                                              mappingReduced=self.mappingReduced,
                                              SO3_grid=self.SO3_grid,
                                              use_m_share_rad=self.use_m_share_rad,
                                              use_attn_renorm=self.use_attn_renorm,
                                              use_grid_mlp=self.use_grid_mlp,
                                              use_sep_s2_act=self.use_sep_s2_act,
                                              norm_type=self.norm_type,
                                              alpha_drop=self.alpha_drop,
                                              drop_path_rate=self.drop_path_rate,
                                              proj_drop=self.proj_drop,
                                              film_dim=self.film_dim,
                                              film_type=film_type
                                              )

        ## Up Blocks
        self.up_blocks = torch.nn.ModuleList()
        for n in range(self.n_scales - 1, -1, -1):
            block = EquiformerV2Block(c_in=self.sphere_channels[n + 1],
                                      c_out=self.sphere_channels[n] if n != 0 else self.field_nn_c,
                                      scale=1 / self.pool_ratio[n],
                                      num_resolutions=self.num_resolutions,
                                      num_distance_basis=self.num_distance_basis[n],
                                      distance_function=self.distance_function,
                                      edge_channels=self.edge_channels[n],
                                      max_neighbors=self.max_neighbors[n],
                                      max_radius=self.max_radius[n],
                                      attn_hidden_channels=self.attn_hidden_channels[n],
                                      num_heads=self.num_heads,
                                      attn_alpha_channels=self.attn_alpha_channels[n],
                                      attn_value_channels=self.attn_value_channels[n],
                                      ffn_hidden_channels=self.ffn_hidden_channels[n],
                                      lmax_list=self.lmax_list,
                                      mmax_list=self.mmax_list,
                                      SO3_rotation=self.SO3_rotation,
                                      mappingReduced=self.mappingReduced,
                                      SO3_grid=self.SO3_grid,
                                      use_m_share_rad=self.use_m_share_rad,
                                      use_attn_renorm=self.use_attn_renorm,
                                      use_grid_mlp=self.use_grid_mlp,
                                      use_sep_s2_act=self.use_sep_s2_act,
                                      norm_type=self.norm_type,
                                      alpha_drop=self.alpha_drop,
                                      drop_path_rate=self.drop_path_rate,
                                      proj_drop=self.proj_drop,
                                      film_dim=self.film_dim,
                                      film_type=film_type
                                      )
            self.up_blocks.append(block)

        # Output blocks for point cloud features
        self.norm_1 = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list),
                                              num_channels=self.field_nn_c)

        # --------------------------------Field Net (Query)--------------------
        self.query_r = 2  # ZXP the radius covers entire workspace
        self.query_num_distance_basis = 1024
        # self.field_c = field_c
        if self.distance_function == 'gaussian':
            self.field_distance_expansion = GaussianRadialBasisLayer(num_basis=self.query_num_distance_basis,
                                                                     cutoff=self.query_r)
        elif self.distance_function == 'gaussian_soft':
            self.field_distance_expansion = GaussianRadialBasisLayerFiniteCutoff(
                num_basis=self.query_num_distance_basis,
                cutoff=self.query_r * 0.99)
        else:
            raise ValueError
        edge_channels_list = [int(self.query_num_distance_basis)] + [self.edge_channels[1]] * 2
        if debug:
            print('field_edge_channel', edge_channels_list)
        # ------------trans field-------------
        self.trans_query_max_neighbors = 300
        self.trans_query_r = 2
        self.trans_field_in_c = field_nn_c
        # self.trans_field_in_c = 60  # for roformer
        self.trans_lmax = [1]
        self.trans_mmax = [1]
        num_attn_heads = 5
        n_trans_layer = 1
        self.field_net = field_net
        if self.field_net == 'equiformer':
            # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
            self.trans_SO3_rotation = nn.ModuleList()
            for i in range(self.num_resolutions):
                self.trans_SO3_rotation.append(SO3_Rotation(self.trans_lmax[i]))
            # Initialize conversion between degree l and order m layouts
            self.trans_mappingReduced = CoefficientMappingModule(self.trans_lmax, self.trans_mmax)
            # Initialize the transformations between spherical and grid representations
            self.trans_SO3_grid = ModuleListInfo('({}, {})'.format(max(self.trans_lmax), max(self.trans_lmax)))
            for l in range(max(self.trans_lmax) + 1):
                trans_SO3_m_grid = nn.ModuleList()
                for m in range(max(self.trans_lmax) + 1):
                    trans_SO3_m_grid.append(
                        SO3_Grid(
                            l,
                            m,
                            resolution=self.grid_resolution,
                            normalization='component'
                        )
                    )
                self.trans_SO3_grid.append(trans_SO3_m_grid)
            self.trans_field_layer = nn.ModuleList()
            for i in range(n_trans_layer):
                self.trans_field_layer.append(SO2EquivariantGraphAttention(
                    sphere_channels=self.trans_field_in_c,
                    hidden_channels=self.attn_hidden_channels[1],
                    num_heads=self.num_heads,
                    attn_alpha_channels=self.attn_alpha_channels[1],
                    attn_value_channels=self.attn_value_channels[1],
                    output_channels=self.trans_field_in_c,  # ZXP field_out for Q_trans
                    lmax_list=self.trans_lmax,
                    mmax_list=self.trans_mmax,
                    SO3_rotation=self.trans_SO3_rotation,
                    mappingReduced=self.trans_mappingReduced,
                    SO3_grid=self.trans_SO3_grid,
                    edge_channels_list=edge_channels_list,
                    use_m_share_rad=self.use_m_share_rad,
                    use_attn_renorm=self.use_attn_renorm,
                    use_sep_s2_act=self.use_sep_s2_act,
                    alpha_drop=self.alpha_drop
                ))
        elif self.field_net == 'roformer':
            self.trans_roformer_emb_dim = self.trans_field_in_c * (self.trans_lmax[0] + 1) ** 2
            # 3D relative positional embeddings
            self.trans_relative_pe_layer = RotaryPositionEncoding3D(self.trans_roformer_emb_dim)
            self.trans_field_layer = FFWRelativeCrossAttentionModule(self.trans_roformer_emb_dim, num_attn_heads,
                                                                     num_layers=1, use_adaln=False)

        # -------------rot field--------------
        # self.rot_field_in_c = 60  # for roformer
        self.rot_field_in_c = field_nn_c
        self.rot_query_max_neighbors = 10000
        self.rot_query_r = 2
        self.rot_lmax = rot_lmax
        n_rot_layer = 1
        if self.field_net == 'equiformer':
            self.rot_mmax = rot_mmax
            # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
            self.rot_SO3_rotation = nn.ModuleList()
            for i in range(self.num_resolutions):
                self.rot_SO3_rotation.append(SO3_Rotation(self.rot_lmax[i]))
            # Initialize conversion between degree l and order m layouts
            self.rot_mappingReduced = CoefficientMappingModule(self.rot_lmax, self.rot_mmax)
            # Initialize the transformations between spherical and grid representations
            self.rot_SO3_grid = ModuleListInfo('({}, {})'.format(max(self.rot_lmax), max(self.rot_lmax)))
            for l in range(max(self.rot_lmax) + 1):
                rot_SO3_m_grid = nn.ModuleList()
                for m in range(max(self.rot_lmax) + 1):
                    rot_SO3_m_grid.append(
                        SO3_Grid(
                            l,
                            m,
                            resolution=None,
                            normalization='component'
                        )
                    )
                self.rot_SO3_grid.append(rot_SO3_m_grid)

            self.rot_field_layer = nn.ModuleList()
            for i in range(n_rot_layer):
                self.rot_field_layer.append(TransBlock(
                    sphere_channels=self.rot_field_in_c,
                    attn_hidden_channels=self.attn_hidden_channels[1],
                    num_heads=self.num_heads,
                    attn_alpha_channels=self.attn_alpha_channels[1],
                    attn_value_channels=self.attn_value_channels[1],
                    ffn_hidden_channels=self.ffn_hidden_channels[1],
                    output_channels=self.rot_field_in_c,  # ZXP field_out for Q_rot and Q_open
                    lmax_list=self.rot_lmax,
                    mmax_list=self.rot_mmax,
                    SO3_rotation=self.rot_SO3_rotation,
                    mappingReduced=self.rot_mappingReduced,
                    SO3_grid=self.rot_SO3_grid,
                    edge_channels_list=edge_channels_list,
                    use_m_share_rad=self.use_m_share_rad,
                    use_attn_renorm=self.use_attn_renorm,
                    use_grid_mlp=self.use_grid_mlp,
                    use_sep_s2_act=self.use_sep_s2_act,
                    norm_type=self.norm_type,
                    alpha_drop=self.alpha_drop,
                    drop_path_rate=self.drop_path_rate,
                    proj_drop=self.proj_drop
                ))
        elif self.field_net == 'roformer':
            self.rot_roformer_emb_dim = self.rot_field_in_c * (self.rot_lmax[0] + 1) ** 2
            # 3D relative positional embeddings
            self.rot_relative_pe_layer = RotaryPositionEncoding3D(self.rot_roformer_emb_dim)
            self.rot_field_layer = FFWRelativeCrossAttentionModule(self.rot_roformer_emb_dim, num_attn_heads,
                                                                   num_layers=1, use_adaln=False)

        # ---------------------------Spherical Convolution----------------------------------
        # modified from https://colab.research.google.com/github/dmklee/image2sphere/blob/main/model_walkthrough.ipynb#scrollTo=wUC3hnVJTG6k
        # s2 filter has global support
        s2_kernel_grid = s2_healpix_grid(max_beta=np.inf, rec_level=1)
        self.s2_conv = S2Conv(self.rot_field_in_c - 1, 1, self.rot_lmax[-1], s2_kernel_grid)
        self.healpix_grid = healpix_grid
        self.rot_mats = rot_mats
        self.register_buffer("output_wigners",
                             flat_wigner(self.rot_lmax[-1], *self.healpix_grid).transpose(0, 1),
                             persistent=False)

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

        # ------------------------------------Transformer encoder---------------------------------
        # self.scene_norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list),
        #                                           num_channels=self.sphere_channels[-1])
        self.instruction_embedding = nn.Linear(512, self.film_dim)
        self.readout = nn.Parameter(torch.randn(1, 1, self.film_dim))
        # dim, depth, heads, dim_head, mlp_dim
        self.scene_instruction_encoder = Transformer(self.film_dim, 4, 6, 64, 256)

    def encode(self, xyz, rgb, curr_gripper, instruction, verbose=False):
        """
        Arguments:
            xyz: (batch, npts, 3) in robot coordinates
            rgb: (batch, npts, 3) in [0, 1]
            curr_gripper: (batch, 8) in xyzrpyg
            instruction: (batch, max_instruction_length, 512)
        -> embedded SH graph, where features are in shape (b query) (lmax+1)**2 c, signal over S2 irreps
        """
        self.dtype = xyz.dtype
        self.device = xyz.device

        if self.include_proprio:
            # The orientaiton are represented by 3 type1, correspond to XYZ axes of the gripper frame in WS.
            gripper_mat = torch3d_tf.quaternion_to_matrix(curr_gripper[:, 3:7][:, [3, 0, 1, 2]])  # type1^T
            gripper_mat = gripper_mat[:, None, :, :]  # b 1 3 3
            pos_eps = sample_ghost_points_uniform_sphere(
                center=np.array([0, 0, 0]),
                radius=1e-5,
                bounds=np.array([3 * [-1e-5, ], 3 * [1e-5, ]]),
                num_points=xyz.shape[0] * self.n_gripper_pos)
            # The position are represented by n points at the gripper pos. Adding eps to avoid 0-length edge.
            gripper_pos = curr_gripper[:, None, :3].repeat(1, self.n_gripper_pos, 1)
            gripper_pos += torch.tensor(pos_eps, device=self.device).reshape(gripper_pos.shape)
            xyz[:, -self.n_gripper_pos:, :] = gripper_pos
            rgb[:, -self.n_gripper_pos:, :] = (curr_gripper[:, None, -1:] - 0.5) * 10
            # # visualize current gripper pose
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # visualize_xyz = xyz[0]
            # visualize_rgb = rgb[0]
            # visualize_rgb[-10:, 0] = 1
            # pcd.points = o3d.utility.Vector3dVector(visualize_xyz.detach().cpu().numpy())
            # pcd.colors = o3d.utility.Vector3dVector(visualize_rgb.detach().cpu().numpy())
            #
            # # Visualize the gripper pose
            # gripper_pose = visualize_xyz[-1:, :].clone().repeat(6, 1)
            # r = 0.1
            # gripper_pose[:3] += r * gripper_mat[0, 0].permute(1, 0)
            # gripper_pose[3:] -= r * gripper_mat[0, 0].permute(1, 0)
            # gripper_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
            # gripper_color = torch.zeros((3, 3))
            # gripper_color[torch.arange(3), torch.arange(3)] = 1
            # gripper_coord = o3d.geometry.LineSet()
            # gripper_coord.points = o3d.utility.Vector3dVector(gripper_pose.detach().cpu().numpy())
            # gripper_coord.lines = o3d.utility.Vector2iVector(gripper_edge.detach().cpu().numpy())
            # gripper_coord.colors = o3d.utility.Vector3dVector(gripper_color.detach().cpu().numpy())
            #
            # # Visualize the point cloud
            # o3d.visualization.draw_geometries([pcd, gripper_coord])

        node_coord = xyz.reshape(-1, 3)  # (b npts) 3
        total_points, batch_size, num_points = node_coord.shape[0], xyz.shape[0], xyz.shape[1]
        node_feature = rgb.reshape(-1, 3) - 0.5  # (b npts) 3
        batch = torch.arange(0, batch_size).repeat_interleave(xyz.shape[1]).to(self.device)  # (b npts)

        ########## Scene Instruction Encoding##########
        # b max_inst_len sphere_channels[-1]
        instruction = self.instruction_embedding(instruction.reshape(-1, 512)).reshape(batch_size, -1, self.film_dim)
        # # b 1 sphere_channels[-1], signal over S2 l=0 irreps
        # scene = node_src.embedding[:, 0, :].reshape(batch_size, -1, self.sphere_channels[-1]).mean(dim=1, keepdim=True)
        # # # reduce variance of node_src
        # scene = scene / 10
        # scene[:] = 0
        # b max_inst_len+1 sphere_channels[-1]
        scene_instruction_token = torch.cat((self.readout.repeat(batch_size, 1, 1), instruction), dim=1)
        # b sphere_channels[-1], condition for the FiLM layer
        condition = self.scene_instruction_encoder(scene_instruction_token)[:, 0, :]

        node_src = SO3_Embedding(
            total_points,
            self.lmax_list,
            self.sphere_channels[0],
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
        if self.include_proprio:
            for b in range(batch_size):
                batch_end = num_points * (b + 1)
                # [n_gripper_pos 3 3]
                node_src.embedding[batch_end - self.n_gripper_pos:batch_end, 1:4, :3] = gripper_mat[b] * 10

        ########### Downstream Block #############
        features = []
        for n, block in enumerate(self.down_blocks):
            #### Downsampling ####
            feature = block(node_coord, node_src, batch, condition=condition)
            features.append(feature)
            node_coord, node_src, batch = feature

        ################# Middle Block ################
        features.pop()
        feature = self.middle_block(node_coord, node_src, batch, condition=condition)
        node_coord, node_src, batch = feature

        ########### Upstream Block #############
        for n, block in enumerate(self.up_blocks):
            if n < self.n_scales - 1:
                skip_node_coord, skip_node_src, skip_batch = features.pop()
            else:
                skip_node_coord, skip_node_src, skip_batch = None, None, None
            node_coord, node_src, batch = block(node_coord, node_src, batch, condition=condition,
                                                up_coord=skip_node_coord, up_batch=skip_batch)
            if n < self.n_scales - 1:
                node_src.embedding = node_src.embedding + skip_node_src.embedding

        # Final layer (no upsampling)
        node_src.embedding = self.norm_1(node_src.embedding)
        features.append(('final', node_src, node_coord))

        embedded_graph = (node_src, node_coord, batch)
        if verbose:
            return embedded_graph, features
        else:
            return embedded_graph

    def trans_query(self, query_xyz_coordinate, embedded_graph):
        # query_xyz_coordinate: b npts 3

        ###############################################################
        # using query_xyz to build graph for querying (field network)
        # pcd to graph (1 query to all vertex in the embedded_graph)
        ###############################################################
        node_src, node_coord_src, batch = embedded_graph
        node_src = node_src.clone()
        node_src.set_embedding(node_src.embedding[:, :(self.trans_lmax[0] + 1) ** 2, :self.trans_field_in_c])

        n_query = query_xyz_coordinate.shape[1]
        b_query = query_xyz_coordinate.shape[0]
        query_coord = query_xyz_coordinate.reshape(-1, 3)
        query_batch = torch.arange(0, b_query).repeat_interleave(n_query).to(batch.device)
        # edge in shape (2 (b nqeury nneighbor))
        edge_r = radius(x=node_coord_src, y=query_coord, r=self.trans_query_r,
                        batch_x=batch, batch_y=query_batch,
                        max_num_neighbors=self.trans_query_max_neighbors - 100)
        edge_knn = knn(x=node_coord_src, y=query_coord,
                       batch_x=batch, batch_y=query_batch,
                       k=100)
        edge_r = edge_r.reshape(2, b_query, n_query, -1)
        edge_knn = edge_knn.reshape(2, b_query, n_query, -1)
        edge = torch.cat((edge_r, edge_knn), dim=-1).reshape(2, -1)
        # edge = radius(x=node_coord_src, y=query_coord, r=self.query_r,
        #               batch_x=batch, batch_y=query_batch,
        #               max_num_neighbors=self.trans_query_max_neighbors)
        edge_dst, edge_src = edge[0], edge[1]
        # notice that edge_distance_vec is dst->src (destination, source), while the output features are for the dst
        edge_vec = node_coord_src.index_select(0, edge_src) - query_coord.index_select(0, edge_dst)
        edge_vec = edge_vec.detach()
        edge_length = torch.norm(edge_vec, dim=-1).detach()

        node_dst = SO3_Embedding(query_batch.shape[0],
                                 self.trans_lmax,
                                 self.trans_field_in_c,
                                 self.device,
                                 self.dtype)

        if self.field_net == 'equiformer':
            ###############################################################
            # Initialize data structures
            ###############################################################
            # Compute 3x3 rotation matrix per edge
            edge_rot_mat = init_edge_rot_mat2(edge_vec)

            # Initialize the WignerD matrices and other values for spherical harmonic calculations
            for i in range(self.num_resolutions):
                self.trans_SO3_rotation[i].set_wigner(edge_rot_mat)

            edge_attr = self.field_distance_expansion(edge_length)
            for layer in self.trans_field_layer:
                node_dst = layer(node_src,
                                 node_dst,
                                 edge_attr,
                                 edge_src,
                                 edge_dst)
            Q_trans = node_dst.embedding[:, 0, 0]

        elif self.field_net == 'roformer':
            flat_node_src = node_src.embedding.reshape(b_query, -1, self.trans_roformer_emb_dim)
            flat_node_dst = node_dst.embedding.reshape(b_query, n_query, self.trans_roformer_emb_dim)
            flat_node_src = flat_node_src.permute(1, 0, 2)  # npt b c
            flat_node_dst = flat_node_dst.permute(1, 0, 2)  # npt b c
            src_pos = self.trans_relative_pe_layer(node_coord_src.reshape(b_query, -1, 3))
            dst_pos = self.trans_relative_pe_layer(query_xyz_coordinate)
            flat_s2_irrep_weight = self.trans_field_layer(
                query=flat_node_dst, value=flat_node_src,
                query_pos=dst_pos, value_pos=src_pos
            )[-1]
            flat_s2_irrep_weight = flat_s2_irrep_weight.permute(1, 0, 2)
            s2_irrep_weight = flat_s2_irrep_weight.reshape(b_query * n_query, -1)
            Q_trans = s2_irrep_weight[:, 0]

        return Q_trans

    def rot_query(self, query_xyz_coordinate, embedded_graph):
        # query_xyz_coordinate: b npts 3

        ###############################################################
        # using query_xyz to build graph for querying (field network)
        # pcd to graph (1 query to all vertex in the embedded_graph)
        ###############################################################
        node_src, node_coord_src, batch = embedded_graph

        n_query = query_xyz_coordinate.shape[1]
        b_query = query_xyz_coordinate.shape[0]
        query_coord = query_xyz_coordinate.reshape(-1, 3)
        query_batch = torch.arange(0, b_query).repeat_interleave(n_query).to(batch.device)
        # # edge in shape (2 (b nqeury nneighbor))
        # edge_r = radius(x=node_coord_src, y=query_coord, r=self.query_r,
        #                 batch_x=batch, batch_y=query_batch,
        #                 max_num_neighbors=self.rot_query_max_neighbors - 100)
        # edge_knn = knn(x=node_coord_src, y=query_coord,
        #                batch_x=batch, batch_y=query_batch,
        #                k=100)
        # edge_r = edge_r.reshape(2, b_query, n_query, -1)
        # edge_knn = edge_knn.reshape(2, b_query, n_query, -1)
        # edge = torch.cat((edge_r, edge_knn), dim=-1).reshape(2, -1)
        edge = radius(x=node_coord_src, y=query_coord, r=self.rot_query_r,
                      batch_x=batch, batch_y=query_batch,
                      max_num_neighbors=self.rot_query_max_neighbors)

        edge_dst, edge_src = edge[0], edge[1]
        # notice that edge_distance_vec is dst->src (destination, source), while the output features are for the dst
        edge_vec = node_coord_src.index_select(0, edge_src) - query_coord.index_select(0, edge_dst)
        edge_vec = edge_vec.detach()
        edge_length = torch.norm(edge_vec, dim=-1).detach()

        lift_node_src = SO3_Embedding(node_src.embedding.shape[0],
                                      self.rot_lmax,
                                      self.rot_field_in_c,
                                      self.device,
                                      self.dtype)
        lmax = min(self.rot_lmax[0], self.lmax_list[0])
        lift_node_src.embedding[:, :(lmax + 1) ** 2, :] = node_src.embedding[:, :(lmax + 1) ** 2, -self.rot_field_in_c:]

        node_dst = SO3_Embedding(query_batch.shape[0],
                                 self.rot_lmax,
                                 self.rot_field_in_c,
                                 self.device,
                                 self.dtype)

        if self.field_net == 'equiformer':
            ###############################################################
            # Initialize data structures
            ###############################################################
            # Compute 3x3 rotation matrix per edge
            edge_rot_mat = init_edge_rot_mat2(edge_vec)

            # Initialize the WignerD matrices and other values for spherical harmonic calculations
            for i in range(self.num_resolutions):
                self.rot_SO3_rotation[i].set_wigner(edge_rot_mat)

            edge_attr = self.field_distance_expansion(edge_length)
            # node_dst = self.rot_field_layer(lift_node_src,
            #                                 node_dst,
            #                                 edge_attr,
            #                                 edge_src,
            #                                 edge_dst)
            for layer in self.rot_field_layer:
                node_dst = layer(lift_node_src,
                                 node_dst,
                                 edge_attr,
                                 edge_src,
                                 edge_dst,
                                 batch=query_batch)

            # ((b query) f_in+1 (lmax+1)**2), signal over S2 irreps
            s2_irrep_weight = node_dst.embedding.permute(0, 2, 1)
        elif self.field_net == 'roformer':
            flat_node_src = lift_node_src.embedding.reshape(b_query, -1, self.rot_roformer_emb_dim)
            flat_node_dst = node_dst.embedding.reshape(b_query, n_query, self.rot_roformer_emb_dim)
            flat_node_src = flat_node_src.permute(1, 0, 2)  # npt b c
            flat_node_dst = flat_node_dst.permute(1, 0, 2)  # npt b c
            src_pos = self.rot_relative_pe_layer(node_coord_src.reshape(b_query, -1, 3))
            dst_pos = self.rot_relative_pe_layer(query_xyz_coordinate)
            flat_s2_irrep_weight = self.rot_field_layer(
                query=flat_node_dst, value=flat_node_src,
                query_pos=dst_pos, value_pos=src_pos
            )[-1]
            flat_s2_irrep_weight = flat_s2_irrep_weight.permute(1, 0, 2)
            s2_irrep_weight = flat_s2_irrep_weight.reshape((b_query * n_query), self.rot_field_in_c, -1)

        # Q_trans/Q_open: (nquery), SO(3) invariant
        s2_feature, Q_open = s2_irrep_weight[:, :-1, :], s2_irrep_weight[:, -1:, 0]
        Q_rot = self.decode_rot(s2_feature)

        # for visualizing query graph
        b0_idx = batch[edge_src] == 0
        query_graph0_nodes = torch.cat((node_coord_src, query_coord[0:1]), dim=0)
        query_graph0_edges = edge[:, b0_idx].permute(1, 0).clone()
        query_graph0_edges[:, 0] = query_graph0_nodes.shape[0] - 1
        query_graph0 = (query_graph0_nodes, query_graph0_edges)
        return Q_rot, Q_open, query_graph0, node_dst

    def decode_rot(self, s2_conv_feature):
        ###############################################################
        # Query field vale; S2->SO(3) lift conv
        ###############################################################
        so3_irrp_weight = []
        for b in range(s2_conv_feature.shape[0]):
            so3_irrp_weight.append(self.s2_conv(s2_conv_feature[b:b + 1]))
        # (b nquery) f_out, (sum_l^L (2*l+1)**2)
        so3_irrp_weight = torch.cat(so3_irrp_weight, dim=0)
        # Q_rot: (b, nhealpix), SO(3) equivariant
        Q_rot = torch.matmul(so3_irrp_weight, self.output_wigners).squeeze(1)
        return Q_rot

    # Initialize the edge rotation matrics
    def init_edge_rot_mat2(self, edge_length_vec):
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


def rot_pcd(pcd, rot_shift_3x3):
    pcd = einops.rearrange(pcd, 'b n d -> b d n')
    # rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(euler_XYZ, "XYZ").unsqueeze(0).repeat(pcd.shape[0], 1, 1)
    rotated_pcd = torch.bmm(rot_shift_3x3.repeat(pcd.shape[0], 1, 1), pcd)
    rotated_pcd = einops.rearrange(rotated_pcd, 'b d n -> b n d')
    return rotated_pcd


def extract_irrp1(s2_embedding):
    forces = s2_embedding.embedding.narrow(1, 1, 3)
    forces = einops.rearrange(forces, 'b d c -> b c d')
    return forces


if __name__ == "__main__":
    device = 'cuda:1'
    torch.manual_seed(0)
    np.random.seed(0)
    atol = 1e-4
    bs = 1
    xyz = torch.rand(bs, 2000, 3) - 0.5
    feats = torch.rand(bs, 2000, 3) - 0.5
    xyz, feats = xyz.to(device), feats.to(device)
    sampled_action_xyz = torch.zeros(bs, 1, 3).to(device)

    from scipy.spatial.transform import Rotation
    from escnn import gspaces
    from escnn import nn as enn

    _group = gspaces.octaOnR3()
    _group_testing_elements = [g for g in _group.testing_elements]
    _g_quat = [g.value for g in _group.testing_elements]
    group_rot_mat = torch.from_numpy(Rotation.from_quat(_g_quat).as_matrix()).float().to(device)
    # for converting to harmonics, need to know Octahedral group elements in SO3
    oct_on_so3 = torch.stack(e3nn.o3.matrix_to_angles(group_rot_mat))
    # print(e3nn.o3.angles_to_matrix(*oct_on_so3))
    # print(e3nn.o3.angles_to_matrix(*oct_on_so3) - group_rot_mat.cpu())

    # model = EquFieldUnet(
    #     max_neighbors=(20, 20),
    #     max_radius=(0.5, 1),
    #     pool_ratio=(0.2, 0.2),
    #     sphere_channels=(8, 16),
    #     attn_hidden_channels=(8, 16),
    #     num_heads=4,
    #     lmax_list=[3],
    #     mmax_list=[2],
    #     rot_lmax=[4],
    #     rot_mmax=[4],
    #     grid_resolution=50,
    #     # None -> 1.2, 10 -> 1.2, 50 -> 1.2
    #     attn_alpha_channels=(16, 32),
    #     attn_value_channels=(8, 16),
    #     ffn_hidden_channels=(8, 16),
    #     edge_channels=(8, 16),
    #     num_distance_basis=(8, 16),
    #     alpha_drop=(0., 0.),
    #     drop_path_rate=(0., 0.),
    #     proj_drop=(0., 0.),
    #     deterministic=True,
    #     healpix_grid=oct_on_so3.cpu(),
    #     rot_mats=group_rot_mat.cpu(),
    # ).to(device)
    model = EquFieldUnet(
        max_radius=(0.1, 0.2, 2),
        alpha_drop=(0., 0., 0.),
        drop_path_rate=(0., 0., 0.),
        proj_drop=(0., 0., 0.),
        deterministic=True,
        pool='fps',
        healpix_grid=oct_on_so3.cpu(),
        rot_mats=group_rot_mat.cpu()
    ).to(device)
    print('total #param: ', model.num_params)
    instruction = torch.zeros(bs, 5, 512).to(device)

    graph, features = model.encode(xyz, feats, None, instruction, verbose=True)
    out = [extract_irrp1(h) for info, h, c in features]
    info, h, c = features[-1]
    out.append(c.unsqueeze(1))
    Q_rot, Q_open, query_graph0, s2_feature = model.rot_query(sampled_action_xyz, graph)
    out.append(extract_irrp1(s2_feature))
    Q_rot_geo = enn.GeometricTensor(Q_rot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                    enn.FieldType(_group, [_group.regular_repr]))

    print([info for info, _, _ in features])
    success = True
    eerr_q = torch.zeros(group_rot_mat.shape[0])
    for i in range(group_rot_mat.shape[0]):
        xyz_tfm = rot_pcd(xyz.clone(), group_rot_mat[i])
        out_feats_tfm_after = [rot_pcd(h.clone(), group_rot_mat[i]) for h in out]
        Q_rot_tfm_after = Q_rot_geo.transform(_group_testing_elements[i]).tensor.reshape(bs, -1)
        out_feats_tfm_after.append(Q_rot_tfm_after)

        graph, embedded_graph_before = model.encode(xyz_tfm, feats, None, instruction, verbose=True)
        out_feats_tfm_before = [extract_irrp1(h) for info, h, c in embedded_graph_before]
        info, h, c = embedded_graph_before[-1]
        out_feats_tfm_before.append(c.unsqueeze(1))
        Q_rot_tfm_before, Q_open, query_graph0, s2_feature = model.rot_query(sampled_action_xyz, graph)
        out_feats_tfm_before.append(extract_irrp1(s2_feature))
        out_feats_tfm_before.append(Q_rot_tfm_before)

        eerr = torch.tensor([torch.round(torch.linalg.norm(before - after, dim=1).max(), decimals=6)
                             for before, after in zip(out_feats_tfm_before, out_feats_tfm_after)])
        print(f"Equ Error at Rotmat {torch.round(group_rot_mat[i].cpu(), decimals=0).reshape(-1).numpy()}: "
              f"{eerr.cpu().numpy()}")
        eerr_q[i] = eerr[-1]
        # q_value = torch.round(torch.stack((out_feats_tfm_before[-1], out_feats_tfm_after[-1]),
        #                                   dim=1).cpu(), decimals=2).detach().numpy()
        # print(f"Q values: {q_value}")
        # s2_irrep1 = torch.round(torch.cat((out_feats_tfm_before[-2][:, 0], out_feats_tfm_after[-2][:, 0]),
        #                                   dim=-1).cpu(), decimals=4).detach().numpy()
        # print(f"s2 irrep 1: {s2_irrep1}")
    print('avg eerror', eerr_q.mean(), 'q std', Q_rot.std())
