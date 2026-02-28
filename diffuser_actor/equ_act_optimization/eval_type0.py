from copy import copy

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
import torch.nn as nn
import torch
import torch.nn as nn
import math
import torch_geometric
import copy

from diffuser_actor.equ_act_optimization.equiformer_v2.radial_function import RadialFunction
from diffuser_actor.equ_act_optimization.equiformer_v2.so2_ops import SO2_Convolution
from diffuser_actor.equ_act_optimization.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2, SO3_FiLM_Linear
from diffuser_actor.equ_act_optimization.equiformer_v2.drop import GraphDropPath, \
    EquivariantDropoutArraySphericalHarmonics
from diffuser_actor.equ_act_optimization.equiformer_v2.activation import (
    SmoothLeakyReLU,
    SeparableS2Activation,
    S2Activation
)
from diffuser_actor.equ_act_optimization.equiformer_v2.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    get_normalization_layer
)


class EvalSO2EquivariantGraphAttention(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (ModuleList,list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].

        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
            self,
            sphere_channels,  # 128
            hidden_channels,  # 64
            num_heads,  # 8
            attn_alpha_channels,  # 64
            attn_value_channels,  # 16
            output_channels,  # 128
            lmax_list,  # [6]
            mmax_list,  # [3]
            SO3_rotation,
            mappingReduced,
            SO3_grid,
            edge_channels_list,
            use_m_share_rad=False,
            use_attn_renorm=True,
            use_sep_s2_act=True,
            alpha_drop=0.0,
    ):
        super(EvalSO2EquivariantGraphAttention, self).__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid

        # Create edge scalar (invariant to rotations) features

        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_m_share_rad = use_m_share_rad

        self.source_embedding, self.target_embedding = None, None
        self.use_attn_renorm = use_attn_renorm
        self.use_sep_s2_act = use_sep_s2_act

        # Create SO(2) convolution blocks
        extra_m0_output_channels = self.num_heads * self.attn_alpha_channels  # 8 * 64
        if self.use_sep_s2_act:
            extra_m0_output_channels = extra_m0_output_channels + self.hidden_channels

        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [2 * self.sphere_channels * (max(self.lmax_list) + 1)]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for l in range(max(self.lmax_list) + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx: (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(
                False if not self.use_m_share_rad
                else True
            ),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad
                else None
            ),
            extra_m0_output_channels=extra_m0_output_channels  # for attention weights and/or gate activation
        )

        if self.use_attn_renorm:
            self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
        else:
            self.alpha_norm = torch.nn.Identity()
        self.alpha_act = SmoothLeakyReLU()
        self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
        # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        std = 1.0 / math.sqrt(self.attn_alpha_channels)
        torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_sep_s2_act:
            # separable S2 activation
            self.s2_act = SeparableS2Activation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list)
            )
        else:
            # S2 activation
            self.s2_act = S2Activation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list)
            )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None  # for attention weights
        )

        self.proj = SO3_LinearV2(self.num_heads * self.attn_value_channels, self.output_channels,
                                 lmax=self.lmax_list[0])

    def forward(
            self,
            x_src,
            x_dst,
            edge_distance,
            edge_src,
            edge_dst,
    ):

        # Compute edge scalar features (invariant to rotations)
        x_edge = edge_distance
        length = len(x_dst.embedding)
        output_src = x_src.clone()
        output_dst = x_dst.clone()

        output_src._expand_edge(edge_src)
        output_dst._expand_edge(edge_dst)

        x_message_data = torch.cat((output_src.embedding, output_dst.embedding), dim=2)

        x_message = SO3_Embedding(
            0,
            output_dst.lmax_list.copy(),
            output_dst.num_channels * 2,
            device=output_dst.device,
            dtype=output_dst.dtype
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(-1, (max(self.lmax_list) + 1), 2 * self.sphere_channels)
            x_edge_weight = torch.index_select(x_edge_weight, dim=1,
                                               index=self.expand_index)  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)
        after_Dij = x_message.embedding.clone()

        # First SO(2)-convolution
        # x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)
        x_0_extra = x_message.embedding[:, 0, :]
        x_message.embedding = x_message.embedding.mean(-1, keepdim=True)
        fij0, fijL = x_0_extra.clone(), x_message.clone()

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels

        if self.use_sep_s2_act:
            x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels,
                                          x_0_extra.shape[1] - x_alpha_num_channels)  # for activation
            x_0_alpha = x_0_extra.narrow(1, 0, x_alpha_num_channels)  # for attention weights
            x_message.embedding = self.s2_act(x_0_gating, x_message.embedding, self.SO3_grid)
        else:
            x_0_alpha = x_0_extra
            x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)
            ##x_message._grid_act(self.SO3_grid, self.value_act, self.mappingReduced)

        # Second SO(2)-convolution
        # x_message = self.so2_conv_2(x_message, x_edge)
        x_message.embedding = x_message.embedding.mean(-1, keepdim=True)

        # Attention weights
        x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
        x_0_alpha = self.alpha_norm(x_0_alpha)
        x_0_alpha = self.alpha_act(x_0_alpha)
        alpha = torch.einsum('bik, ik -> bi', x_0_alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)
        attn = attn * alpha
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        x_message.embedding = attn
        aij, vij = alpha.clone(), attn.clone()

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_message._reduce_edge(edge_dst, length)

        # Project
        out_embedding = self.proj(x_message)

        return out_embedding, after_Dij, fij0, fijL, aij, vij


if __name__ == "__main__":

    norm_type = 'rms_norm_sh'
    deterministic = False
    lmax_list = [3]
    mmax_list = [2]
    grid_resolution = None
    use_m_share_rad = False
    distance_function = "gaussian_soft"
    use_attn_renorm = True
    use_grid_mlp = False
    use_sep_s2_act = True
    weight_init = 'normal'
    healpix_grid = None  # SO(3) equal distance grid
    rot_mats = None  # rotation matrix for the grid
    rot_lmax = [4]
    rot_mmax = [3]
    rot_field_net = 'equiformer'  # 'equiformer' 'roformer'
    trans_field_net = 'equiformer'  # 'equiformer' 'roformer'
    rot_field_in_c = 64
    trans_field_in_c = 64
    include_proprio = True
    n_gripper_pos = 10
    film_dim = 128
    trans_lmax = [1]
    trans_mmax = [1]

    trans_SO3_rotation = nn.ModuleList()
    for i in range(1):
        trans_SO3_rotation.append(SO3_Rotation(trans_lmax[i]))
    # Initialize conversion between degree l and order m layouts
    trans_mappingReduced = CoefficientMappingModule(trans_lmax, trans_mmax)
    # Initialize the transformations between spherical and grid representations
    trans_SO3_grid = ModuleListInfo('({}, {})'.format(max(trans_lmax), max(trans_lmax)))
    for l in range(max(trans_lmax) + 1):
        trans_SO3_m_grid = nn.ModuleList()
        for m in range(max(trans_lmax) + 1):
            trans_SO3_m_grid.append(
                SO3_Grid(
                    l,
                    m,
                    resolution=grid_resolution,
                    normalization='component'
                )
            )
        trans_SO3_grid.append(trans_SO3_m_grid)

    query_num_distance_basis = 8
    edge_channels_list = [int(query_num_distance_basis)] + [1] * 2
    trans_field_layer = EvalSO2EquivariantGraphAttention(
        sphere_channels=1,
        hidden_channels=1,
        num_heads=1,
        attn_alpha_channels=1,
        attn_value_channels=1,
        output_channels=1,
        lmax_list=trans_lmax,
        mmax_list=trans_mmax,
        SO3_rotation=trans_SO3_rotation,
        mappingReduced=trans_mappingReduced,
        SO3_grid=trans_SO3_grid,
        edge_channels_list=edge_channels_list,
        use_m_share_rad=use_m_share_rad,
        use_attn_renorm=use_attn_renorm,
        use_sep_s2_act=use_sep_s2_act,
        alpha_drop=0
    )

    ###############################################################
    # Initialize data structures
    ###############################################################
    node_src0 = SO3_Embedding(
        3,
        trans_lmax,
        1,
        'cpu',
        torch.float
    )
    embedding = torch.arange(12).reshape(3, 4, 1).float()
    node_src0.embedding = embedding

    # Compute 3x3 rotation matrix per edge
    edge_vec = torch.tensor([[0.01, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0]]).float()
    edge_rot_mat = init_edge_rot_mat2(edge_vec)

    # Initialize the WignerD matrices and other values for spherical harmonic calculations
    for i in range(1):
        trans_SO3_rotation[i].set_wigner(edge_rot_mat)

    edge_length = torch.tensor([1, 1, 1]).float()
    field_distance_expansion = GaussianRadialBasisLayerFiniteCutoff(
        num_basis=query_num_distance_basis,
        cutoff=2 * 0.99)
    edge_attr0 = field_distance_expansion(edge_length)
    edge_src0, edge_dst0 = torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0])
    node_dst0 = copy.copy(node_src0)
    node_dst0.embedding = node_src0.embedding[0:1, ...]
    node_dst, after_Dij, fij0, fijL, aij, vij = trans_field_layer(node_src0,
                                                                  node_dst0,
                                                                  edge_attr0,
                                                                  edge_src0,
                                                                  edge_dst0)

    # print(node_src0.embedding[:, 0, 0])
    print(node_dst.embedding)
    # print(after_Dij, fij0, fijL.embedding[:, 0, 0], aij, vij[:, 0, 0])

    # ----------------- rotates edges -------------------
    # Compute 3x3 rotation matrix per edge
    edge_vec = torch.tensor([[0.01, 0, 0],
                             [0, 1, 0],
                             [1, 0, 0]]).float()
    edge_rot_mat = init_edge_rot_mat2(edge_vec)

    # Initialize the WignerD matrices and other values for spherical harmonic calculations
    for i in range(1):
        trans_SO3_rotation[i].set_wigner(edge_rot_mat)
    node_dst, after_Dij, fij0, fijL, aij, vij = trans_field_layer(node_src0,
                                                                  node_dst0,
                                                                  edge_attr0,
                                                                  edge_src0,
                                                                  edge_dst0)

    # print(node_src0.embedding[:, 0, 0])
    print(node_dst.embedding)
    # print(after_Dij, fij0, fijL.embedding[:, 0, 0], aij, vij[:, 0, 0])
