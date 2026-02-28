import time

import einops
import torch
import numpy as np
import torch.nn as nn
import utils.pytorch3d_transforms as pytorch3d_transforms
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
import dgl.geometry as dgl_geo

from diffuser_actor.equ_act_optimization.equ_field_unet import EquFieldUnet
from diffuser_actor.equ_act_optimization.se3_pcd_augmentation import aug_se3_at_origin
# from diffuser_actor.equ_act_optimization.equ_net import EquNet
from diffuser_actor.utils.position_encodings import RotaryPositionEncoding3D
from diffuser_actor.utils.layers import FFWRelativeCrossAttentionModule
from diffuser_actor.utils.utils import (
    normalise_quat,
    sample_ghost_points_uniform_cube,
    sample_ghost_points_uniform_sphere,
    compute_rotation_matrix_from_ortho6d
)
from diffuser_actor.utils.resnet import load_resnet50
from diffuser_actor.utils.clip import load_clip
import open3d as o3d
from diffuser_actor.equ_act_optimization.spherical_conv_utils import *


class EquAct(nn.Module):

    def __init__(self,
                 gripper_loc_bounds=None,
                 num_ghost_points=300,
                 num_ghost_points_val=1000,
                 weight_tying=True,
                 gp_emb_tying=True,
                 ins_pos_emb=False,
                 num_sampling_level=3,
                 fine_sampling_ball_diameter=0.25,
                 regress_position_offset=False,
                 workspace_bound=None,
                 use_instruction=False,
                 trans_temperature=None,
                 rot_temperature=None,
                 training=True,
                 table_height=0.751,
                 n_total_pts=None,
                 lmax=3,
                 film_type='iFiLM',
                 field_nn='equiformer',
                 field_nn_c=64,
                 rot_aug_range=[]):
        super().__init__()
        assert num_sampling_level in [1, 2, 3, 4]

        self.topk = 10 if training else 3
        self.num_ghost_points = num_ghost_points // num_sampling_level
        self.num_ghost_points -= self.num_ghost_points % self.topk
        self.num_ghost_points_val = num_ghost_points_val // num_sampling_level
        self.num_ghost_points_val -= self.num_ghost_points_val % self.topk
        assert self.num_ghost_points > 0
        assert self.num_ghost_points_val > 0
        self.num_sampling_level = num_sampling_level
        self.sampling_ball_diameter_pyramid = [
            None,
            fine_sampling_ball_diameter,
            fine_sampling_ball_diameter / 4.0,
            fine_sampling_ball_diameter / 16.0
        ]
        self.gripper_loc_bounds = np.array(gripper_loc_bounds)
        focused_pcd_bounds = gripper_loc_bounds.copy()
        focused_pcd_bounds[0, 2] = table_height  # table height
        workspace_bound = torch.tensor(workspace_bound) if workspace_bound is not None \
            else torch.tensor([[-0.3, -0.5, 0.6], [0.7, 0.5, 1.6]])
        self.trans_aug_range = torch.tensor([0., ] * 3)
        self.rot_aug_range = torch.tensor(rot_aug_range)
        self.trans_rad_aug_r = 0.0001  # translational RAD augmentation in meter
        self.register_buffer('workspace_bound', workspace_bound)
        self.register_buffer('gripper_bound', torch.tensor(gripper_loc_bounds))
        self.register_buffer('focused_pcd_bounds', torch.tensor(focused_pcd_bounds))
        self.regress_position_offset = regress_position_offset
        self.weight_tying = weight_tying
        self.gp_emb_tying = gp_emb_tying
        self.ins_pos_emb = ins_pos_emb
        # self.trans_temperature = trans_temperature
        self.trans_loss_func = torch.nn.CrossEntropyLoss()
        self.rot_temperature = rot_temperature
        self.n_total_pts = n_total_pts

        self.healpix_grid = so3_healpix_grid(rec_level=3 if training else 5)
        self.register_buffer("rot_mats", o3.angles_to_matrix(*self.healpix_grid), persistent=False)
        self.n_healpix = self.healpix_grid.shape[1]
        assert use_instruction is True
        self.equ_net = EquFieldUnet(healpix_grid=self.healpix_grid,
                                    rot_mats=self.rot_mats,
                                    alpha_drop=0.1,
                                    drop_path_rate=0.,
                                    proj_drop=0.1,
                                    deterministic=False,
                                    lmax_list=[lmax],
                                    film_type=film_type,
                                    field_net=field_nn,
                                    field_nn_c=field_nn_c)

    def forward(self, visible_rgb, visible_pcd, instruction, curr_gripper, gt_action=None):
        """
        Training: given expert demo, the equ net estimates the action, and calculate loss
        Or testing: given obs, estimate action (gt_action=None)
        Arguments:
            visible_rgb: (batch, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch, num_cameras, 3, height, width) in robot coordinates
            curr_gripper: (batch, 8)
            instruction: (batch, max_instruction_length, 512)
            gt_action: (batch, 8) in world coordinates
        """
        batch_size, num_cameras, _, height, width = visible_rgb.shape
        device = visible_rgb.device
        training = gt_action is not None

        # time.sleep(1)

        # FPS to n_points points
        xyz = einops.rearrange(visible_pcd, 'b n c h w -> b (n h w) c')
        rgb = einops.rearrange(visible_rgb, 'b n c h w -> b (n h w) c')
        # remove outbounded points
        inside_min = xyz >= self.workspace_bound[0]
        inside_max = xyz <= self.workspace_bound[1]
        inside = inside_min & inside_max
        inside_index = inside.all(dim=2)

        fps_xyz, fps_rgb = [], []
        n_not_focused = 512
        n_total_pts = self.n_total_pts
        for i in range(visible_pcd.shape[0]):
            inside_xyz = xyz[i][inside_index[i]].unsqueeze(0)
            inside_rgb = rgb[i][inside_index[i]].unsqueeze(0)
            # focus on points nearby action
            focused_min = inside_xyz[0] >= self.focused_pcd_bounds[0]
            focused_max = inside_xyz[0] <= self.focused_pcd_bounds[1]
            focused_idx = focused_min & focused_max
            focused_idx = focused_idx.all(dim=1)
            not_focused_xyz = inside_xyz[:, torch.logical_not(focused_idx)]
            not_focused_rgb = inside_rgb[:, torch.logical_not(focused_idx)]
            focused_xyz = inside_xyz[:, focused_idx]
            focused_rgb = inside_rgb[:, focused_idx]
            fps_table_index = dgl_geo.farthest_point_sampler(not_focused_xyz,
                                                             n_not_focused, start_idx=0).reshape(-1)
            fps_obj_index = dgl_geo.farthest_point_sampler(focused_xyz,
                                                           n_total_pts - n_not_focused, start_idx=0).reshape(-1)
            fps_out_xyz = torch.cat((not_focused_xyz[:, fps_table_index],
                                     focused_xyz[:, fps_obj_index]), dim=1)
            fps_out_rgb = torch.cat((not_focused_rgb[:, fps_table_index],
                                     focused_rgb[:, fps_obj_index]), dim=1)
            fps_xyz.append(fps_out_xyz)  # (1 n_points 3)
            fps_rgb.append(fps_out_rgb)  # (1 n_points 3)

        fps_xyz = torch.cat(fps_xyz, dim=0)  # (b n_points 3)
        fps_rgb = torch.cat(fps_rgb, dim=0)  # (b n_points 3)

        # random permutation to avoid deterministic behavior of radius()
        perm_idx = torch.randperm(fps_xyz.shape[1]).to(device)
        fps_xyz = fps_xyz[:, perm_idx, :]
        fps_rgb = fps_rgb[:, perm_idx, :]

        if self.training:
            # SE(3) data augmentation
            fps_xyz, curr_gripper[:, :-1], gt_action[:, :-1] = \
                aug_se3_at_origin(fps_xyz.clone(), curr_gripper[:, :-1].clone(), gt_action[:, :-1].clone(),
                                  self.gripper_bound, self.trans_aug_range, self.rot_aug_range)
            # Translational RAD augmentation (only applies to PCD, not action)
            if self.trans_rad_aug_r != 0:
                xyz_shift = sample_ghost_points_uniform_sphere(
                    center=np.array([0, 0, 0]),
                    radius=self.trans_rad_aug_r,
                    bounds=np.array([3 * [-self.trans_rad_aug_r, ], 3 * [self.trans_rad_aug_r, ]]),
                    num_points=batch_size * n_total_pts)
                fps_xyz += torch.tensor(xyz_shift, device=device).reshape(batch_size, n_total_pts, 3)
            # Ground truth label
            gt_position = gt_action[:, :3].unsqueeze(1).detach()
            gt_rot_xyzw = gt_action[:, 3:-1]  # quaternion in xyzw
            gt_rot_wxyz = gt_rot_xyzw[:, (3, 0, 1, 2)]
            gt_rot = pytorch3d_transforms.quaternion_to_matrix(gt_rot_wxyz)
            # gt_rot_idx = nearest_rotmat(gt_rot, self.rot_mats)
        else:
            gt_position = None
        # encode(xyz, rgb, curr_gripper) -> equivariant s2_feature
        s2_feature = self.equ_net.encode(fps_xyz, fps_rgb, curr_gripper, instruction)

        # # visualize FPS PCD
        # pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # pcd.points = o3d.utility.Vector3dVector(xyz[0].detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(rgb[0].detach().cpu().numpy())
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])
        # pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # pcd.points = o3d.utility.Vector3dVector(fps_xyz[0].detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(fps_rgb[0].detach().cpu().numpy())
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])

        # pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # pcd.points = o3d.utility.Vector3dVector(fps_xyz[0].detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(fps_rgb[0].detach().cpu().numpy())
        # plane_model, inliers = pcd.segment_plane(distance_threshold=0.003,
        #                                          ransac_n=3,
        #                                          num_iterations=3)
        # [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        #
        # inlier_cloud = pcd.select_by_index(inliers)
        # inlier_cloud.paint_uniform_color([1.0, 0, 0])
        # outlier_cloud = pcd.select_by_index(inliers, invert=True)
        #
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([outlier_cloud])

        ghost_pcd_pyramid = []
        position_pyramid = []
        Q_pose_pyramid = []
        loss = {}

        for level in range(self.num_sampling_level):
            # Sample ghost points
            if level == 0:
                anchors = None
            else:
                anchors = position_pyramid[-1]  # ((b k) 1 3)
            # (b ngpts 3)
            sampled_action_xyz = self._sample_ghost_points(batch_size, device, level=level, anchors=anchors)
            if level == self.num_sampling_level - 1 and training:  # ZXP adding gt coordinate to the sampled coordinate
                sampled_action_xyz[:, 0:1, :] = gt_position

            Q_trans_level = self.equ_net.trans_query(sampled_action_xyz, s2_feature)
            Q_trans_level = einops.rearrange(Q_trans_level, '(b npts) -> b npts', b=batch_size)

            trans_topk_idx = torch.topk(Q_trans_level, self.topk, dim=1).indices  # (b k)
            trans_topk_idx = trans_topk_idx + torch.arange(batch_size).unsqueeze(1).to(
                trans_topk_idx.device) * Q_trans_level.shape[-1]
            trans_topk_idx = trans_topk_idx.reshape(-1)  # ((b k))
            best_trans_level = sampled_action_xyz.reshape(-1, 3)[trans_topk_idx].unsqueeze(1)  # ((b k) 1 3)
            if gt_position is not None:
                best_trans_level[self.topk - 1::self.topk, ...] = gt_position

            if training:
                # gt_trans_idx = nearest_pose_idx(sampled_action_xyz, gt_position)  # (b)
                # # Cross Entropy Loss
                # loss['trans_loss_{}'.format(level)] = F.cross_entropy(Q_trans_level, gt_trans_idx)
                # Multi-Class Cross Entropy Loss
                loss['trans_loss_{}'.format(level)] = self._compute_position_loss(Q_trans_level,
                                                                                  sampled_action_xyz,
                                                                                  gt_position)

            ghost_pcd_pyramid.append(sampled_action_xyz[0])
            position_pyramid.append(best_trans_level.clone())
            Q_pose_pyramid.append(Q_trans_level[0].clone())

        rot_acc_degree = None
        # Q_rot in (b nhealpix)
        if training:
            Q_roq = self.equ_net.rot_query(gt_position, s2_feature)
        else:
            Q_roq = self.equ_net.rot_query(best_trans_level[::self.topk, ...], s2_feature)
        Q_rot, Q_open, query_graph, _ = Q_roq
        gripper = torch.sigmoid(Q_open)  # (b 1)

        # gripper = Q_open_npts_level[top_idx_xyz].unsqueeze(1)  # (b 1)
        # gripper = torch.sigmoid(gripper)
        # s2_conv_feature = s2_conv_npts_feature[top_idx_xyz]  # (b c i)
        # Q_rot = self.equ_net.decode_rot(s2_conv_feature)  # (b nhealpix)

        top_idx_rot = torch.max(Q_rot, dim=1).indices  # (b)
        rotation_level = self.rot_mats[top_idx_rot.reshape(-1)]  # (b 3 3)

        if training:
            loss['open_loss'] = F.binary_cross_entropy_with_logits(Q_open, gt_action[:, -1:])
            # loss['rot_loss'] = F.cross_entropy(Q_rot, gt_rot_idx)
            loss['rot_loss'] = self._compute_rotation_loss(Q_rot, gt_rot)
            with torch.no_grad():
                rot_acc_degree = rotation_error(gt_rot, rotation_level).cpu().numpy() / np.pi * 180

        position = position_pyramid[-1][::self.topk, 0, :]
        xyzw_rotation = pytorch3d_transforms.matrix_to_quaternion(rotation_level)[:, (1, 2, 3, 0)]
        # position_pyramid = [pos[::self.topk] for pos in position_pyramid]  # only record the best pose

        # # visualize PCD, Q_trans
        # pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # q_red = torch.cat(Q_pose_pyramid, dim=0)
        # q_red -= q_red.min()
        # q_red /= q_red.max()
        # q_red = q_red.reshape(-1, 1).repeat(1, 3)
        # q_red[:, (2)] = 1 - q_red[:, (2)]
        # # q_red[-self.num_ghost_points] = 0
        # # q_red[-self.num_ghost_points, 0] = 1
        # action_xyz = torch.cat(ghost_pcd_pyramid, dim=0)
        # visualize_xyz = torch.cat((fps_xyz[0], action_xyz), dim=0)
        # visualize_rgb = torch.cat((fps_rgb[0], q_red), dim=0)
        # pcd.points = o3d.utility.Vector3dVector(visualize_xyz.detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(visualize_rgb.detach().cpu().numpy())
        # # Visualize the expert pose
        # if gt_position is not None:
        #     expert_pose = gt_position[0].clone().repeat(6, 1)
        #     r = 0.1
        #     expert_pose[torch.arange(3), torch.arange(3)] -= r
        #     expert_pose[torch.arange(3) + 3, torch.arange(3)] += r
        #     expert_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        #     expert_color = torch.zeros((3, 3))
        #     expert_color[:, 0] = 1
        #     expert_coord = o3d.geometry.LineSet()
        #     expert_coord.points = o3d.utility.Vector3dVector(expert_pose.detach().cpu().numpy())
        #     expert_coord.lines = o3d.utility.Vector2iVector(expert_edge.detach().cpu().numpy())
        #     expert_coord.colors = o3d.utility.Vector3dVector(expert_color.detach().cpu().numpy())
        #
        # # Visualize the agent pose
        # agent_pose = position_pyramid[-1][0, 0, :].clone().repeat(6, 1) + 0.001  # offset to distinguish from expert
        # r = 0.1
        # agent_pose[torch.arange(3), torch.arange(3)] -= r
        # agent_pose[torch.arange(3) + 3, torch.arange(3)] += r
        # agent_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        # agent_color = torch.zeros((3, 3))
        # agent_color[:, 1] = 1
        # agent_color[:, 0] = 0.5
        # agent_coord = o3d.geometry.LineSet()
        # agent_coord.points = o3d.utility.Vector3dVector(agent_pose.detach().cpu().numpy())
        # agent_coord.lines = o3d.utility.Vector2iVector(agent_edge.detach().cpu().numpy())
        # agent_coord.colors = o3d.utility.Vector3dVector(agent_color.detach().cpu().numpy())
        # # Visualize the point cloud
        # if gt_position is not None:
        #     o3d.visualization.draw_geometries([pcd, expert_coord, agent_coord])
        # else:
        #     o3d.visualization.draw_geometries([pcd, agent_coord])
        #
        # # Visualizing s2_embedding
        # pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # s2_embed, s2_xyz, _ = s2_feature
        # s2_color = s2_embed.embedding[:s2_xyz.shape[0] // batch_size, 0, :3]
        # s2_color -= s2_color.min(dim=0, keepdim=True)[0]
        # s2_color /= s2_color.max(dim=0, keepdim=True)[0]
        # pcd.points = o3d.utility.Vector3dVector(s2_xyz[:s2_xyz.shape[0] // batch_size, ...].detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(s2_color.detach().cpu().numpy())
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])
        #
        # # Visualizing PCD and Field Graph
        # pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # pcd.points = o3d.utility.Vector3dVector(fps_xyz[0].detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(fps_rgb[0].detach().cpu().numpy())
        # # Construct Graph for the Field NN
        # query_graph_nodes, query_graph_edges = query_graph
        # colors = torch.zeros_like(query_graph_nodes[:self.equ_net.rot_query_max_neighbors])
        # colors[:-100, (1)] = 1
        # colors[-100:, (0, 2)] = 1
        # line_set = o3d.geometry.LineSet()
        # line_set.points = o3d.utility.Vector3dVector(query_graph_nodes.detach().cpu().numpy())
        # line_set.lines = o3d.utility.Vector2iVector(query_graph_edges.detach().cpu().numpy())
        # line_set.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy())
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd, line_set])
        #
        # # Visualize Q_rot
        # gt_rot_quat = gt_rot_wxyz[0].detach().cpu() if training else None
        # plot_so3_distribution(torch.nn.functional.softmax(Q_rot[0]).detach().cpu(),
        #                       self.rot_mats.detach().cpu(), gt_rot_quat)
        #
        # idx = 1
        # # visualize gt pose
        # pcd = o3d.geometry.PointCloud()
        # visualize_xyz = fps_xyz[idx]
        # visualize_rgb = fps_rgb[idx]
        # pcd.points = o3d.utility.Vector3dVector(visualize_xyz.detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(visualize_rgb.detach().cpu().numpy())
        # # Visualize the expert pose
        # if gt_position is not None:
        #     expert_pose = gt_position[idx].clone().repeat(6, 1)
        #     r = 0.1
        #     expert_pose[:3] += r * gt_rot[idx].permute(1, 0)
        #     expert_pose[3:] -= r * gt_rot[idx].permute(1, 0)
        #     expert_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        #     expert_color = torch.zeros((3, 3))
        #     expert_color[torch.arange(3), torch.arange(3)] = 1
        #     expert_coord = o3d.geometry.LineSet()
        #     expert_coord.points = o3d.utility.Vector3dVector(expert_pose.detach().cpu().numpy())
        #     expert_coord.lines = o3d.utility.Vector2iVector(expert_edge.detach().cpu().numpy())
        #     expert_coord.colors = o3d.utility.Vector3dVector(expert_color.detach().cpu().numpy())
        #
        # # Visualize the agent pose
        # agent_pose = position_pyramid[-1][1+self.topk*idx, 0, :].clone().repeat(6, 1) + 0.001  # offset to distinguish from expert
        # r = 0.1
        # agent_pose[torch.arange(3), torch.arange(3)] -= r
        # agent_pose[torch.arange(3) + 3, torch.arange(3)] += r
        # agent_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        # agent_color = torch.zeros((3, 3))
        # agent_color[:, 1] = 1
        # agent_color[:, 0] = 0.5
        # agent_coord = o3d.geometry.LineSet()
        # agent_coord.points = o3d.utility.Vector3dVector(agent_pose.detach().cpu().numpy())
        # agent_coord.lines = o3d.utility.Vector2iVector(agent_edge.detach().cpu().numpy())
        # agent_coord.colors = o3d.utility.Vector3dVector(agent_color.detach().cpu().numpy())
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd, expert_coord, agent_coord])

        # visualize PCD, Q_trans
        # pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # q_red = torch.cat(Q_pose_pyramid, dim=0)
        # q_red -= q_red.min()
        # q_red /= q_red.max()
        # q_red = q_red.reshape(-1, 1).repeat(1, 3)
        # q_red[:, 0] += 0.5
        # q_red[:, (1,2)] = 0.1
        # # q_red[:, (2)] = 1 - q_red[:, (2)]
        # # q_red[-self.num_ghost_points] = 0
        # # q_red[-self.num_ghost_points, 0] = 1
        # action_xyz = torch.cat(ghost_pcd_pyramid, dim=0)
        # visualize_xyz = torch.cat((fps_xyz[0], action_xyz[300:]), dim=0)
        # visualize_rgb = torch.cat((fps_rgb[0]/3+0.5, q_red[300:]), dim=0)
        # # visualize_xyz = xyz[0][inside_index[0]]
        # # visualize_rgb = rgb[0][inside_index[0]]
        # pcd.points = o3d.utility.Vector3dVector(visualize_xyz.detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(visualize_rgb.detach().cpu().numpy())
        # # Visualize the expert pose
        # if gt_position is not None:
        #     expert_pose = gt_position[0].clone().repeat(6, 1)
        #     r = 0.1
        #     expert_pose[torch.arange(3), torch.arange(3)] -= r
        #     expert_pose[torch.arange(3) + 3, torch.arange(3)] += r
        #     expert_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        #     expert_color = torch.zeros((3, 3))
        #     expert_color[:, 0] = 1
        #     expert_coord = o3d.geometry.LineSet()
        #     expert_coord.points = o3d.utility.Vector3dVector(expert_pose.detach().cpu().numpy())
        #     expert_coord.lines = o3d.utility.Vector2iVector(expert_edge.detach().cpu().numpy())
        #     expert_coord.colors = o3d.utility.Vector3dVector(expert_color.detach().cpu().numpy())
        #
        # # Visualize the agent pose
        # agent_pose = position_pyramid[-1][0, 0, :].clone().repeat(6, 1) + 0.001  # offset to distinguish from expert
        # r = 0.1
        # agent_pose[torch.arange(3), torch.arange(3)] -= r
        # agent_pose[torch.arange(3) + 3, torch.arange(3)] += r
        # agent_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        # agent_color = torch.zeros((3, 3))
        # agent_color[:, 1] = 1
        # agent_color[:, 0] = 0.5
        # agent_coord = o3d.geometry.LineSet()
        # agent_coord.points = o3d.utility.Vector3dVector(agent_pose.detach().cpu().numpy())
        # agent_coord.lines = o3d.utility.Vector2iVector(agent_edge.detach().cpu().numpy())
        # agent_coord.colors = o3d.utility.Vector3dVector(agent_color.detach().cpu().numpy())
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])
        # # if gt_position is not None:
        # #     o3d.visualization.draw_geometries([pcd, expert_coord, agent_coord])
        # # #
        # # # # Visualizing s2_embedding
        # # # pcd = o3d.geometry.PointCloud()
        # # # # Assign points and colors to the PointCloud
        # # # s2_embed, s2_xyz, _ = s2_feature
        # # # s2_color = s2_embed.embedding[:s2_xyz.shape[0] // batch_size, 0, :3]
        # # # s2_color -= s2_color.min(dim=0, keepdim=True)[0]
        # # # s2_color /= s2_color.max(dim=0, keepdim=True)[0]
        # # # s2_color += 0.2
        # # # pcd.points = o3d.utility.Vector3dVector(s2_xyz[:s2_xyz.shape[0] // batch_size, ...].detach().cpu().numpy())
        # # # pcd.colors = o3d.utility.Vector3dVector(s2_color.detach().cpu().numpy())
        # # # # Visualize the point cloud
        # # # o3d.visualization.draw_geometries([pcd])
        # #
        # # # Visualizing PCD and Field Graph
        # # pcd = o3d.geometry.PointCloud()
        # # # Assign points and colors to the PointCloud
        # # pcd.points = o3d.utility.Vector3dVector(fps_xyz[0].detach().cpu().numpy())
        # # pcd.colors = o3d.utility.Vector3dVector(fps_rgb[0].detach().cpu().numpy())
        # # # Construct Graph for the Field NN
        # # query_graph_nodes, query_graph_edges = query_graph
        # # query_graph_edges[:, 0] = 99
        # # query_graph_edges[-100:, 1] = torch.arange(100).to(query_graph_edges.device)
        # # colors = torch.zeros_like(query_graph_nodes[:self.equ_net.rot_query_max_neighbors])
        # # colors[:, (1)] = 1
        # # query_graph_nodes[99, :] = torch.tensor([0.6, -0.2, 1.2]).to(query_graph_edges.device)
        # # # colors[:-100, (1)] = 1
        # # # colors[-100:, (0, 2)] = 1
        # # line_set = o3d.geometry.LineSet()
        # # line_set.points = o3d.utility.Vector3dVector(query_graph_nodes[:100].detach().cpu().numpy())
        # # line_set.lines = o3d.utility.Vector2iVector(query_graph_edges[-100:].detach().cpu().numpy())
        # # line_set.colors = o3d.utility.Vector3dVector(colors[-100:].detach().cpu().numpy())
        # # # Visualize the point cloud
        # # o3d.visualization.draw_geometries([pcd, line_set])
        # #
        # # # Visualize Q_rot
        # # gt_rot_quat = gt_rot_wxyz[0].detach().cpu() if training else None
        # # # plot_so3_distribution(torch.nn.functional.softmax(Q_rot[0]).detach().cpu(),
        # # #                       self.rot_mats.detach().cpu(), gt_rot_quat)
        # # plot_so3_distribution(torch.nn.functional.softmax(Q_rot[0]).detach().cpu(),
        # #                       self.rot_mats.detach().cpu())
        # #
        # # idx = 1
        # # # visualize gt pose
        # # pcd = o3d.geometry.PointCloud()
        # # visualize_xyz = fps_xyz[idx]
        # # visualize_rgb = fps_rgb[idx]
        # # pcd.points = o3d.utility.Vector3dVector(visualize_xyz.detach().cpu().numpy())
        # # pcd.colors = o3d.utility.Vector3dVector(visualize_rgb.detach().cpu().numpy())
        # # # Visualize the expert pose
        # # if gt_position is not None:
        # #     expert_pose = gt_position[idx].clone().repeat(6, 1)
        # #     r = 0.1
        # #     expert_pose[:3] += r * gt_rot[idx].permute(1, 0)
        # #     expert_pose[3:] -= r * gt_rot[idx].permute(1, 0)
        # #     expert_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        # #     expert_color = torch.zeros((3, 3))
        # #     expert_color[torch.arange(3), torch.arange(3)] = 1
        # #     expert_coord = o3d.geometry.LineSet()
        # #     expert_coord.points = o3d.utility.Vector3dVector(expert_pose.detach().cpu().numpy())
        # #     expert_coord.lines = o3d.utility.Vector2iVector(expert_edge.detach().cpu().numpy())
        # #     expert_coord.colors = o3d.utility.Vector3dVector(expert_color.detach().cpu().numpy())
        # #
        # # # Visualize the agent pose
        # # agent_pose = position_pyramid[-1][1+self.topk*idx, 0, :].clone().repeat(6, 1) + 0.001  # offset to distinguish from expert
        # # r = 0.1
        # # agent_pose[torch.arange(3), torch.arange(3)] -= r
        # # agent_pose[torch.arange(3) + 3, torch.arange(3)] += r
        # # agent_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        # # agent_color = torch.zeros((3, 3))
        # # agent_color[:, 1] = 1
        # # agent_color[:, 0] = 0.5
        # # agent_coord = o3d.geometry.LineSet()
        # # agent_coord.points = o3d.utility.Vector3dVector(agent_pose.detach().cpu().numpy())
        # # agent_coord.lines = o3d.utility.Vector2iVector(agent_edge.detach().cpu().numpy())
        # # agent_coord.colors = o3d.utility.Vector3dVector(agent_color.detach().cpu().numpy())
        # # # Visualize the point cloud
        # # o3d.visualization.draw_geometries([pcd, expert_coord, agent_coord])

        return {
            "loss": loss,
            "rot_acc_degree": rot_acc_degree,
            # Action
            "position": position,
            "rotation": xyzw_rotation,
            "gripper": gripper,
            # Auxiliary outputs used to compute the loss or for visualization
            "position_pyramid": position_pyramid,
            "ghost_pcd_pyramid": ghost_pcd_pyramid,
        }

    def _compute_rotation_loss(self, Q_rot, gt_rot):
        # Boltzmann distribution with respect to geodesics distance (in degree)
        # as a proxy label for a soft cross-entropy loss
        l2_i = rotation_error(gt_rot, self.rot_mats) / torch.pi * 180
        # l2_i = l2_i.reshape(gt_rot.shape[0], -1)
        label_i = torch.softmax(-l2_i / self.rot_temperature, dim=-1).detach()
        loss = F.cross_entropy(Q_rot, label_i, label_smoothing=0).mean()
        return loss

    def _compute_position_loss(self, Q_pos, sampled_action_xyz, gt_position):
        l2_i = ((sampled_action_xyz - gt_position) ** 2).sum(2).sqrt()  # (b npts)

        # # Boltzmann distribution with respect to geodesics distance (in meter)
        # # as a proxy label for a soft cross-entropy loss
        # label_i = torch.softmax(-l2_i / self.trans_temperature, dim=-1).detach()
        # loss = F.cross_entropy(Q_pos, label_i, label_smoothing=0).mean()

        # Cross-entropy loss for each level of ghost points
        label_i = torch.argmin(l2_i, dim=-1).detach().long().reshape(-1)
        loss = self.trans_loss_func(Q_pos, label_i)

        return loss

    def prepare_action(self, pred) -> torch.Tensor:
        rotation = pred["rotation"]
        # print(pred["position"], rotation, pred["gripper"])
        return torch.cat(
            [pred["position"], rotation, pred["gripper"]],
            dim=1,
        )

    def _sample_ghost_points(self, batch_size, device, level, anchors=None):
        """Sample ghost points.

        If level==0, sample num_ghost_points_X points uniformly within the workspace bounds.

        If level>0, sample num_ghost_points_X points uniformly within a local sphere
        of the workspace bounds centered around the anchors. If there are more than 1 anchor, sample
        num_ghost_points_X / num_anchors for each anchor.

        return: uniform_pcd in shape (b npts 3)
        """
        if self.training:
            num_ghost_points = self.num_ghost_points
        else:
            num_ghost_points = self.num_ghost_points_val

        if level == 0:
            bounds = np.stack([self.gripper_loc_bounds for _ in range(batch_size)])
            uniform_pcd = np.stack([
                sample_ghost_points_uniform_cube(
                    bounds=bounds[i],
                    num_points=num_ghost_points
                )
                for i in range(batch_size)
            ])

        elif level >= 1:
            num_anchors = len(anchors) // batch_size
            num_ghost_points //= num_anchors
            anchor_ = anchors[:, 0].cpu().numpy()
            bounds_min = np.clip(
                anchor_ - self.sampling_ball_diameter_pyramid[level] / 2,
                a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
            )
            bounds_max = np.clip(
                anchor_ + self.sampling_ball_diameter_pyramid[level] / 2,
                a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
            )
            bounds = np.stack([bounds_min, bounds_max], axis=1)
            uniform_pcd = np.stack([
                sample_ghost_points_uniform_sphere(
                    center=anchor_[i],
                    radius=self.sampling_ball_diameter_pyramid[level] / 2,
                    bounds=bounds[i],
                    num_points=num_ghost_points
                )
                for i in range(len(anchors))
            ])
            uniform_pcd = uniform_pcd.reshape(batch_size, -1, 3)

        uniform_pcd = torch.from_numpy(uniform_pcd).float().to(device)

        return uniform_pcd
