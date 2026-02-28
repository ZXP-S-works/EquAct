import einops
import torch
import utils.pytorch3d_transforms as pytorch3d_transforms

def transform_pcd(pcd, matrix_4x4):
    """
    pcd: (bs, npts, 3)
    matrix_4x4: (bs, 4, 4)
    """
    bs = pcd.shape[0]
    transformed_pcd = []

    # homogeneous point cloud
    p_flat = einops.rearrange(pcd, 'b n d -> b d n')
    p_flat_4x1 = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)
    p_flat_4x1[:, :3, :] = p_flat

    # apply transformation
    perturbed_p_flat_4x1 = torch.bmm(matrix_4x4, p_flat_4x1)  # (bs, 4, npts)
    perturbed_p = einops.rearrange(perturbed_p_flat_4x1, 'b d n -> b n d')[:, :, :3]  # (bs, npts, 3)
    return perturbed_p


def gripper_xyzxyzw_pose_to_matrix(action_gripper_pose: torch.Tensor):
    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0) \
        .repeat(action_gripper_pose.shape[0], 1, 1).to(device=action_gripper_pose.device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = action_gripper_pose[:, 3:][:, (3, 0, 1, 2)]
    action_gripper_rot = pytorch3d_transforms.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans
    return action_gripper_4x4


def gripper_matrix_to_xyzxyzw_pose(action_gripper_4x4: torch.Tensor):
    action_gripper_trans = action_gripper_4x4[:, 0:3, 3]
    action_gripper_wxyz = pytorch3d_transforms.matrix_to_quaternion(action_gripper_4x4[:, :3, :3])
    action_gripper_xyzw = action_gripper_wxyz[:, (1, 2, 3, 0)]
    action_gripper_pose = torch.cat([action_gripper_trans, action_gripper_xyzw], dim=1)
    return action_gripper_pose


def rand_dist(size, min=-1.0, max=1.0):
    return (max - min) * torch.rand(size) + min


def get_augment_matrix(trans_aug_range, rot_aug_range, bs, device, center):
    canonicalize_trans = torch.eye(4).unsqueeze(0)
    canonicalize_trans[0, :3, -1] = -center
    uncanonicalize_trans = torch.eye(4).unsqueeze(0)
    uncanonicalize_trans[0, :3, -1] = center
    augmentation_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1)

    # sample translation perturbation with specified range
    trans_shift = trans_aug_range * rand_dist((bs, 3))
    augmentation_4x4[:, 0:3, 3] = trans_shift

    # sample rotation perturbation at specified resolution and range
    rot_shift = torch.deg2rad(rot_aug_range) * rand_dist((bs, 3))
    rot_shift_3x3 = pytorch3d_transforms.euler_angles_to_matrix(rot_shift, "XYZ")
    augmentation_4x4[:, :3, :3] = rot_shift_3x3

    augmentation_4x4 = uncanonicalize_trans @ augmentation_4x4 @ canonicalize_trans
    return augmentation_4x4.to(device)


def aug_se3_at_origin(fps_xyz, curr_gripper_pose, gt_action_pose, pose_bound, trans_aug_range, rot_aug_range):
    """
    SE(3) data augmentation.
    g ~ SE(3), though usually g is a small trans-rotal perturbation constrained by trans_aug_range, rot_aug_range
    fps_xyz, curr_gripper_pose, gt_action_pose = g·fps_xyz, g·curr_gripper_pose, g·gt_action_pose
    input:
    trans_aug_range: (1 3), +-xyz Cartesian augmentation in meter
    rot_aug_range: (1 3), +- rpy Euler augmentation in degree
    fps_xyz: (batch, num_cameras, 3, height, width) in robot coordinates
    curr_gripper_pose: (batch, 8), 8 = xyz coordinate + xyzw quaternion
    gt_action_pose: (batch, 8), 8 = xyz coordinate + xyzw quaternion
    """

    bs, npts, d = fps_xyz.shape
    device = fps_xyz.device
    curr_gripper_pose_4x4 = gripper_xyzxyzw_pose_to_matrix(curr_gripper_pose)
    gt_action_pose_4x4 = gripper_xyzxyzw_pose_to_matrix(gt_action_pose)

    for tries in range(50):
        aug_4x4 = get_augment_matrix(trans_aug_range, rot_aug_range, bs, device, pose_bound.mean(0))

        # apply perturbation to poses
        perturbed_curr_gripper_pose_4x4 = torch.bmm(aug_4x4, curr_gripper_pose_4x4)
        perturbed_curr_gripper_pose = gripper_matrix_to_xyzxyzw_pose(perturbed_curr_gripper_pose_4x4)

        # perturb ground truth action, if it's exist
        perturbed_gt_action_pose_4x4 = torch.bmm(aug_4x4, gt_action_pose_4x4)
        perturbed_gt_action_pose = gripper_matrix_to_xyzxyzw_pose(perturbed_gt_action_pose_4x4)
        if torch.all((pose_bound[0:1] < perturbed_gt_action_pose[:, :3])
                     & (perturbed_gt_action_pose[:, :3] < pose_bound[1:2])):
            # apply perturbation to point-clouds
            perturbed_fps_xyz = transform_pcd(fps_xyz, aug_4x4)
            break
    if tries == 49:
        print('cannot find valid augmentation matrix, using identity')
        perturbed_fps_xyz = fps_xyz
        perturbed_curr_gripper_pose = curr_gripper_pose
        perturbed_gt_action_pose = gt_action_pose

    return perturbed_fps_xyz, perturbed_curr_gripper_pose, perturbed_gt_action_pose
