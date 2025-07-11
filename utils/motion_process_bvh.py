import sys
sys.path.append('../')

from os.path import join as pjoin
from scipy.ndimage import gaussian_filter1d

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *

import torch
from tqdm import tqdm

from utils import bvh_io
from einops import rearrange, repeat
from collections import defaultdict

contact_joint_names = ["L_foot0001_bind_JNT", "L_foot0002_bind_JNT", "R_foot0001_bind_JNT", "R_foot0002_bind_JNT"]

face_joint_names = ["R_legUpper0001_bind_JNT", "L_legUpper0001_bind_JNT", "R_clavicle_bind_JNT", "L_clavicle_bind_JNT"]

joints_dict = {}
kinematic_tree = []

def process_bvh_motion(filepath, now_fps, target_fps, feet_thre, shift_one_frame=False, animation=None):

    if animation is None:
        animation = bvh_io.load(filepath)
    if shift_one_frame:
        animation = animation[1:]
    joints_dict = {animation.names[i]:i for i in range(len(animation.names))}

    contact_ids = [joints_dict[name] for name in contact_joint_names]
    face_joint_ids = [joints_dict[name] for name in face_joint_names]

    r_hip, l_hip, r_sdr, l_sdr = face_joint_ids
    fid_l, fid_r = contact_ids[:2], contact_ids[2:]

    """Downsampling to fps"""
    assert now_fps % target_fps == 0, "Invalid target fps!!"
    ds_rate = now_fps // target_fps
    rotations = animation.rotations[::ds_rate]
    positions = animation.positions[::ds_rate]

    """Do FK"""
    skeleton = Skeleton(animation.offsets, animation.parents, device="cpu")
    global_quat, global_pos = skeleton.fk_local_quat_np(rotations, positions[:, 0])

    """"Remove unuseful joints (skipped)"""


    """Put on Floor"""
    # Y+ is up axis
    # floor_height = global_pos.min(axis=0).min(axis=0)[1]
    seq = np.sort(global_pos[:, contact_ids, 1].flatten())
    nmin_seq = int(len(seq) *0.1)
    floor_height = seq[:nmin_seq].mean()
    global_pos[:, :, 1] -= floor_height

    """Extract forward direction and smooth"""
    
    across = (
        (global_pos[:, l_sdr] - global_pos[:, r_sdr]) + 
        (global_pos[:, l_hip] - global_pos[:, r_hip])
    )

    across = across / np.sqrt(across**2).sum(axis=-1)[..., np.newaxis]
    direction_filterwidth = 5
    forward = gaussian_filter1d(
        np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode="nearest"
        )
    # forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_rotations = qbetween_np(forward, target)[:, np.newaxis]
    root_rotations = np.repeat(root_rotations, global_pos.shape[1], axis=1)

    """All initially face z+"""
    root_rotation_init = root_rotations[0:1].repeat(len(root_rotations), axis=0)
    root_rotations = qmul_np(qinv_np(root_rotation_init), root_rotations)
    global_quat = qmul_np(root_rotation_init, global_quat)
    global_pos = qrot_np(root_rotation_init, global_pos)

    """Re-gain global positions"""
    global_pos = skeleton.fk_global_quat_np(global_quat, global_pos[:, 0])

    """Get foot contact"""
    def detect_contact(positions, thres):
        # TODO check the height
        velfactor, heightfactor = np.array([thres, thres]), np.array([9.0, 4.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1, fid_l, 1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(float)
        #     feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1, fid_r, 1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(float)
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r
    
    def get_con6d_params(r_rot, r_pos, quat_params):
        """"Remove root rotations from joint rotations"""
        quat_params = qmul_np(r_rot, quat_params)

        """Quaternion to continuous 6D representation"""
        cont6d_params = quaternion_to_cont6d_np(quat_params)

        """Root Linear Velocity"""
        velocity = (r_pos[1:] - r_pos[:-1]).copy()
        velocity = qrot_np(r_rot[:-1, 0], velocity)

        """Root angular velocity"""
        #TODO Double check if it's continuous, To-Do
        r_velocity = qmul_np(r_rot[1:, 0], qinv_np(r_rot[:-1, 0]))
        r_velocity = r_velocity / np.linalg.norm(r_velocity,  axis=-1)[:, None]
        r_velocity = np.arctan2(r_velocity[:, 2:3], r_velocity[:, 0:1])*2
        return cont6d_params[:-1], velocity, r_velocity
    
    def get_local_positions(r_rot, positions):
        positions = positions.copy()
        """Local pose"""
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        """All pose face Z+"""
        positions = qrot_np(r_rot, positions)

        """Get Joint Velocity"""
        local_vel = positions[1:] - positions[:-1]
        return positions[:-1], local_vel
    
    feet_l, feet_r = detect_contact(global_pos, thres = feet_thre)
    cont6d_param, l_velocity, r_velocity = get_con6d_params(root_rotations, global_pos[:, 0], global_quat)

    local_positions, local_velocity = get_local_positions(root_rotations, global_pos)

    """Root height"""
    root_y = local_positions[:, 0, 1:2]

    """Linear root velocity"""
    l_velocity = l_velocity[:, [0, 2]]
    
    """Root data"""
    root_data = np.concatenate([r_velocity, l_velocity, root_y], axis=-1)

    """Get joint rotation representation"""
    rot_data = cont6d_param.reshape(len(cont6d_param), -1)

    """Get root-rotation-invariant position representation"""
    ric_data = local_positions.reshape(len(local_positions), -1)
    """Get Joint Velocity Representation"""
    vel_data = local_velocity.reshape(len(local_velocity), -1)

    data = np.concatenate([root_data, rot_data, ric_data, vel_data, feet_l, feet_r], axis=-1)
    return data


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# rot_data (B, seq_len, joint_num * 6)
# ric_data (B, seq_len, joint_num * 3)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis Rotation from Rotation Velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang / 2, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    # (vx, vy, vz, r) - > (cos(r/2), vx * sin(r/2), vy * sin(r/2), vz * sin(r/2))
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    """Get Root Positions"""
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    #     print(torch.sum(r_pos**2, axis=-1)[:100])
    """Add Y-axis Rotation to Root Positions"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    #     print(torch.sum(r_pos**2, axis=-1)[:100])

    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# rot_data (B, seq_len, joint_num * 6)
# ric_data (B, seq_len, joint_num * 3)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_bvh_from_rot(data, joints_num, skeleton, keep_shape=True):
    b, l, _ = data.shape
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    start_indx = 1 + 2 + 1
    end_indx = start_indx + joints_num * 6
    # print(data.shape)
    cont6d_params = data[..., start_indx:end_indx].reshape(b, l, joints_num, 6)
    quat_params = cont6d_to_quaternion(cont6d_params)
    # print(quat_params.shape)
    
    # print(r_rot_quat.shape, quat_params.shape)
    quat_params = rearrange(quat_params, 'b l j d -> (b l) j d')
    r_rot_quat = repeat(r_rot_quat, 'b l d -> (b l) j d', j=quat_params.shape[1])
    r_pos = rearrange(r_pos, 'b l d -> (b l) d')

    global_quats = qmul(qinv(r_rot_quat), quat_params)
    # global_quats = qmul(qinv(r_rot_quat)[:, np.newaxis].repeat(1,joints_num, 1), quat_params)
    local_quats = skeleton.global_to_local_quat(global_quats)
    
    if keep_shape:
        global_quats = rearrange(global_quats, '(b l) j d -> b l j d', b = b)
        local_quats = rearrange(local_quats, '(b l) j d -> b l j d', b = b)
        r_pos = rearrange(r_pos, '(b l) d -> b l d', b = b)
    return global_quats, local_quats, r_pos

def recover_pos_from_rot(data, joints_num, skeleton):
    # print(data.shape)
    b, l, _ = data.shape
    global_quats, _, r_pos = recover_bvh_from_rot(data, joints_num, skeleton, keep_shape=False)
    global_pos = skeleton.fk_global_quat(global_quats, r_pos)
    global_pos = rearrange(global_pos, '(b l) j d -> b l j d', b = b)
    return global_pos

def recover_pos_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    start_indx = 1 + 2 + 1 + joints_num * 6
    end_indx = start_indx + joints_num * 3
    positions = data[..., start_indx:end_indx]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    # print(positions.shape)

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # '''Concate root and joints'''
    # positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def extract_kinematic_chains(animation):
    kinematic_chains = []
    chain = []
    for i, parent_i in enumerate(animation.parents):
        if parent_i != i-1:
            kinematic_chains.append(chain)
            chain = [parent_i, i]
        else:
            chain.append(i)
            
    kinematic_chains.append(chain)
    return kinematic_chains


if __name__ == "__main__":
    src_root = "/mnt/local-disk/snap_motion_dataset/raw_data/"
    tgt_root = "/mnt/local-disk/snap_motion_dataset/raw_data/renamed_feats"
    # folders = os.listdir(src_root)
    folders = ["renamed_bvhs"]
    length_dict = defaultdict(int)
    # print(folders)
    for folder in folders:
        print(folder)
        fnames = os.listdir(pjoin(src_root, folder))
        target_path = pjoin(tgt_root, folder)
        os.makedirs(target_path, exist_ok=True)
        num_frame = 0
        for name in tqdm(fnames):
            # target_path = pjoin(target_root, "m_"+name)
            try:
                # if dac
                # if os.path.exists(pjoin(target_path,"%s.npy"%name[:-4])):
                #     continue
                data = process_bvh_motion(pjoin(src_root, folder, name), 30, 30, feet_thre=0.11, shift_one_frame=True)
                np.save(pjoin(target_path,"%s.npy"%name[:-4]), data)
                num_frame += len(data)
                length_dict[name.split('_')[0]] += len(data)
            except Exception as e:
                print(name)
                print(e)

        print("Total duration: %04fh, average duration: %04fs"%(num_frame/30/3600, num_frame/30/len(fnames)))
        # for key, value in length_dict:
        #     print("Total duration of %s: %04fh"%(key, value/30/3600))
            