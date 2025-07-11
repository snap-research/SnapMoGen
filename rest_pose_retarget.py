import numpy as np
from os.path import join as pjoin
from utils import bvh_io
from einops import repeat
from common import animation as anim
from common.quaternion import *
import os
from tqdm import tqdm
# import sys
# sys.path.append('../')


def rest_pose_correction(rotation, offset_tgt, offset_src, parents):
    for i in range(rotation.shape[1]):
        # We can not do rotation correction on joint with multi-child
        if parents[i] in [-1, 0, 3, 15]:
            continue
        rot_btw = qbetween_np(offset_tgt[i], offset_src[i])
        rotation[:, parents[i]] = qmul_np(rotation[:, parents[i]], repeat(rot_btw, 'j -> k j', k=len(rotation)))
    return rotation


def fk_local_quat(local_quats, parents):
        global_quats = np.zeros_like(local_quats)
        global_quats[:, 0] = local_quats[:, 0]

        for i in range(1, len(parents)):
            global_quats[:, i] = qmul_np(global_quats[:, parents[i]], local_quats[:, i])
        return global_quats


def fk_local_quat_pos(local_quats, parents, offsets, root_pos):
        global_pos = np.zeros(local_quats.shape[:-1] + (3,))
        global_pos[:, 0] = root_pos
        global_quats = np.zeros_like(local_quats)
        global_quats[:, 0] = local_quats[:, 0]

        # offsets = self.offsets.expand(local_quats.shape[0], -1, -1).float()
        offsets = repeat(offsets, 'j k -> i j k', i=len(local_quats))

        for i in range(1, len(parents)):
            global_quats[:, i] = qmul_np(global_quats[:, parents[i]], local_quats[:, i])
            global_pos[:, i] = qrot_np(global_quats[:, parents[i]], offsets[:, i]) + global_pos[:, parents[i]]
        return global_quats, global_pos


def global_to_local_quat(global_quat, parents):
        local_quat = np.zeros_like(global_quat)
        local_quat[:, 0] = global_quat[:, 0]

        for i in range(1, len(parents)):
            local_quat[:, i] = qmul_np(qinv_np(global_quat[:, parents[i]]), global_quat[:, i])
            # global_quats[:, i] = qmul(global_quats[:, self.parents[i]], local_quats[:, i])
        return local_quat


class RestPoseRetargeter:
    def __init__(self):
          self.A_pose_anim = bvh_io.load('./utils/A_Pose.bvh')
          self.T_pose_anim = bvh_io.load('./utils/T_Pose.bvh')

    def rest_pose_retarget(self, src_anim, tgt_rest='A'):
         assert tgt_rest in ['A', 'T'], "Not supported rest pose type"
         global_rot = fk_local_quat(src_anim.rotations, src_anim.parents)
         tgt_anim = self.A_pose_anim if tgt_rest == 'A' else self.T_pose_anim

         corr_global_rot = rest_pose_correction(global_rot, tgt_anim.offsets, src_anim.offsets, src_anim.parents)
         corr_local_rot = global_to_local_quat(corr_global_rot, src_anim.parents)
         new_anim = anim.Animation(corr_local_rot,
                                src_anim.positions,
                                tgt_anim.orients,
                                tgt_anim.offsets,
                                tgt_anim.parents,
                                tgt_anim.names,
                                tgt_anim.frametime)
         return new_anim
    
if __name__ == "__main__":
    # Example usage
    # src_anim = bvh_io.load('./gp_00064.bvh')
    retargeter = RestPoseRetargeter()
    # tgt_anim = retargeter.rest_pose_retarget(src_anim, tgt_rest='A')
    # bvh_io.save('./gp_00064_A.bvh', tgt_anim)
    src_dir = '/mnt/local-disk/renamed_bvhs'
    tgt_dir = '/mnt/local-disk/renamed_bvhs_A'

    for file in tqdm(os.listdir(src_dir)):
        if file.endswith('.bvh'):
            src_anim = bvh_io.load(pjoin(src_dir, file))
            tgt_anim = retargeter.rest_pose_retarget(src_anim, tgt_rest='A')
            # Save the retargeted animation
            save_path = pjoin(tgt_dir, file)
            # bvh_io.save(save_path, tgt_anim)
            # print(f"Saved retargeted animation to {save_path}")
            bvh_io.save(save_path,
                        tgt_anim, 
                        names=tgt_anim.names,
                        frametime=tgt_anim.frametime,
                        order='xyz', 
                        quater=True)