from common.quaternion import *

import numpy as np
from einops import repeat

from common.quaternion_tool import Quaternions

from utils.paramUtil import kinematic_chain
from utils.utils import plot_3d_motion


def detect_contact(positions, fid_l, fid_r, thres):
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

def detect_slide_foot_contact(positions, fid_l, fid_r, vel_thres, h_thres):
    positions = positions.copy()
    
    velfactor, heightfactor = np.array([vel_thres, vel_thres]), np.array([h_thres+7, h_thres])
    position_local = positions - positions[:, 0:1]

    feet_l_x = (position_local[1:, fid_l, 0] - position_local[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (position_local[1:, fid_l, 2] - position_local[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]

    feet_l = (((0.5*feet_l_x + feet_l_y + 0.5*feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(float)


    feet_r_x = (position_local[1:, fid_r, 0] - position_local[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (position_local[1:, fid_r, 2] - position_local[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    
    feet_r = (((0.5*feet_r_x + feet_r_y + 0.5*feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(float)

    return feet_l, feet_r


def fk_local_quat(offsets, local_quats, root_pos, parents):
    global_pos = np.zeros(local_quats.shape[:-1] + (3,))
    global_pos[:, 0] = root_pos
    global_quats = np.zeros_like(local_quats)
    global_quats[:, 0] = local_quats[:, 0]

    offsets = repeat(offsets, 'j d -> i j d', i=len(local_quats))

    for i in range(1, len(parents)):
        global_quats[:, i] = qmul_np(global_quats[:, parents[i]], local_quats[:, i])
        global_pos[:, i] = qrot_np(global_quats[:, parents[i]], offsets[:, i]) + global_pos[:, parents[i]]
    return global_quats, global_pos


def lerp(a, l, r):
    return (1 - a) * l + a * r

def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1
    
def remove_fs(glb, foot_contact, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True):
    
    foot = foot_contact.transpose(1, 0)
    glb = glb.copy()

    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
    
    sort_height = np.sort(foot_heights)
    temp_len = len(sort_height)
    floor_height = np.mean(sort_height[:int(0.1*temp_len)])
    # plot_3d_motion('./glb0.mp4', kinematic_chain, glb, 
    #                             title='None', fps=30, radius=100)
    glb[:, :, 1] -= floor_height
    # plot_3d_motion('./glb1.mp4', kinematic_chain, glb, 
    #                             title='None', fps=30, radius=100)
    for i, fidx in enumerate(fid):
        fixed = foot[i]  # [T]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            # avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                # avg += glb[t, fidx].copy()
            # avg /= (t - s + 1)
            avg = glb[s:t+1, fidx].mean(axis=0)
            max_fs_dist = np.linalg.norm(glb[s:t+1, fidx, [0, 2]] - avg[[0, 2]], axis=-1).max()
            # print(s, t, max_fs_dist)
            att = 0
            while  max_fs_dist > 7.0 and att < 5:

                scale_factor = 0.75
                glb_root = glb[:, 0:1].copy()
                glb_local = glb - glb_root
                glb_root_xz_vel = glb_root[1:, :, [0, 2]] - glb_root[:-1, :, [0, 2]]
                glb_root_xz_vel[s:t] *= scale_factor
                glb_root_xz = np.cumsum(np.concatenate([glb_root[0:1, :, [0, 2]], glb_root_xz_vel], axis=0), axis=0)
                glb_root[:, :, [0, 2]] = glb_root_xz
                glb = glb_local + glb_root
                
                avg = glb[s:t+1, fidx].mean(axis=0)
                max_fs_dist = np.linalg.norm(glb[s:t+1, fidx, [0, 2]] - avg[[0, 2]], axis=-1).max()
                att += 1
                # print(fidx, s, t, max_fs_dist)

            
            if force_on_floor:
                avg[1] = 0.0

            # for j in range(s, t + 1):
            #     glb[j, fidx] = avg.copy()
            glb[s:t+1, fidx] = avg.copy()
            # print(s, t, avg)

            s = t + 1

        # print(glb[800])
        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    return glb


def children_list(parents):

    def joint_children(i):
        return [j for j, p in enumerate(parents) if p == i]

    return list(map(lambda j: np.array(joint_children(j)), joints(parents)))


def joints(parents):
    return np.arange(len(parents), dtype=int)



class BasicInverseKinematics:
    """
    Basic Inverse Kinematics Solver

    This is an extremely simple full body IK
    solver.

    It works given the following conditions:

        * All joint targets must be specified
        * All joint targets must be in reach
        * All joint targets must not differ
          extremely from the starting pose
        * No bone length constraints can be violated
        * The root translation and rotation are
          set to good initial values

    It works under the observation that if the
    _directions_ the joints are pointing toward
    match the _directions_ of the vectors between
    the target joints then the pose should match
    that of the target pose.

    Therefore it iterates over joints rotating
    each joint such that the vectors between it
    and it's children match that of the target
    positions.

    Parameters
    ----------

    animation : Animation
        animation input

    positions : (F, J, 3) ndarray
        target positions for each frame F
        and each joint J

    iterations : int
        Optional number of iterations.
        If the above conditions are met
        1 iteration should be enough,
        therefore the default is 1

    silent : bool
        Optional if to suppress output
        defaults to False
    """

    def __init__(self, animation, positions, iterations=1, silent=True):

        self.animation = animation
        self.positions = positions
        self.iterations = iterations
        self.silent = silent

    def __call__(self):

        children = children_list(self.animation.parents)

        for i in range(self.iterations):

            for j in joints(self.animation.parents):

                c = np.array(children[j])
                if len(c) == 0: continue


                anim_rotations, anim_positions = fk_local_quat(self.animation.offsets, self.animation.rotations, self.animation.positions[:, 0], self.animation.parents)
                
                anim_rotations = Quaternions(anim_rotations)
                jdirs = anim_positions[:, c] - anim_positions[:, np.newaxis, j]
                ddirs = self.positions[:, c] - anim_positions[:, np.newaxis, j]

                jsums = np.sqrt(np.sum(jdirs ** 2.0, axis=-1)) + 1e-10
                dsums = np.sqrt(np.sum(ddirs ** 2.0, axis=-1)) + 1e-10

                jdirs = jdirs / jsums[:, :, np.newaxis]
                ddirs = ddirs / dsums[:, :, np.newaxis]

                angles = np.arccos(np.sum(jdirs * ddirs, axis=2).clip(-1, 1))
                axises = np.cross(jdirs, ddirs)
                axises = -anim_rotations[:, j, np.newaxis] * axises
                # print(np.linalg.norm())

                rotations = Quaternions.from_angle_axis(angles, axises)
                # print(rotations)

                if rotations.shape[1] == 1:
                    averages = rotations[:, 0]
                else:
                    averages = Quaternions.exp(rotations.log().mean(axis=-2))
    
                # print(self.animation.rotations[:, j].shape, averages.qs.shape)            
                self.animation.rotations[:, j] = qmul_np(self.animation.rotations[:, j], averages.qs)
                # print(averages)

            if not self.silent:
                anim_rotations, anim_positions = fk_local_quat(self.animation.offsets, self.animation.rotations, self.animation.positions[:, 0], self.animation.parents)
                error = np.mean(np.sum((anim_positions - self.positions) ** 2.0, axis=-1) ** 0.5)
                print('[BasicInverseKinematics] Iteration %i Error: %f' % (i + 1, error))

        return self.animation


def foot_lock(anim, foot_contact=None):

    # Forward Kinematics
    _, global_pos = fk_local_quat(anim.offsets, anim.rotations, anim.positions[:, 0], anim.parents)
    # plot_3d_motion('./glb-1.mp4', kinematic_chain, global_pos, 
    #                             title='None', fps=30, radius=100)
    # Detect Contact Steps
    if foot_contact is None:
        feet_l, feet_r = detect_slide_foot_contact(global_pos.copy(), [18, 19], [22, 23], 1.5, 2.0)
        foot_contact = np.concatenate([feet_l, feet_r], axis=-1)

    # plot_3d_motion('./glb.mp4', kinematic_chain, global_pos, 
    #                             title='None', fps=30, radius=100)
    # Foot Locking
    ik_pos = remove_fs(global_pos[1:].copy(), foot_contact, fid_l=(18, 19), fid_r=(22, 23), interp_length=5, force_on_floor=True)

    # Inverse Kinematics
    new_anim = anim[1:].copy()
    new_anim.positions[:, 0] = ik_pos[:, 0]
    
    ik_solver = BasicInverseKinematics(new_anim, ik_pos, 20, silent=True)
    new_anim = ik_solver()

    upper_body_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    new_anim.rotations[:, upper_body_joint_indices] = anim.rotations[1:, upper_body_joint_indices]

    return new_anim, global_pos, ik_pos