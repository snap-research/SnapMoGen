from common.quaternion import *
import numpy
import torch

# Implemented in PyTorch-backend by default
class Skeleton:
    def __init__(self, offsets, parents, device):
        self.device = device
        if isinstance(offsets, numpy.ndarray):
            self.offsets = torch.from_numpy(offsets).to(device).float()
        self.parents = parents
        self.children = [[] for _ in range(len(parents))]
        for i in range(len(self.parents)):
            if self.parents[i] >= 0:
                self.children[self.parents[i]].append(i)

    '''
    Forward Kinematics from local quanternion based rotations
    local_quats: (b, nj, 4)
    root_pos: (b, 3)

    Note: make sure root joint is at the 1st entry
    '''
    def fk_local_quat(self, local_quats, root_pos):
        global_pos = torch.zeros(local_quats.shape[:-1] + (3,)).to(self.device)
        local_quats = local_quats.to(self.device)
        root_pos = root_pos.to(self.device)
        global_pos[:, 0] = root_pos
        global_quats = torch.zeros_like(local_quats).to(self.device)
        global_quats[:, 0] = local_quats[:, 0]

        offsets = self.offsets.expand(local_quats.shape[0], -1, -1).float()

        for i in range(1, len(self.parents)):
            global_quats[:, i] = qmul(global_quats[:, self.parents[i]], local_quats[:, i])
            global_pos[:, i] = qrot(global_quats[:, self.parents[i]], offsets[:, i]) + global_pos[:, self.parents[i]]
        return global_quats, global_pos
    
    def fk_local_quat_np(self, local_quats, root_pos):
        global_quats, global_pos = self.fk_local_quat(torch.from_numpy(local_quats).float(),
                                                      torch.from_numpy(root_pos).float())
        return global_quats.cpu().numpy(), global_pos.cpu().numpy()
    
    '''
    Forward Kinematics from global quanternion based rotations
    global_quats: (b, nj, 4)
    root_pos: (b, 3)

    Note: make sure root joint is at the 1st entry
    '''
    def fk_global_quat(self, global_quats, root_pos):
        global_pos = torch.zeros(global_quats.shape[:-1] + (3,)).to(self.device)
        global_pos[:, 0] = root_pos
        offsets = self.offsets.expand(global_quats.shape[0], -1, -1).float()

        for i in range(1, len(self.parents)):
            global_pos[:, i] = qrot(global_quats[:, self.parents[i]], offsets[:, i]) + global_pos[:, self.parents[i]]
        return global_pos
    
    def fk_global_quat_np(self, global_quats, root_pos):
        global_pos = self.fk_global_quat(torch.from_numpy(global_quats).float(),
                                         torch.from_numpy(root_pos).float())
        return global_pos.numpy()
    
    '''
    Forward Kinematics from local 6D based rotations
    local_cont6d: (b, nj, 6)
    root_pos: (b, 3)

    Note: make sure root joint is at the 1st entry
    '''
    def fk_local_cont6d(self, local_cont6d, root_pos):

        global_pos = torch.zeros(local_cont6d.shape[:-1]+(3,)).to(self.device)
        global_pos[:, 0] = root_pos

        local_cont6d_mat = cont6d_to_matrix(local_cont6d)
        global_cont6d_mat = torch.zeros_like(local_cont6d_mat).to(self.device)
        global_cont6d_mat[:, 0] = local_cont6d_mat[:, 0]
        offsets = self.offsets.expand(local_cont6d.shape[0], -1, -1).float()


        for i in range(1, len(self.parents)):

            global_cont6d_mat[:, i] = torch.matmul(global_cont6d_mat[:, self.parents[i]].clone(),
                                                   local_cont6d_mat[:, i])
            global_pos[:, i] = torch.matmul(global_cont6d_mat[:, self.parents[i]],
                                                offsets[:, i].unsqueeze(-1)).squeeze() + global_pos[:, self.parents[i]]
        return matrix_to_cont6D(global_cont6d_mat), global_pos

    def fk_local_cont6d_np(self, local_cont6d, root_pos):
        global_cont6d, global_pos = self.fk_local_cont6d(torch.from_numpy(local_cont6d).float(),
                                                         torch.from_numpy(root_pos).float())
        return global_cont6d.numpy(), global_pos.numpy()

    '''
    Forward Kinematics from global 6D based rotations
    global_cont6d: (b, nj, 6)
    root_pos: (b, 3)

    Note: make sure root joint is at the 1st entry
    '''
    def fk_global_cont6d(self, global_cont6d, root_pos):

        global_cont6d_mat = cont6d_to_matrix(global_cont6d)
        global_pos = torch.zeros(global_cont6d.shape[:-1] + (3,)).to(self.device)
        global_pos[:, 0] = root_pos
        offsets = self.offsets.expand(global_cont6d.shape[0], -1, -1).float()

        for i in range(1, len(self.parents)):
            global_pos[:, i] = torch.matmul(global_cont6d_mat[:, self.parents[i]],
                                            offsets[:, i].unsqueeze(-1)).squeeze() + global_pos[:, self.parents[i]]
        return global_pos

    def fk_global_cont6d_np(self, global_cont6d, root_pos):
        global_pos = self.fk_global_cont6d(torch.from_numpy(global_cont6d).float(),
                                          torch.from_numpy(root_pos).float())
        return global_pos.numpy()

    def global_to_local_quat(self, global_quat):
        local_quat = torch.zeros_like(global_quat).to(global_quat.device)
        local_quat[:, 0] = global_quat[:, 0]

        for i in range(1, len(self.parents)):
            local_quat[:, i] = qmul(qinv(global_quat[:, self.parents[i]]), global_quat[:, i])
            # global_quats[:, i] = qmul(global_quats[:, self.parents[i]], local_quats[:, i])
        return local_quat

    def global_to_local_quat_np(self, global_quat):
        local_quat = self.global_to_local_quat(torch.from_numpy(global_quat).float())
        return local_quat.numpy()