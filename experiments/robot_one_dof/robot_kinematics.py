# Kinematics under pinocchio

import numpy as np
import pinocchio as pino

import os
env_dir = os.path.dirname(__file__)


class Kinematics(object):

    def __init__(self, urdf_file):
        self.model = pino.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()
        
        self.num_joints = self.model.nq
        self.current_joint_pos = None

    def random_configuration(self):
        return pino.randomConfiguration(self.model)
    
    def print_fk(self):
        for name, oMi in zip(self.model.names, self.data.oMi):
            print(("{:<24} : {: .2f} {: .2f} {: .2f}"
                .format( name, *oMi.translation.T.flat )))

    def forward(self, joint_position):
        if np.any(self.current_joint_pos != joint_position):
            pino.forwardKinematics(self.model, self.data, joint_position)
            pino.computeJointJacobians(self.model, self.data, joint_position)
            pino.updateFramePlacements(self.model, self.data)
            self.current_joint_pos = joint_position.copy()
        return self.current_joint_pos

    def inverse_pos(self, desired_position, initial_position, frame_idx):
        q_cur = initial_position.copy()
        success = False
        eps = 1e-3
        dt = 0.1
        damp = 1e-6
        for i in range(1000):
            pino.forwardKinematics(self.model, self.data, q_cur)
            pino.updateFramePlacements(self.model, self.data)
            J = pino.computeFrameJacobian(self.model, self.data, q_cur, frame_idx, pino.LOCAL_WORLD_ALIGNED)[:3]

            print(self.data.oMf[frame_idx].translation)
            pos_err = self.data.oMf[frame_idx].translation - desired_position
            if np.linalg.norm(pos_err) < eps:
                success = True
                break

            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(3), pos_err))
            q_cur += v * dt

        idx = np.where(q_cur > self.model.upperPositionLimit)
        q_cur[idx] -= np.pi * 2
        idx = np.where(q_cur < self.model.lowerPositionLimit)
        q_cur[idx] += np.pi * 2
        if success and (np.any(q_cur < self.model.lowerPositionLimit) or np.any(q_cur > self.model.upperPositionLimit)):
            success = False
        return success, q_cur

    def get_frame(self, idx):
        return self.data.oMf[idx]

    def get_jacobian(self, idx, frame=pino.LOCAL_WORLD_ALIGNED):
        return pino.getFrameJacobian(self.model, self.data, idx, frame)

    @property
    def position_lower_limit(self):
        return self.model.lowerPositionLimit

    @property
    def position_upper_limit(self):    
        return self.model.upperPositionLimit


def test_kinematics():
    kinematics = Kinematics(env_dir + '/models/one_dof.urdf')
    
    q = np.array([np.pi/2,0])
    
    kinematics.forward(q)
    tcp_frame_id = kinematics.model.getFrameId("joint_tcp")
    tcp_pose = kinematics.get_frame(tcp_frame_id)
    
    # print(q)
    # print(tcp_frame_id)
    print(tcp_pose.translation.T)

if __name__ == '__main__':
    test_kinematics()
