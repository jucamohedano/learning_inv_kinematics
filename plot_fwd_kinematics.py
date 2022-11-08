# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

# FORWARD KINEMATICS
# 1. Get the Transformation matrices from frame {1} w.r.t. frame {0}, frame {2} w.r.t. frame {1}, and so on.
# 2. Multiply all frames together to get the position of the end effector frame {ee} w.r.t frame {0}
# 3. The result of the matrix multiplication is the forward kinematics of the robot arm. Now we can find the pose of
#    the {ee} w.r.t. {0} by substituting params theta_i, where i is the same number as the joints

def rotation_matrix(phi,theta,psi):

    # Pure rotation in X
    def Rx(phi):
        return np.matrix([[  1, 0           , 0           ],
                          [  0, np.cos(phi) ,-np.sin(phi) ],
                          [  0, np.sin(phi) , np.cos(phi)]], dtype=np.float16)
    
    # Pure rotation in y
    def Ry(theta):
        return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                            [ 0             , 1, 0            ],
                            [-np.sin(theta) , 0, np.cos(theta)]], dtype=np.float16)
    
    # Pure rotation in z
    def Rz(psi):
        return np.matrix([[ np.cos(psi), -np.sin(psi) , 0  ],
                            [ np.sin(psi) , np.cos(psi)  , 0  ],
                            [ 0           , 0            , 1 ]], dtype=np.float16)
    return Rx(phi)*Ry(theta)*Rz(psi)

class RobotArm3D:
    """
        Init the base of the robot at (0,0,0).
        >>> Homogeneous matrix representation.
        Shape is (1,4,4)
        [[[1,0,0,0]
            [0,1,0,0]
            [0,0,1,0]
            [0,0,0,1]]]
    """
    def __init__(self):
        position = np.array([0,0,0], dtype=np.float16).reshape(3,1)
        rot_axis = rotation_matrix(0,0,0)
        # create the 4x4 homogeneous matrix
        T = np.hstack((rot_axis, position))
        T = np.vstack((T, np.array([0,0,0,1])))
        # init joints with base of the robot
        self.joints = np.expand_dims(T, axis=0)
    
    @property
    def num_joints(self):
        return len(self.joints)

    def add_revolute_link(self, position, rot=False):
        '''creates a new joint 
            Args:
                position - position of the joint. Local position is (0,0,0)
                rot - rotation matrix 3x3
        '''

        point = np.dot(rot, position) # apply rotation to point
        T = np.hstack((rot, point))
        
        T = np.vstack((T, np.array([0,0,0,1])))
        T = np.expand_dims(T, axis=0)
        self.joints = np.vstack((self.joints, T))

    def get_ee_pose(self):
        m = self.joints[0]
        for joint in self.joints[1:]:
            m = np.dot(m, joint)
        return m

    def get_tf_in_base_frame(self, frame_i):
        """get any frame in global frame (robot base) coordinates
            Args:
                frame_i - index of joint to convert
        """
        m = self.joints[0]
        for joint in self.joints[1:frame_i+1]:
            m = np.dot(m, joint)
        return m

    def rotate_joint(self, joint_i, rot):
        """ rotate joint by the given rotation matrix and all the following joints
            that may suffer the rotatation
            Args:
                joint_i - joint index to rotate
                rot - rotation matrix to apply
        """
        rotated_point = np.dot(rot, self.joints[joint_i][:3,-1]).reshape(3,1)
        self.joints[joint_i] = np.vstack(
                                    (
                                        np.hstack((rot, rotated_point)), 
                                        np.array([0,0,0,1], dtype=np.float16)
                                    ))

        # rotate subsequent joints if it applies
        for joint_idx in range(joint_i+1, self.num_joints):
            rotated_point = np.dot(rot, self.joints[joint_idx][:3,-1])
            self.joints[joint_idx][:3,-1] = rotated_point

    def plot_robot(self):
        # plot robot arm with matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = []
        y = []
        z = []
        for joint in range(robot.num_joints):
            m = robot.get_tf_in_base_frame(joint)
            x.append(m[0][-1])
            y.append(m[1][-1])
            z.append(m[2][-1])
        ax.plot(x,y,z, marker=".", markeredgecolor="red")
        plt.show()


if __name__ == '__main__':
    robot = RobotArm3D()
    robot.add_revolute_link(np.array([0,0,1]).reshape(3,1), rotation_matrix(0,0,0))
    robot.add_revolute_link(np.array([0,0,1]).reshape(3,1), rotation_matrix(0,0,0))
    robot.add_revolute_link(np.array([0,0,1]).reshape(3,1), rotation_matrix(0,0,0))

    JOINT_1 = np.array([0, 2*np.pi]) # around z-axis
    JOINT_2 = np.array([0, np.pi]) # around x-axis
    JOINT_3 = np.array([0, np.pi])  # around y-axis
    JOINT_4 = np.array([0, 2*np.pi]) # around z-axis

    joint14step = np.linspace(start=JOINT_1[0], stop=JOINT_1[1], num=5)
    joint23step = np.linspace(start=JOINT_2[0], stop=JOINT_2[1], num=5)
    
    joints_values = []
    with open("test.npy", "wb") as f:
        for step_i in joint14step:
            robot.rotate_joint(0, rotation_matrix(0,0,step_i))
            for step_j in joint23step:
                robot.rotate_joint(1, rotation_matrix(step_j,0,0))
                for step_k in joint23step:
                    robot.rotate_joint(2, rotation_matrix(0,step_k,0))
                    for step_r in joint14step:
                        robot.rotate_joint(3, rotation_matrix(0,0,step_r))
                        joints_and_ee = np.vstack((robot.joints, np.expand_dims(robot.get_ee_pose(), axis=0)))
                        joints_values.append(joints_and_ee)
        
        np.save(f, joints_values)