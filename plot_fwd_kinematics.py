# %matplotlib inline
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R


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
    """
    def __init__(self):
        position = np.array([0,0,0], dtype=np.float16).reshape(3,1)
        rot_axis = rotation_matrix(0,0,0)
        T = np.hstack((rot_axis, position))
        T = np.vstack((T, np.array([0,0,0,1])))
        self.joints = np.expand_dims(T, axis=0)
    
    @property
    def num_joints(self):
        return len(self.joints)

    def add_revolute_link(self, position, rot):
        '''
            Args:
                length of the link
                3D angles - thetaInit, betaInit, alphaInit
        '''
        T = np.hstack((rot, position))
        T = np.vstack((T, np.array([0,0,0,1])))
        T = np.expand_dims(T, axis=0)
        self.joints = np.vstack((self.joints, T))

    def get_ee_pose(self):
        m = self.joints[0]
        for joint in self.joints[1:]:
            m = np.dot(m, joint)
        return m

    def get_tf_in_base_frame(self, i):
        m = self.joints[0]
        for joint in self.joints[1:i+1]:
            m = np.dot(m, joint)
        return m

    def rotate_joint(self, link_i, rot):
        rotated_point = np.dot(rot, self.joints[link_i][:3,-1])
        print(rotated_point)
        self.joints[link_i][:3,-1] = rotated_point
        print(np.rint(self.joints[link_i]))

robot = RobotArm3D()
robot.add_revolute_link(np.array([0,0,1]).reshape(3,1), rotation_matrix(0,0,0))
robot.add_revolute_link(np.array([1,0,0]).reshape(3,1), rotation_matrix(0,0,0))
robot.add_revolute_link(np.array([1,0,0]).reshape(3,1), rotation_matrix(0,0,0))
robot.add_revolute_link(np.array([1,0,0]).reshape(3,1), rotation_matrix(0,0,0))
robot.add_revolute_link(np.array([0,0,1]).reshape(3,1), rotation_matrix(0,0,0))


# new_pose = robot.rotate_joint(5, rotation_matrix(np.pi,0,0))
# new_pose = robot.rotate_joint(4, rotation_matrix(0,0,-np.pi/2))


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
print()
ax.plot(x,y,z)
plt.show()
print(np.array((x,y,z)).T)
# Axes3D.plot()


# FORWARD KINEMATICS
# 1. Get the Transformation matrices from frame {1} w.r.t. frame {0}, frame {2} w.r.t. frame {1}, and so on.
# 2. Multiply all frames together to get the position of the end effector frame {ee} w.r.t frame {0}
# 3. The result of the matrix multiplication is the forward kinematics of the robot arm. Now we can find the pose of
#    the {ee} w.r.t. {0} by substituting params theta_i, where i is the same number as the joints

