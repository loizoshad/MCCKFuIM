#!/usr/bin/env python3

import numpy as np
from scipy import signal
import math

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped
from colorama import Fore, Back, Style

class MCKF():
    def __init__(self):
        self.init_model()
        self.init_weights()
        self.init_ros()

        self.uref = np.array([ (-69.35174258)**2, (69.22467804)**2, (-68.94344922)**2, (69.07103220)**2], ndmin=2).T # Experimentally determined

        self.cam_pos = [0.0, 0.0, 0.0]
        self.uwb_pos = [0.0, 0.0, 0.0]
        self.gps_pos = [0.0, 0.0, 0.0]

        # self.cam_available = False; self.cam_ctr = 10
        self.cam_available_x = False; self.cam_ctr_x = 10
        self.cam_available_y = False; self.cam_ctr_y = 10
        self.uwb_available = False; self.uwb_ctr = 10


    def init_ros(self):
        self.cam_pos_sub = rospy.Subscriber('dji/camera_position', PointStamped, self.cam_pos_callback)
        self.uwb_pos_sub = rospy.Subscriber('dji/uwb_position', PointStamped, self.uwb_pos_callback)
        self.gps_sub = rospy.Subscriber('dji/gps', PointStamped, self.gps_callback)
        self.imu_sub = rospy.Subscriber('dji/imu', Imu, self.imu_callback)
        self.state_pub = rospy.Publisher('dji/est_state', Float64MultiArray, queue_size=1)
        self.log_data_pub = rospy.Publisher('dji/log/mckf', Float64MultiArray, queue_size=1)
        self.ctrl_inp_sub = rospy.Subscriber('dji/control_input', Float64MultiArray, self.ctrl_inp_callback)

    def ctrl_inp_callback(self, msg):
        self.U = np.array([msg.data[0], msg.data[1], msg.data[2], msg.data[3]], ndmin=2).T
        # Take into account the reference input needed to hover (Equilibrium point)
        self.U = self.U - self.uref

    def cam_pos_callback(self, msg):
        # # Check if the new measurement is reaasonably close to the previous one

        if abs(msg.point.x - self.x[0]) <= 0.15:
            self.Z[0] = msg.point.x
            self.cam_pos[0] = msg.point.x
            self.cam_pos[2] = msg.point.z
            self.cam_available_x = True    
            self.cam_ctr_x = 0
        if abs(msg.point.y - self.x[1]) <= 0.15:
            self.Z[1] = msg.point.y
            self.cam_pos[1] = msg.point.y
            self.cam_pos[2] = msg.point.z
            self.cam_available_x = True
            self.cam_ctr_x = 0


    def uwb_pos_callback(self, msg):
        self.Z[3] = msg.point.x
        self.Z[4] = msg.point.y
        self.uwb_available = True
        self.uwb_ctr = 0

    def gps_callback(self, msg):
        self.Z[3] = msg.point.x
        self.Z[4] = msg.point.y

        self.Z[2] = msg.point.z
        self.Z[5] = msg.point.z
        self.gps_pos[0] = msg.point.x
        self.gps_pos[1] = msg.point.y
        self.gps_pos[2] = msg.point.z

    def imu_callback(self, msg):
        self.Z[6] = msg.orientation.x
        self.Z[7] = msg.orientation.y
        self.Z[8] = msg.orientation.z

    def init_model(self):
        self.n = 12                             # Number of states
        self.ni = 4                             # Number of inputs
        self.nm = 9                             # Number of measurements
        g = 9.81                                # Gravity [m/s^2]
        dt = 0.01                               # Time step [s]
        b = 0.0002597                           # Thrust factor (Experimentally determined)
        d = 0.000256                            # Torque factor (Experimentally determined)
        self.m = 0.5069                         # Mass of Mavic 2 Pro [kg]
        Ix = 0.000913855                        # Inertia of Mavic 2 Pro in x-axis [kg*m^2] (From Webots model)
        Iy = 0.00236242                         # Inertia of Mavic 2 Pro in y-axis [kg*m^2] (From Webots model)
        Iz = 0.00279965                         # Inertia of Mavic 2 Pro in z-axis [kg*m^2] (From Webots model)
        l1 = 0.1609554879 * math.sin(math.pi/4) # Distance in x,y from the CoG to the propeller of the front right motor [m]
        l2 = 0.1609554879 * math.sin(math.pi/4) # Distance in x,y from the CoG to the propeller of the front left motor [m]
        l3 = 0.2205957181 * math.sin(math.pi/4) # Distance in x,y from the CoG to the propeller of the rear left motor [m]
        l4 = 0.2205957181 * math.sin(math.pi/4) # Distance in x,y from the CoG to the propeller of the rear right motor [m]

        # State matrix
        A = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   # x
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   # y
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # z
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # rollx
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # pitch
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # yaw
            [0, 0, 0, 0, g, 0, 0, 0, 0, 0, 0, 0],   # x_dot
            [0, 0, 0, -g, 0, 0, 0, 0, 0, 0, 0, 0],   # y_dot
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # z_dot
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # roll_dot
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # pitch_dot
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # yaw_dot
        ])

        B = np.array([
            [0, 0, 0, 0],   # x
            [0, 0, 0, 0],   # y
            [0, 0, 0, 0],   # z
            [0, 0, 0, 0],   # roll
            [0, 0, 0, 0],   # pitch
            [0, 0, 0, 0],   # yaw
            [0, 0, 0, 0],   # x_dot
            [0, 0, 0, 0],   # y_dot
            [b/self.m, b/self.m, b/self.m, b/self.m],   # z_dot
            [-b*l1/Ix, b*l2/Ix, b*l3/Ix, -b*l4/Ix],   # roll_dot
            [-b*l1/Iy, -b*l2/Iy, b*l3/Iy, b*l4/Iy],   # pitch_dot
            [d/Iz, -d/Iz, d/Iz, -d/Iz]    # yaw_dot
        ])

        C = np.array([  # Output matrix
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Camera-based localization for x
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Camera-based localization for y
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Camera-based localization for z
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # UWB-based localization for x
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # UWB-based localization for y
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # UWB-based localization for z
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # IMU roll
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # IMU pitch
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]    # IMU yaw            
        ])
        
        self.E = np.eye(self.nm)                    # USed for intermittent measurement handling
        self.C_aug = np.zeros((self.nm, self.nm))

        D = np.zeros((self.nm, self.ni))            # Feedforward matrix

        # Discritized state space representation
        self.A, self.B, self.C, self.D, dt = signal.cont2discrete((A, B, C, D), dt)

        self.x = np.zeros((self.n, 1))      # Initial state
        self.P = np.eye(self.n)             # Initial covariance
        self.Z = np.zeros((self.nm, 1))     # Initial measurement
        self.U = np.zeros((self.ni, 1))     # Initial input

        self.sigma = 40

    def init_weights(self):
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.R = np.diag([1.0, 1.0, 0.0000001, 0.01, 0.01, 0.0000001, 0.0001, 0.0001, 0.0001])

    def predict(self):
        self.x_old = self.x
        self.x = self.A@self.x + self.B@self.U
        self.P = self.A@self.P@self.A.T + self.Q

    def update(self):

        self.C_aug = self.C

        R_inv = np.linalg.inv(self.R)
        innov_num = self.Z - self.C_aug@self.x
        innov_den = self.x - (self.A@self.x_old + self.B@self.U)
        normm_num = np.linalg.norm(innov_num)
        normm_den = np.linalg.norm(innov_den)
        num = np.exp(-(normm_num**2) / (2*(self.sigma**2)))
        den = np.exp(-(normm_den**2) / (2*(self.sigma**2)))
        Gkernel = num/den
        K = Gkernel * np.linalg.inv(np.linalg.inv(self.P) + Gkernel * self.C_aug.T @ R_inv @ self.C_aug) @ self.C_aug.T @ R_inv
        self.x = self.x + K@(innov_num)
        self.P = (np.eye(self.n) - K@self.C_aug) @ self.P @ (np.eye(self.n) - K@self.C_aug).T + (K@self.R@K.T)

        # # Posteriori estimated state
        # print(f'*************UPDATE************')
        # print(f'|    x    |    y    |    z    |')
        # print(Fore.RED + '| {:07.4f} | '.format(self.x[0][0]) + Fore.RESET, end='') if abs(self.x[0][0] - self.gps_pos[0]) > 0.15 else print(Fore.GREEN + '| {:07.4f} | '.format(self.x[0][0]), end='')
        # print(Fore.RED + '{:07.4f} | '.format(self.x[1][0]) + Fore.RESET, end='') if abs(self.x[1][0] - self.gps_pos[1]) > 0.15 else print(Fore.GREEN + '{:07.4f} | '.format(self.x[1][0]), end='')
        # print(Fore.RED + '{:07.4f} |'.format(self.x[2][0]) + Fore.RESET) if abs(self.x[2][0] - self.gps_pos[2]) > 0.15 else print(Fore.GREEN + '{:07.4f} |'.format(self.x[2][0]))
        # print(f'-------------------------------')


    def run(self):
        rate = rospy.Rate(100) # 100
        while not rospy.is_shutdown():
            self.predict()
            self.update()            

            # Convert self.x to Float64MultiArray ROS message
            msg = Float64MultiArray()
            msg.data = [ self.x[0][0], self.x[1][0], self.x[2][0], self.x[3][0], self.x[4][0], self.x[5][0], self.x[6][0], self.x[7][0], self.x[8][0], self.x[9][0], self.x[10][0], self.x[11][0] ]
            
            self.state_pub.publish(msg)
            self.log_data_pub.publish(msg)
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('maximum_correntropy_kalman_filter', anonymous=True)
    mc_kf = MCKF()
    mc_kf.run()