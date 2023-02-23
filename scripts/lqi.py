#! /usr/bin/env python3

import numpy as np
import math
import scipy.linalg as la
from scipy import signal
import control

import rospy
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path
import tf

class LQI:
    def __init__(self):
        self.mot_vel_pub = rospy.Publisher('dji/motor_vel_command', Float64MultiArray, queue_size=1)
        self.ctrl_inp_pub = rospy.Publisher('dji/control_input', Float64MultiArray, queue_size=1)
        self.true_state_sub = rospy.Subscriber('dji/true_state', Float64MultiArray, self.true_state_callback)
        self.est_state_sub = rospy.Subscriber('dji/est_state', Float64MultiArray, self.est_state_callback)
        self.path_sub = rospy.Subscriber('dji/path', Path, self.path_callback)
        self.traj_x = []; self.traj_y = []; self.traj_z = []; self.traj_yaw = []
        self.estimator_online = False        # This will be set to True when the state estimator is ready
        self.mot_vel = Float64MultiArray()
        self.mot_vel.data = [0, 0, 0, 0]
        self.ctrl_inp = Float64MultiArray()
        self.ctrl_inp.data = [0, 0, 0, 0]
        self.iter = 0
        
        self.new_path = False
        self.index = 0

        self.init_weights()
        self.init_model()

    def init_weights(self):
        self.Q = np.diag([1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,    1, 1, 1, 1])
        wr = 0.0001
        self.R = np.diag([wr, wr, wr ,wr])

    def init_model(self):
        self.n = 12                             # Number of states
        self.ni = 4                             # Number of inputs
        self.nr = 4                             # Number of tracked states
        self.nm = 12                            # Number of measurements
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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # roll
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

        C = np.eye(self.nm)

        D = np.zeros((self.nm, self.ni))

        E = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

        # Create augmented system
        A_aug = np.block([
            [A,    np.zeros((self.n, self.nr))],
            [-E@C, np.eye(self.nr)]
        ])

        B_aug = np.block([
            [B],
            [np.zeros((self.nr, self.ni))]
        ])

        C_aug = np.block([
            [C, np.zeros((self.nm, self.nr ))]
        ])

        # Discritized state space representation
        self.A, self.B, self.C, self.D, dt = signal.cont2discrete((A_aug, B_aug, C_aug, D), dt, method = 'zoh')
        # Solve the Riccati equation to obtain the optimal feedback gain
        self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.L = la.inv(self.B.T@self.P@self.B + self.R) @ self.B.T@self.P@self.A
        self.Lx = self.L[:, 0:self.n]   # Proportional
        self.Li = self.L[:, self.n:]    # Integral        

        self.r = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ndmin=2).T # State Reference
        self.ri = np.array([0.0, 0.0, 1.0, 0.0], ndmin=2).T # Reference integral [x, y, z, yaw]
        self.ie = np.array([0.0, 0.0, 0.0, 0.0], ndmin=2).T # Error
        self.x = np.array([0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ndmin=2).T # Initial state
        self.x_true = self.x
        self.x_est = self.x

    def true_state_callback(self, msg):
        self.x_true = np.array([msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5], msg.data[6], msg.data[7], msg.data[8], msg.data[9], msg.data[10], msg.data[11]], ndmin=2).T

    def path_callback(self, msg):
        '''
        msg is of the type nav_msgs/Path
        '''
        
        # Check if the new path is the same as the old path
        if not len(self.traj_x) == 0:
            if self.traj_x[0] == msg.poses[0].pose.position.x and self.traj_y[0] == msg.poses[0].pose.position.y and self.traj_z[0] == msg.poses[0].pose.position.z:
                pass
        else:
            self.traj_x = []; self.traj_y = []; self.traj_z = []; self.traj_yaw = []
            for i in range(len(msg.poses)):
                self.traj_x.append(msg.poses[i].pose.position.x)
                self.traj_y.append(msg.poses[i].pose.position.y)
                self.traj_z.append(1.0)

                # Convert quaternion to euler angles
                q = [msg.poses[i].pose.orientation.x, msg.poses[i].pose.orientation.y, msg.poses[i].pose.orientation.z, msg.poses[i].pose.orientation.w]
                euler = tf.transformations.euler_from_quaternion(q)
                self.traj_yaw.append(euler[2])
            self.new_path = True



    def est_state_callback(self, msg):
        self.x_est = np.array([msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5], msg.data[6], msg.data[7], msg.data[8], msg.data[9], msg.data[10], msg.data[11]], ndmin=2).T
        if not self.estimator_online:
            self.Q = np.diag([1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,    1, 1, 1, 1])
            wr = 0.0001            
            self.R = np.diag([wr, wr, wr ,wr])
            self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)
            self.L = la.inv(self.B.T@self.P@self.B + self.R) @ self.B.T@self.P@self.A
            self.Lx = self.L[:, 0:self.n]   # Proportional
            self.Li = self.L[:, self.n:]    # Integral
            self.estimator_online = True            

    def mot_speeds(self):
        mot_safe_speed = 60.0
        max_mot_vel = 100.0
        
        try:
            m1 = -min(max_mot_vel, math.sqrt( abs(self.u[0]) ) )
        except:
            m1 = -mot_safe_speed
        try:
            m2 = min(max_mot_vel, math.sqrt( abs(self.u[1]) ) )
        except:
            m2 = mot_safe_speed
        try:
            m3 = -min(max_mot_vel, math.sqrt( abs(self.u[2]) ))
        except:
            m3 = -mot_safe_speed
        try:
            m4 = min(max_mot_vel, math.sqrt( abs(self.u[3]) ))
        except:
            m4 = mot_safe_speed

        self.mot_vel.data = [m1, m2, m3, m4]
        self.ctrl_inp.data = [self.u[0], self.u[1], self.u[2], self.u[3]]


    def check_target_reached(self):
        # If the current position is within 0.02m of the target position, then the target is reached
        dist = math.sqrt( (self.x[0] - self.traj_x[self.index])**2 + (self.x[1] - self.traj_y[self.index])**2 + (self.x[2] - self.traj_z[self.index])**2 )

        if dist < 0.1:                                             # (For the 90 degree turn trajectory)
        # if dist < 0.05 or self.x[0] > self.traj_x[self.index]:    # (For the normal trajectory)
            if self.index < len(self.traj_x) - 1:
                self.index += 1
            
        self.r[0] = self.traj_x[self.index]
        self.r[1] = self.traj_y[self.index]
        self.r[2] = self.traj_z[self.index]
        self.r[5] = self.traj_yaw[self.index]

        self.ri[0] = self.traj_x[self.index]
        self.ri[1] = self.traj_y[self.index]
        self.ri[2] = self.traj_z[self.index]
        self.ri[3] = self.traj_yaw[self.index]


    def run(self):
        acc_pos_err_max = 0.1       # [m]
        acc_orient_err_max = 0.1   # [rad]
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            self.x = self.x_est if self.estimator_online else self.x_true
            xi = np.array([self.x[0], self.x[1], self.x[2], self.x[5]]) # Extract integral states

            if self.new_path:
                # Update reference if the target is reached
                self.check_target_reached()

            print(f'Goal: {self.ri.T}')

            # Anti-windup
            # Integral error (TODO: Keep this approach for the anti-windup at the time being)
            self.ie = self.ie + (self.ri - xi)     

            if self.ie[0] > acc_pos_err_max:
                self.ie[0] = acc_pos_err_max
            if self.ie[1] > acc_pos_err_max:
                self.ie[1] = acc_pos_err_max
            if self.ie[2] > acc_pos_err_max:
                self.ie[2] = acc_pos_err_max
            if self.ie[3] > acc_orient_err_max:
                self.ie[3] = acc_orient_err_max

            # Reset every 100 iterations
            if self.iter % 100 == 0:
                self.ie = np.array([0.0, 0.0, 0.0, 0.0], ndmin=2).T

            uref = np.array([ (-69.35174258)**2, (69.22467804)**2, (-68.94344922)**2, (69.07103220)**2], ndmin=2).T # Experimentally determined
            self.u = - self.Lx @ (self.x - self.r) - self.Li @ self.ie + uref

            # Convert the control inputs to motor velocities
            self.mot_speeds()

            # Publish the motor velocities and the control input signal for the Kalman filter
            self.mot_vel_pub.publish(self.mot_vel)
            self.ctrl_inp_pub.publish(self.ctrl_inp)

if __name__ == '__main__':
    rospy.init_node('LQI Controller', anonymous=True)
    lqi = LQI()
    lqi.run()



