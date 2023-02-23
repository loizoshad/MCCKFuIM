#!/usr/bin/env python3

import numpy as np
from scipy import signal

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped

class TrueState():
    def __init__(self):
        self.Z = np.zeros((12, 1))  # Measurement vector
        self.gps_sub = rospy.Subscriber('dji/gps', PointStamped, self.gps_callback)        
        self.imu_sub = rospy.Subscriber('dji/imu_true', Imu, self.imu_callback)
        self.lin_vel_sub = rospy.Subscriber('dji/lin_vel', Float64MultiArray, self.lin_vel_callback)
        self.gyro_sub = rospy.Subscriber('dji/gyro', Float64MultiArray, self.gyro_callback)
    
        self.state_pub = rospy.Publisher('dji/true_state', Float64MultiArray, queue_size=10)

    def gps_callback(self, msg):
        self.Z[0] = msg.point.x
        self.Z[1] = msg.point.y
        self.Z[2] = msg.point.z

    def imu_callback(self, msg):
        self.Z[3] = msg.orientation.x
        self.Z[4] = msg.orientation.y
        self.Z[5] = msg.orientation.z

    def lin_vel_callback(self, msg):
        self.Z[6] = msg.data[0]
        self.Z[7] = msg.data[1]
        self.Z[8] = msg.data[2]

    def gyro_callback(self, msg):
        self.Z[9] = msg.data[0]
        self.Z[10] = msg.data[1]
        self.Z[11] = msg.data[2]


    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            
            # # Pose
            # print(f'*************************************************************')
            # print(f'|    x    |    y    |    z    |   roll  |  pitch  |   yaw   |')
            # print(f'| {self.Z[0][0]:+.4f} | {self.Z[1][0]:+.4f} | {self.Z[2][0]:+.4f} | {self.Z[3][0]:+.4f} | {self.Z[4][0]:+.4f} | {self.Z[5][0]:+.4f} |')
            # print(f'-------------------------------------------------------------')   

            # Velocities
            # print(f'*************************************************************')
            # print(f'|    x    |    y    |    z    |   roll  |  pitch  |   yaw   |')
            # print(f'| {self.Z[6][0]:+.4f} | {self.Z[7][0]:+.4f} | {self.Z[8][0]:+.4f} | {self.Z[9][0]:+.4f} | {self.Z[10][0]:+.4f} | {self.Z[11][0]:+.4f} |')
            # print(f'-------------------------------------------------------------')   

            # Convert self.x to Float64MultiArray ROS message
            msg = Float64MultiArray()
            temp = self.Z
            msg.data = temp.flatten().tolist()
            self.state_pub.publish(msg)

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('true_state', anonymous=True)
    true_state = TrueState()
    true_state.run()