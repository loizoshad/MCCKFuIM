#! /usr/bin/env python3

import rospy
import math
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class PathGenerator:
    def __init__(self) -> None:
        self.path_publisher = rospy.Publisher("dji/path", Path, queue_size=10)
        self.path = Path()
        self.path.header.frame_id = "map"
        self.path.poses = []

        waypoints = [
            [0.0,   0.0],
            [4.0,   2.5],
            [9.5,  2.5],
            [13.0,  0.0]
        ]


        path = self.generate_path(waypoints)
        self.smoothed_waypoints = self.smooth_path(path, alpha = 0.05, beta = 0.95, tol = 0.000001, smoothing = True)
        # self.plot_path(path, self.smoothed_waypoints)

    def generate_path(self, waypoints):
        xs = []; ys = []
        for i in range(np.shape(waypoints)[0]-1):
            # Generate intermediate points between each pair of waypoints
            multiplier = 13#10 # Number of points per unit length [m]
            path_len = int(math.sqrt( (waypoints[i][0] - waypoints[i + 1][0])**2 + (waypoints[i][1] - waypoints[i + 1][1])**2 ))
            xs_temp = np.linspace( waypoints[i][0],  waypoints[i + 1][0], multiplier*path_len)
            ys_temp = np.linspace( waypoints[i][1],  waypoints[i + 1][1], multiplier*path_len)
            xs = xs + xs_temp.tolist()
            ys = ys + ys_temp.tolist()

        path = []
        for i in range(len(xs)):
            path.append([xs[i],ys[i]])     

        return path

    def smooth_path(self, path, alpha = 0.025, beta = 0.975, tol = 0.000001, smoothing = True):
        smooth = deepcopy(path)
        length = len(path[0])
        delta = tol
        while delta >= tol:
            delta = 0
            for i in range(1, len(smooth) - 1):
                for j in range(length):
                    x_i = path[i][j]
                    y_i = smooth[i][j]
                    y_prev = smooth[i - 1][j]
                    y_next = smooth[i + 1][j]
                    y_i_old = y_i
                    y_i += alpha*(x_i - y_i) + beta*(y_next + y_prev - (2*y_i))
                    smooth[i][j] = y_i
                    delta += abs(y_i - y_i_old)
        
        return smooth
                    
    def plot_path(self, path, smoothed_path):
        # Plot the original path
        plt.plot([x[0] for x in path], [x[1] for x in path], 'r--', label='Original Path')
        # Plot the smoothed path in the same plot
        plt.plot([x[0] for x in smoothed_path], [x[1] for x in smoothed_path], 'b--', label='Smoothed Path')
        plt.legend()
        plt.xlim(0, 30)
        plt.ylim(-5, 5)
        plt.grid(True)
        plt.xlabel('Position - x [m]')
        plt.ylabel('Position - y [m]')
        plt.show()

    def run(self):
        first_time = False
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            
            self.path.poses = []
            for i in range(len(self.smoothed_waypoints)):
                pose = PoseStamped()
                pose.pose.position.x = self.smoothed_waypoints[i][0]
                pose.pose.position.y = self.smoothed_waypoints[i][1]
                pose.pose.position.z = 1.0
                pose.pose.orientation.x = 0.0; pose.pose.orientation.y = 0.0; pose.pose.orientation.z = 0.0; pose.pose.orientation.w = 1.0
                self.path.poses.append(pose)

            if first_time:
                # Plot the smoothed path
                x = [x.pose.position.x for x in self.path.poses]
                y = [x.pose.position.y for x in self.path.poses]
                plt.plot(x, y, 'b--', label='Smoothed Path')
                plt.grid(True)
                plt.xlabel('Position - x [m]')
                plt.ylabel('Position - y [m]')
                plt.xlim(0, 30)
                plt.ylim(-5, 5)
                plt.show()
                first_time = False
                for i in range(len(self.smoothed_waypoints)):
                    print(f'[x, y]({i}) = [{self.smoothed_waypoints[i][0]}, {self.smoothed_waypoints[i][1]}]')
                


            self.path.header.stamp = rospy.Time.now()
            self.path_publisher.publish(self.path)
            rate.sleep()



if __name__ == "__main__":
    rospy.init_node("path_generator")
    path_generator = PathGenerator()
    path_generator.run()