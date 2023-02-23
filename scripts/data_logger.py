#! /usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path
from colorama import Fore, Style
from pynput import keyboard
import os

class DataLogger:
    def __init__(self) -> None:
        self.true_state_sub = rospy.Subscriber("dji/true_state", Float64MultiArray, self.true_state_cb)
        self.interm_kf_sub = rospy.Subscriber("dji/log/interm_kf", Float64MultiArray, self.interm_kf_cb)
        self.mckf_sub = rospy.Subscriber("dji/log/mckf", Float64MultiArray, self.mckf_cb)
        self.path_sub = rospy.Subscriber("dji/path", Path, self.path_cb)

        self.true_state = None
        self.interm_kf = None
        self.mckf = None

        self.true_state_data = []
        self.interm_kf_data = []
        self.mckf_data = []

        self.true_state_online = False
        self.interm_kf_online = False
        self.mckf_online = False

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        self.log_data = False

        self.received_path = False
        self.traj_x = []; self.traj_y = []

    def on_press(self, key):
        try:
            if key.char == 'l':
                print(f'Log data!')
                self.log_data = True
            elif key.char == 'q':
                # Exit the program
                exit()
            else:
                # Do nothing
                pass            
        except AttributeError:
            # print('special key {0} pressed'.format(key))
            pass

    def path_cb(self, msg):
        if not self.received_path:
            for i in range(len(msg.poses)):
                self.traj_x.append(msg.poses[i].pose.position.x)
                self.traj_y.append(msg.poses[i].pose.position.y)
        self.received_path = True

    def true_state_cb(self, msg):
        self.true_state = list(msg.data)
        self.true_state_online = True

    def interm_kf_cb(self, msg):
        self.interm_kf = list(msg.data)
        self.interm_kf_online = True

    def mckf_cb(self, msg):
        self.mckf = list(msg.data)
        self.mckf_online = True

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
    
            if self.true_state_online and self.interm_kf_online and self.mckf_online:
                                   
                if self.true_state is not None:
                    self.true_state_data.append(self.true_state)
                    # self.true_state = None
                else:
                    self.true_state_data.append([None] * 12)

                if self.interm_kf is not None:
                    self.interm_kf_data.append(self.interm_kf)
                    # self.interm_kf = None
                else:
                    self.interm_kf_data.append([None] * 12)

                if self.mckf is not None:
                    self.mckf_data.append(self.mckf)
                    # self.mckf = None
                else:
                    self.mckf_data.append([None] * 12)
                                  
                if self.log_data:
                    self.log_data = False
                    print("Plotting and saving data...")
                    # convert the list of lists to a numpy array
                    true_state_data = np.array(self.true_state_data)
                    interm_kf_data = np.array(self.interm_kf_data)
                    mckf_data = np.array(self.mckf_data)

                    # Set absolute path to be used for saving cause this is weird -.-
                    abs_path = os.path.dirname(os.path.abspath(__file__))
                    # Add to the absolute path the name of the directory 'results'
                    abs_path = os.path.join(abs_path, 'results/intermittent/not_smooth')
                    test_num = 'mckf_test_1'

                    # Plot the data
                    plt.figure()
                    plt.plot(true_state_data[:, 0], true_state_data[:, 1], label='Ground truth')
                    plt.plot(interm_kf_data[:, 0], interm_kf_data[:, 1], label='KF')
                    plt.plot(mckf_data[:, 0], mckf_data[:, 1], label='MCKF')
                    plt.legend(); plt.grid()
                    plt.xlabel('Position - x [m]'); plt.ylabel('Position - y [m]')
                    plt.title('Ground truth, KF, and MCKF')
                    # Save figure 
                    plt.savefig(os.path.join(abs_path, test_num + '.png'), format='png')
                    plt.show()

                    # In new figure plot the desired path alongside the true state
                    plt.figure()
                    plt.plot(self.traj_x, self.traj_y, label='Desired path')
                    plt.plot(true_state_data[:, 0], true_state_data[:, 1], label='Actual path')
                    plt.legend(); plt.grid()
                    plt.xlabel('Position - x [m]'); plt.ylabel('Position - y [m]')
                    plt.title('True state and desired path')
                    # Save figure
                    plt.savefig(os.path.join(abs_path, test_num + '_traj.png'), format='png')
                    plt.show()


                    # Compute MSE and RMSE for position x
                    kf_mse_x = np.mean((true_state_data[:, 0] - interm_kf_data[:, 0])**2)
                    kf_rmse_x = np.sqrt(kf_mse_x)
                    mckf_mse_x = np.mean((true_state_data[:, 0] - mckf_data[:, 0])**2)
                    mckf_rmse_x = np.sqrt(mckf_mse_x)
                    # Compute MSE for position y
                    kf_mse_y = np.mean((true_state_data[:, 1] - interm_kf_data[:, 1])**2)
                    kf_rmse_y = np.sqrt(kf_mse_y)
                    mckf_mse_y = np.mean((true_state_data[:, 1] - mckf_data[:, 1])**2)
                    mckf_rmse_y = np.sqrt(mckf_mse_y)
                    # Compute combined MSE in all 12 states
                    kf_mse = np.mean((true_state_data - interm_kf_data)**2)
                    kf_rmse = np.sqrt(kf_mse)
                    mckf_mse = np.mean((true_state_data - mckf_data)**2)
                    mckf_rmse = np.sqrt(mckf_mse)

                    
                    # Print RMSE
                    # print('KF RMSE x: {:.6f} m'.format(kf_rmse_x))
                    # print('KF RMSE y: {:.6f} m'.format(kf_rmse_y))                    
                    
                    print("MCKF RMSE x: {:.6f} m".format(mckf_rmse_x))
                    print("MCKF RMSE y: {:.6f} m".format(mckf_rmse_y))



                    # Save data to csv file
                    # Save in the absolute path given by the variable 'abs_path'
                    np.savetxt(os.path.join(abs_path, f'{test_num}_true_state.csv'), true_state_data, delimiter=',')
                    np.savetxt(os.path.join(abs_path, f'{test_num}_kf_est.csv'), interm_kf_data, delimiter=',')
                    np.savetxt(os.path.join(abs_path, f'{test_num}_mckf_est.csv'), mckf_data, delimiter=',')
                    np.savetxt(os.path.join(abs_path, f'{test_num}_desired_path.csv'), np.array([self.traj_x, self.traj_y]).T, delimiter=',')
                    # Save the MSE values to a csv files
                    np.savetxt(os.path.join(abs_path, f'{test_num}_mse.csv'), np.array([kf_mse_x, kf_mse_y, kf_mse, mckf_mse_x, mckf_mse_y, mckf_mse]), delimiter=',')                   
                    # Save the RMSE values to a csv file
                    np.savetxt(os.path.join(abs_path, f'{test_num}_rmse.csv'), np.array([kf_rmse_x, kf_rmse_y, kf_rmse, mckf_rmse_x, mckf_rmse_y, mckf_rmse]), delimiter=',')

                    # Reset lists
                    self.true_state_data = []
                    self.interm_kf_data = []
                    self.mckf_data = []
                    self.true_state_online = False
                    self.interm_kf_online = False
                    self.mckf_online = False

            rate.sleep()



if __name__ == "__main__":
    rospy.init_node("data_logger")
    data_logger = DataLogger()
    data_logger.run()
