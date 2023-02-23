from controller import Robot, Motor, Supervisor
from scipy.optimize import fsolve
import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
# from mcc_kf.msg import detected_landmarks


from colorama import Fore, Back, Style


class DjiController (Supervisor):
    def __init__(self) -> None:
        Supervisor.__init__(self)

        rospy.init_node('webots_controller', anonymous=True)
        
        self.time_step = int(self.getBasicTimeStep())
        self.mot_vel = [0.0, 0.0, 0.0, 0.0]
        self.prev_pos = [0.0, 0.0, 0.0]


        # Intermittent
        self.pub_cam_prob = 0.3
        self.pub_uwb_prob = 0.3
        self.imu_noise = 0.001
        self.uwb_noise = 0.15

        # # Not intermittent
        # self.pub_cam_prob = 1.0
        # self.pub_uwb_prob = 1.0
        # self.imu_noise = 0.001
        # self.uwb_noise = 0.1      


        self.init_actuators()
        self.init_sensors()
        self.init_ros()

        self.init_cam_local()
        self.init_uwb_local()

    def init_actuators(self):
        self.m1 = self.getDevice("front right propeller")
        self.m2 = self.getDevice("front left propeller")
        self.m3 = self.getDevice("rear left propeller")
        self.m4 = self.getDevice("rear right propeller")        
        self.camera_roll_motor = self.getDevice("camera roll")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_yaw_motor = self.getDevice("camera yaw")
        self.camera_pitch_motor.setPosition(0.0) # 0.1
        motors = [self.m1, self.m2,
                  self.m3, self.m4]

        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)                         

    def init_sensors(self):
        self.imu = self.getDevice("inertial unit")  # Use to get the orientation of the robot
        self.imu.enable(self.time_step)             
        
        self.camera = self.getDevice("camera")      
        self.camera.enable(self.time_step)
        self.camera.recognitionEnable(self.time_step)

        self.gps = self.getDevice("gps")    # Use to get the true position of the robot
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")  # Use to get the true orientation of the robot
        self.gyro.enable(self.time_step)

    def init_ros(self):
        rospy.Subscriber("dji/motor_vel_command", Float64MultiArray, self.motor_vel_callback)
        self.imu_pub = rospy.Publisher("dji/imu", Imu, queue_size=1)                                # IMU data
        self.imu_true_pub = rospy.Publisher("dji/imu_true", Imu, queue_size=1)                      # True IMU data
        self.camera_pos_pub = rospy.Publisher("dji/camera_position", PointStamped, queue_size=1)    # Position of the camera in the world frame
        self.uwb_pos_pub = rospy.Publisher("dji/uwb_position", PointStamped, queue_size=1)                  # UWB data
        self.gps_pub = rospy.Publisher("dji/gps", PointStamped, queue_size=1)                       # True position of the robot
        self.gyro_pub = rospy.Publisher("dji/gyro", Float64MultiArray, queue_size=1)                # True angular velocity of the robot
        self.lin_vel_pub = rospy.Publisher("dji/lin_vel", Float64MultiArray, queue_size=1)          # True linear velocity of the robot

    def motor_vel_callback(self, data):
        self.mot_vel = data.data

    def init_cam_local(self):
        self.obj_ids = [1669, 1783, 1724, 1687, 1578, 1827, 1733, 1794, 1633, 1700, 1597, 1839, 1651, 1805, 1742, 1615, 1751, 1712, 1851, 1816]
        positions = []; node_names = []
        # Get list of all existing nodes in the simulation
        for id in self.obj_ids:
            node = self.getFromId(id)
            node_names.append(node.getTypeName())
            positions.append(np.array(node.getPosition()))
        # Create dictionary of node ids and positions
        self.obj_info = {
            'id': self.obj_ids,
            'name': node_names,
            'position': positions
        }

    def init_uwb_local(self):
        # Set position of the tags in the world frame
        self.tags = {
            'id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'position': [
                [3.0,  4.5, 1.0],
                [3.0, -4.5, 1.0],
                [ 6.0,  0.0, 1.0],
                [ 7.0,  4.5, 1.0],
                [11.0,  0.5, 1.0],
                [14.0,  4.5, 1.0],
                [12.0, -4.5, 1.0],
                [16.0, -0.5, 1.0],
                [19.0, -4.5, 1.0],
                [22.0, -0.5, 1.0],
                [26.0, -4.5, 1.0],
                [26.0,  4.5, 1.0],
                [31.0,  0.5, 1.0]
            ],
            'distance': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        }

    def camera_compute_position(self):
        true_pos = self.gps.getValues()
        # Compute the position of the camera in the world frame from info of each object
        position_ = []
        for id in self.detected_obj_info['id']:
            position_.append( self.obj_info['position'][self.obj_info['id'].index(id)] - self.detected_obj_info['position'][self.detected_obj_info['id'].index(id)] )

        # Compute the average position for x, y, z
        position_ = np.array(position_)
        position = np.mean(position_, axis=0)


        # # Print the estimated 'y' position of the camera for each landmark alongside the ID of the landmark (if the position is greater than 0.1 print in red, else in green using colorama)
        # print(f'*****************************************************')
        # for i in range(len(position_)):
        #     if abs(position_[i,0] - true_pos[0]) > 0.2:
        #         print(Fore.RED + 'Landmark ID:' + str(self.detected_obj_info['id'][i]) + 'Estimated x:' + str(position_[i,0] - true_pos[0]))
        #     elif abs(position_[i,0] - true_pos[0]) > 0.1:
        #         print(Fore.YELLOW + 'Landmark ID:' + str(self.detected_obj_info['id'][i]) + 'Estimated x:' + str(position_[i,0] - true_pos[0]))
        #     else:
        #         print(Fore.GREEN + 'Landmark ID:' + str(self.detected_obj_info['id'][i]) + 'Estimated x: ' + str(position_[i,0] - true_pos[0]))

        #     if abs(position_[i,1] - true_pos[1]) > 0.2:
        #         print(Fore.RED + 'Landmark ID:' + str(self.detected_obj_info['id'][i]) + 'Estimated y:' + str(position_[i,1] - true_pos[1]))
        #     elif abs(position_[i,1] - true_pos[1]) > 0.1:
        #         print(Fore.YELLOW + 'Landmark ID:' + str(self.detected_obj_info['id'][i]) + 'Estimated y:' + str(position_[i,1] - true_pos[1]))
        #     else:
        #         print(Fore.GREEN + 'Landmark ID:' + str(self.detected_obj_info['id'][i]) + 'Estimated y: ' + str(position_[i,1] - true_pos[1]))

        #     if abs(position_[i,2] - true_pos[2]) > 0.2:
        #         print(Fore.RED + 'Landmark ID:' + str(self.detected_obj_info['id'][i]) + 'Estimated z:' + str(position_[i,2] - true_pos[2]))
        #     elif abs(position_[i,2] - true_pos[2]) > 0.1:
        #         print(Fore.YELLOW + 'Landmark ID:' + str(self.detected_obj_info['id'][i]) + 'Estimated z:' + str(position_[i,2] - true_pos[2]))
        #     else:
        #         print(Fore.GREEN + 'Landmark ID:' + str(self.detected_obj_info['id'][i]) + 'Estimated z: ' + str(position_[i,2] - true_pos[2]))
        # print(f'-----------------------------------------------------')
    

        # Camera offset:
        x_offset = 0.0412774; y_offset = -0.00469654; z_offset = -0.00405862
        position[0] -= x_offset; position[1] -= y_offset; position[2] -= z_offset
        # Rotate the position of the camera to the world frame
        self.Rx = np.array([[1, 0, 0], [0, np.cos(self.imu.getRollPitchYaw()[0]), -np.sin(self.imu.getRollPitchYaw()[0])], [0, np.sin(self.imu.getRollPitchYaw()[0]), np.cos(self.imu.getRollPitchYaw()[0])]])
        self.Ry = np.array([ [np.cos(self.imu.getRollPitchYaw()[1]), 0, np.sin(self.imu.getRollPitchYaw()[1])], [0, 1, 0], [-np.sin(self.imu.getRollPitchYaw()[1]), 0, np.cos(self.imu.getRollPitchYaw()[1])]])
        self.Rz = np.array([[np.cos(self.imu.getRollPitchYaw()[2]), -np.sin(self.imu.getRollPitchYaw()[2]), 0], [np.sin(self.imu.getRollPitchYaw()[2]), np.cos(self.imu.getRollPitchYaw()[2]), 0], [0, 0, 1]])
        self.R = self.Rz @ self.Ry @ self.Rx

        position = self.R @ position


        return position

    def read_imu(self):
        imu_data = self.imu.getRollPitchYaw()
        # imu_noise = self.imu.getNoise()
        imu_msg = Imu()
        imu_msg.orientation.x = imu_data[0]
        imu_msg.orientation.y = imu_data[1]
        imu_msg.orientation.z = imu_data[2]
        imu_msg.orientation.w = 1
        # imu_msg.orientation_covariance = [imu_noise, 0, 0, 0, imu_noise, 0, 0, 0, imu_noise]
        imu_msg.header.stamp = rospy.Time.now()
        self.imu_true_pub.publish(imu_msg)
        # Add custom defined random noise to the imu measurements
        
        noise = np.random.normal(0, self.imu_noise, 3) 
        imu_msg.orientation.x += noise[0]
        imu_msg.orientation.y += noise[1]
        imu_msg.orientation.z += noise[2]
        self.imu_pub.publish(imu_msg)

    def read_camera(self):
        ## INTEGRATED COMPUTATION OF POSUTION OF THE CAMERA
        ## Extract position of objects detected by the camera.
        self.detected_objects = self.camera.getRecognitionObjects() # List of objects detected by the camera
        pos = []; orient = []; id = []; obj_model = []; image_pos = []
        for objects in self.detected_objects:
            pos_ = objects.getPosition(); orient_ = objects.getOrientation()
            pos.append(np.array(pos_))
            orient.append(np.array(orient_))
            id.append(objects.getId())
            obj_model.append(objects.getModel())
            image_pos.append(objects.getPositionOnImage())

        # Create dictionary of detected objects
        self.detected_obj_info = {
            'id': id,
            'name': obj_model,
            'position': pos,
        }

        # Compute the position of the robot using the detected objects
        if len(self.detected_obj_info['id']) > 0:   # Check if any objects have been detected
            cam_pos = self.camera_compute_position()

            ## Drone position computed from the camera-based object detection
            camera_pos_msg = PointStamped()
            camera_pos_msg.point.x = cam_pos[0]
            camera_pos_msg.point.y = cam_pos[1]
            camera_pos_msg.point.z = cam_pos[2]
            camera_pos_msg.header.frame_id = "world"
            camera_pos_msg.header.stamp = rospy.Time.now()

            # Publish the data with a probability of p
            
            if np.random.rand() < self.pub_cam_prob:
                self.camera_pos_pub.publish(camera_pos_msg)        

    def read_uwb(self):
        # Use true position to add noise based on true distance between uwb tags and the anchor (Quadrotor)
        pos = self.gps.getValues()
        pos[2] = 1.0   # Assume that the altitude is irrelevant (cause we only care about x-y position)
        max_dist = 10#10 # [m]
        available_tags = { 'position': [], 'distance': [] }
        
        # Add gaussian noise to the distance measurement based on the true distance between the uwb tag and the anchor
        for i in range(len(self.tags['position'])):
            distance = np.linalg.norm(self.tags['position'][i] - np.array(pos))
            if distance > max_dist:
                self.tags['distance'][i] = max_dist + 1   # Any value grater than max_dist [m] will be considered as an outlier later on
            else:
                ratio = distance/max_dist
                # noise = np.random.normal(0, 0.5*ratio, 1)
                noise = np.random.normal(0, self.uwb_noise*ratio, 1)
                self.tags['distance'][i] = distance + noise
                # Add the position and the distance to the available tags
                available_tags['position'].append(self.tags['position'][i])
                available_tags['distance'].append(self.tags['distance'][i])

        if len(available_tags['distance']) >= 3:
            # Perform trilateration to estimate the position of the quadrotor
            # Take the first 3 available tags with the smallest distance
            available_tags['distance'], available_tags['position'] = zip(*sorted(zip(available_tags['distance'], available_tags['position'])))
            available_tags['distance'] = list(available_tags['distance']); available_tags['position'] = list(available_tags['position'])
            available_tags['distance'] = available_tags['distance'][:3]; available_tags['position'] = available_tags['position'][:3]

            # Perform trilateration
            self.px = [available_tags['position'][0][0], available_tags['position'][1][0], available_tags['position'][2][0]]
            self.py = [available_tags['position'][0][1], available_tags['position'][1][1], available_tags['position'][2][1]]
            self.pz = [available_tags['position'][0][2], available_tags['position'][1][2], available_tags['position'][2][2]]
            self.r = [available_tags['distance'][0], available_tags['distance'][1], available_tags['distance'][2]]
            
            xyz0 = np.array([pos[0], pos[1], pos[2]], ndmin=1)
            # Reshape xyz into a (3,) array
            xyz0 = xyz0.reshape(3)

            sol = fsolve(self.pos_solver, xyz0)
            x = sol[0]; y = sol[1]

            ## Drone position computed from the camera-based object detection
            uwb_pos_msg = PointStamped()
            uwb_pos_msg.point.x = sol[0]; uwb_pos_msg.point.y = sol[1]; uwb_pos_msg.point.z = sol[2]    # 'z' is not used in the filter later on
            uwb_pos_msg.header.frame_id = "world"
            uwb_pos_msg.header.stamp = rospy.Time.now()

            # Publish the data with a probability of p
            if np.random.rand() < self.pub_uwb_prob:
                self.uwb_pos_pub.publish(uwb_pos_msg)

            # print(Fore.GREEN + 'x: {:+.4f} [m], y: {:+.4f} [m]'.format(x, y) + Fore.RESET)
        else:
            # print(Fore.RED + 'Not enough tags available. ({})'.format(len(available_tags['distance'])) + Fore.RESET)
            pass

    def pos_solver(self, xyz):
        self.x = xyz[0]; self.y = xyz[1]; self.z = xyz[2]
        
        self.f = (self.x - self.px[0])**2 + (self.y - self.py[0])**2 + (self.z - self.pz[0])**2 - self.r[0]**2
        self.g = (self.x - self.px[1])**2 + (self.y - self.py[1])**2 + (self.z - self.pz[1])**2 - self.r[1]**2
        self.h = (self.x - self.px[2])**2 + (self.y - self.py[2])**2 + (self.z - self.pz[2])**2 - self.r[2]**2

        outp = np.array([self.f, self.g, self.h], ndmin=1)
        # Reshape outp into a (3,) array
        outp = outp.reshape(3)

        return outp

    def read_gps(self):
        ## True position of the robot using the GPS
        gps_data = self.gps.getValues()
        position_msg = PointStamped()
        position_msg.point.x = gps_data[0]
        position_msg.point.y = gps_data[1]
        position_msg.point.z = gps_data[2]
        position_msg.header.frame_id = "world"
        position_msg.header.stamp = rospy.Time.now()
        self.gps_pub.publish(position_msg)

        ## True linear velocity of the robot using u = (x-x_prev)/dt
        vx = (gps_data[0] - self.prev_pos[0])/ (self.time_step*0.001)
        vy = (gps_data[1] - self.prev_pos[1])/ (self.time_step*0.001)
        vz = (gps_data[2] - self.prev_pos[2])/ (self.time_step*0.001)
        self.prev_pos = gps_data
        v = [vx, vy, vz]
        lin_vel_msg = Float64MultiArray()
        lin_vel_msg.data = v
        self.lin_vel_pub.publish(lin_vel_msg)

        ## Get true angular velocity of the robot
        gyro_data = self.gyro.getValues()
        gyro_msg = Float64MultiArray()
        gyro_msg.data = gyro_data
        self.gyro_pub.publish(gyro_msg)

    def fix_camera(self):
        # Rotate the camera so that its orientation is always rpy = [0, 0, 0] wrt the world frame. This is done to ensure that the camera is always pointing in the same direction.
        rpy = self.imu.getRollPitchYaw()

        self.camera_yaw_motor.setPosition(-rpy[2])
        self.camera_pitch_motor.setPosition(-rpy[1])
        self.camera_roll_motor.setPosition(-rpy[0])

    def read_sensors(self):
        self.read_imu()
        self.read_camera()
        self.read_uwb()
        self.read_gps()

    def run(self):
        while self.step(self.time_step) != -1:
            self.fix_camera()
            self.read_sensors()
            self.m1.setVelocity(self.mot_vel[0])   # Front right
            self.m2.setVelocity(self.mot_vel[1])   # Front left
            self.m3.setVelocity(self.mot_vel[2])   # Rear left
            self.m4.setVelocity(self.mot_vel[3])   # Rear right
def main():
    # Create the Robot instance.
    robot = DjiController()
    robot.run()

if __name__ == "__main__":
    main()    
























# If you want to use this code, you can use this one within the read_camera() function.
# ## EXTERNAL COMPUTATION OF POSITION OF THE CAMERA
# ## Extract the position within the image in pixels of the objects detected by the camera.
# self.detected_objects = self.camera.getRecognitionObjects() # List of objects detected by the camera
# pos = []; orient = []; id = []; obj_model = []; image_pos = []

# det_obj_msg = detected_landmarks()
# det_obj_msg.x = []; det_obj_msg.y = []; det_obj_msg.id = []

# for object in self.detected_objects:
#     pos_ = object.getPositionOnImage()
#     det_obj_msg.x.append(pos_[0])
#     det_obj_msg.y.append(pos_[1])
#     det_obj_msg.id.append(object.getId())
#     # pos.append(np.array(pos_))
#     # id.append(object.getId())
#     # obj_model.append(object.getModel())

# # Publish the message
# self.detected_landmarks_pub.publish(det_obj_msg)
# print(f'Published detected landmarks: \n{det_obj_msg}')