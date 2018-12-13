import rospy
from gazebo_msgs.msg import ModelState
from turtlebot_ctrl.srv import TurtleBotControl
from turtlebot_ctrl.msg import TurtleBotScan
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from geometry_msgs.msg import *

start_position = [0, 0]
heading_distance_list = list()
ground_truth_list = list()
noisy_heading_list = list()
noisy_distance_list = list()
scan_list = list()
CHUNK_LEN = 30

""" Parses a trajectory input file and returns lists of corresponding data"""
def parse_trajectory(input_file):
    # Extract the start position of the robot
    with open(input_file, 'r') as input:
        lines = input.readlines()
        start_position = [int(lines[1].split()[1]), int(lines[2].split()[1])]

        # Used to keep track of the top of a reading chunk
        reading_num = 4

        # read every chunk of data
        while reading_num < len(lines) - CHUNK_LEN:
            # Append the heading and distance to the heading_distance_list
            heading_words = lines[reading_num].split()
            distance_words = lines[reading_num + 1].split()
            heading = Float32(heading_words[2])
            distance = Float32(distance_words[2])
            heading_distance_list.append((heading, distance))

            # Reading ground truth data
            # Assign Pose
            pose = Pose()
            # position
            pose.position.x = Float64(lines[reading_num + 6].split()[1])
            pose.position.y = Float64(lines[reading_num + 7].split()[1])
            pose.position.z = Float64(lines[reading_num + 8].split()[1])

            # orientation
            pose.orientation.x = Float64(lines[reading_num + 10].split()[1])
            pose.orientation.y = Float64(lines[reading_num + 11].split()[1])
            pose.orientation.z = Float64(lines[reading_num + 12].split()[1])
            pose.orientation.w = Float64(lines[reading_num + 13].split()[1])

            # Assign Twist
            twist = Twist()
            # linear
            twist.linear.x = Float64(lines[reading_num + 16].split()[1])
            twist.linear.y = Float64(lines[reading_num + 17].split()[1])
            twist.linear.z = Float64(lines[reading_num + 18].split()[1])

            # angular
            twist.angular.x = Float64(lines[reading_num + 20].split()[1])
            twist.angular.y = Float64(lines[reading_num + 21].split()[1])
            twist.angular.z = Float64(lines[reading_num + 22].split()[1])

            # Append ground truth to list
            model_state = ModelState()
            model_state.pose = pose
            model_state.twist = twist
            ground_truth_list.append(model_state)

            # Append noisy headings and distances to their list
            noisy_heading_list.append(Float32(lines[reading_num + 27].split()[1]))
            noisy_distance_list.append(Float32(lines[reading_num + 29].split()[1]))

            # Append scan data to list
            scan_data = lines[reading_num + 29][7:].split()
            scan_data_list = TurtleBotScan().ranges
            for data in scan_data:
                scan_data_list.append(Float32(data))
            scan_list.append(scan_data_list)

            reading_num = reading_num + 30

        return start_position, heading_distance_list, ground_truth_list, noisy_heading_list, noisy_distance_list, scan_list
