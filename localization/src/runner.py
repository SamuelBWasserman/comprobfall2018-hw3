import rospy
import matplotlib.pyplot as plt
from trajectory_parser import *

if __name__ == '__main__':
    start_position, heading_distance_list, ground_truth_list, noisy_heading_list, noisy_distance_list, scan_list = parse_trajectory(
        "trajectories_1.txt")


def graph(data, output):
    # coordinates to plot
    observed_x = list()
    observed_y = list()
    estimated_x = list()
    estimated_y = list()

    # add coordinates to list
    for d in data:
        observed_x.append(d[2])
        observed_y.append(d[3])

    for o in output:
        estimated_x.append(o[0])
        estimated_y.append(o[1])

    # plot observed and estimated data
    plt.plot(observed_x, observed_y, 'bo', linestyle='solid', label='observed')
    plt.plot(estimated_x, estimated_y, 'ro', linestyle='solid', label='estimated')
    plt.show()

    return
