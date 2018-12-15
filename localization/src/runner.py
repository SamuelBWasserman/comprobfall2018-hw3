#!/usr/bin/env python

import matplotlib.pyplot as plt
import trajectory_parser
import map_parser
import numpy as np
from numpy.random import uniform
from prt_filtr import *

"""Graph the ground truth data against the estimated positions of the turtlebot"""
def graph(data, output):
    # coordinates to plot
    ground_truth_x = list()
    ground_truth_y = list()
    estimated_x = list()
    estimated_y = list()

    # add coordinates to list
    for d in data:
        ground_truth_x.append(d.pose.position.x.data)
        ground_truth_y.append(d.pose.position.y.data)

    for o in output:
        estimated_x.append(o[0])
        estimated_y.append(o[1])

    # plot observed and estimated data
    plt.plot(ground_truth_x, ground_truth_y, 'bo', linestyle='solid', label='Ground Truth')
    plt.plot(estimated_x, estimated_y, 'ro', linestyle='solid', label='Estimated')
    plt.show()

    return

def create_uniform_particles(x_range, y_range, heading_range, n):
    particles = np.empty((n, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=n)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=n)
    particles[:, 2] = uniform(heading_range[0], heading_range[1], size=n)
    particles[:,2] %= 2 * np.pi
    return particles


if __name__ == '__main__':
    print "Running"
    # Extract relevant information from map and trajectory files
    start_position, heading_distance_list, ground_truth_list, noisy_heading_distance_list, scan_list = trajectory_parser.parse_trajectory("trajectories_1.txt")
    corners, obstacles, num_obstacles = map_parser.parse_map("map_1.txt")
    estimated_positions = list()
    print obstacles[0][0][0]
    # Create array of obstacles
    obs = []
    for obstacle in obstacles:
        for point in obstacle:
            obs.append(point)

    # Convert the Float32's in the trajectory lists to float to make the algorithm more readable
    for j in range(len(noisy_heading_distance_list)):
        noisy_heading_distance_list[j] = [float(noisy_heading_distance_list[j][0].data), float(noisy_heading_distance_list[j][1].data)]

    for k in range(len(scan_list)):
        for scan in scan_list[k]:
            scan = float(scan.data)

    # Loop over every control
    state = start_position
    for i in range(len(noisy_heading_distance_list)):
        control = heading_distance_list[i]
        observation_scan = scan_list[i]

        # Run particle filter to get estimated pose
        particles = create_uniform_particles((-10, 10), (-10, 10), (-1*np.pi, np.pi), num_obstacles)

        run_filter(particles, noisy_heading_distance_list, scan_list, obstacles)

    # Graph result against ground truth
    graph(ground_truth_list, estimated_positions)

