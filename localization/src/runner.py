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

def test_motion_model(start_pos, gt_controls):
    pos = np.zeros((3,1))
    pos[0,0] = start_pos[0]
    pos[1,0] = start_pos[1]
    pos[2,0] = 0.0
    pos.astype(np.float64)
    print(pos)
    plist = list()
    plist.append(pos)
    print("INITIAL POS:", pos)
    for i in range(len(gt_controls)):
        controls = gt_controls[i]
        print controls
        #controls.append(float(gt_controls[i][0].data))
        #controls.append(float(gt_controls[i][1].data))
        next_pos = motion_model(pos, np.asarray(controls, dtype=np.float64))
        plist.append([next_pos[0,0], next_pos[1,0]])
        print("POS[", i, "]", next_pos)
        pos = next_pos
    return plist


if __name__ == '__main__':
    print "Running"
    # Extract relevant information from map and trajectory files
    start_position, heading_distance_list, ground_truth_list, noisy_heading_distance_list, scan_list = trajectory_parser.parse_trajectory("trajectories_2.txt")
    corners, obstacles, num_obstacles = map_parser.parse_map("map_2.txt")
    estimated_positions = list()
    # Create array of obstacles
    obs = []
    for obstacle in obstacles:
        for point in obstacle:
            obs.append(point)
    # Convert the Float32's in the trajectory lists to float to make the algorithm more readable
    for j in range(len(noisy_heading_distance_list)):
        noisy_heading_distance_list[j] = [float(noisy_heading_distance_list[j][0].data), float(noisy_heading_distance_list[j][1].data)]

    # Convert the Float32's in the trajectory lists to float to make the algorithm more readable
    for j in range(len(heading_distance_list)):
        heading_distance_list[j] = [float(heading_distance_list[j][0].data),
                                          float(heading_distance_list[j][1].data)]

    flat_scan_list = list()
    for k in range(len(scan_list)):
        new_scan_list = list()
        for scan in scan_list[k]:
            new_scan_list.append(float(scan.data))
        flat_scan_list.append(new_scan_list)

    # Loop over every control
    state = start_position

    # Get bounds of world
    min_x = corners[2][0]
    max_x = corners[0][0]
    min_y = corners[2][1]
    max_y = corners[0][1]

    #particles = create_uniform_particles([min_x, max_x], [min_y, max_y], [-1 * math.pi, math.pi])
    particles = create_initial_particles(start_position[0], start_position[1], [-1 * math.pi, math.pi])
    for i in range(len(noisy_heading_distance_list)):
        #break
        control = heading_distance_list[i]
        observation_scan = scan_list[i]

        # Run particle filter to get estimated pose
        particles = run_filter(particles, noisy_heading_distance_list[i], flat_scan_list[i], obstacles, 0.1, corners)

        estimate = get_max_prob_estimate(particles)
        estimated_pnt = (estimate[0], estimate[1])
        estimated_positions.append(estimated_pnt)

    #estimate_list_gt = test_motion_model(start_position, heading_distance_list)

    # Graph result against ground truth
    #print(estimated_positions[0])
    print estimated_positions
    graph(ground_truth_list, estimated_positions)
    #graph(ground_truth_list, estimate_list_gt)

