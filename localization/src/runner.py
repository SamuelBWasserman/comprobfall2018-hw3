#!/usr/bin/env python

import matplotlib.pyplot as plt
import trajectory_parser
import map_parser


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


if __name__ == '__main__':
    print "Running"
    start_position, heading_distance_list, ground_truth_list, noisy_heading_distance_list, scan_list = trajectory_parser.parse_trajectory("trajectories_1.txt")
    corners, obstacles, num_obstacles = map_parser.parse_map("map_1.txt")
    estimated_positions = list()
    # Run algorithm and get position estimates
    state = start_position
    for i in range(len(heading_distance_list)):
        control = heading_distance_list[i]
        observation_scan = scan_list[i]
        # Call particle filter to get an estimated current pose
        estimated_pose = (i, i)
        estimated_positions.append(estimated_pose)  # Append estimated pose (x, y)
        state = estimated_pose

    # Graph result against ground truth
    graph(ground_truth_list, estimated_positions)

