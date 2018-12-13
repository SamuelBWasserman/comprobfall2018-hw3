import rospy
import matplotlib.pyplot as plt

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
