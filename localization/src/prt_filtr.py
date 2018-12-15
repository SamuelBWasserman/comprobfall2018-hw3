import numpy as np
import math

NUM_PARTICLES = 100  # number of particles to sample
OBS_SIZE = 2  # dimentionality of the obstacles(2D in this case)
STATE_SIZE = 3


class Particle:
    """Class to hold particle info
    Args:
        N_LM: Coordinates of the landmarks
    """

    def __init__(self, x, y, theta, NUM_OBS):
        self.w = 1.0 / NUM_PARTICLES
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0


def create_uniform_particles(x_range, y_range, theta_range, NUM_OBS):
    """ Creates a uniform distribution of particles for initialization purposes
    Args:
        x_range: Range in the x-coordinate of the world. List of [min, max]
        y_range: Range in the y-coordinate of the world. List of [min, max]
        theta_range: Range in the theta values for the robot. List of [min, max]. Should be -pi to pi.
    """
    # create a list of initial random guesses for particle location
    particles_x = np.random.uniform(x_range[0], x_range[1], size=NUM_PARTICLES)
    particles_y = np.random.uniform(y_range[0], y_range[1], size=NUM_PARTICLES)
    particles_theta = np.random.uniform(theta_range[0], theta_range[1], size=NUM_PARTICLES)

    # create a list of particles
    particles = list()
    for n in range(NUM_PARTICLES):
        particles.append(Particle(particles_x, particles_y, particles_theta, NUM_OBS))

    return particles


def run_filter(particles, u, z, obs):
    """ Runs the particle filtering algorithm
    Args:
        particles: An initial or prev list of particles of size NUM_PARTICLES
        u: The control at a time t
        z: The observation at time t
    Retuns:
        particles: A list of resamples particles
    """
    # perform the prediction step
    particles = dynamics_prediction(particles, u)

    # perform observation update
    particles = observation_update(particles, z, obs)

    # perform resampling update
    particles = resample(particles)

    return particles


def dynamics_prediction(particles, u):
    """ Sample a new for each particle using motion model and odometry
    Args:
        particles: A list of particles of size NUM_PARTICLES
        u: A numpy vector of size 2 corresponding to (phi, dist)
    """

    for n in range(NUM_PARTICLES):
        # create a numpy vector to represent one particle
        x_prev = np.zeros((STATE_SIZE, 1))
        x_prev[0, 0] = particles[n].x
        x_prev[1, 0] = particles[n].y
        x_prev[2, 0] = particles[n].theta

        # run the prediction on one particle
        x_pred = motion_model(x_prev, u)

        # store these values back into the list
        particles[n].x = x_pred[0, 0]
        particles[n].y = x_pred[1, 0]
        particles[n].theta = x_pred[2, 0]

    return particles


def motion_model(xp, u):
    """ Computes a predicted state given previous state and current control
    Args:
        xp: Previous state as a numpy array of size (3,1).
            Where xp_1 and xp_2 are x and y coordinates and xp_3 is theta.
        u: Current control as a numpy vector of [phi, d]
    Returns:
        x: Current state. Same size and form as xp.
    """

    # get new theta value because we rotate first
    xp[2, 0] = xp[2, 0] + u[0]

    # get new x coordinate
    xp[0, 0] = xp[0, 0] + u[1] * math.cos(xp[2, 0])

    # get new y coordinate
    xp[1, 0] = xp[1, 0] + u[1] * math.sin(xp[2, 0])

    return xp


def observation_update(particles, z, obs):
    """ Performs the observation update given a list of observations
    Args:
        particles: A list of particles of size NUM_PARTICLES
        z: A list of observations from -30 to 30 degrees at increments of 1.125 degrees. Length is 54.
    """

    # sanity check
    if len(z) != 54:
        raise Exception("Error: Observation must be length 54!")

    # loop through all observation values
    pz = list()
    for iz in range(NUM_PARTICLES):
        print("test")

        # compute each particle's estimated z_t
        pz.append(compute_obs(particles[iz], obs))

        # take the distance between the two vectors

        # build a distribution where z_t is the mean and the std is the std is the scan noise parameter

        # find the probability of each particle's estimate z_t from the distribution build above

    return particles


def compute_obs(particle, obs):
    """ Computes the observation for a particle.
    Args:
        particle: A `Particle` to get the observation for
    Returns:
        z_hat: An estimate of the particle's z_t
    """
    particle_scan_list = list()
    obstacle_sides = list()
    for obstacle in obs:
        obstacle_iter = iter(obstacle)
        # Save first corner and skip to second corner upon entering the loop
        prev_corner = obstacle[0]
        next(obstacle_iter)
        # Loop through corners and build lines for each side of the polygonal obstacle
        for corner in obstacle_iter:
            obstacle_sides.append(find_line(prev_corner, corner))

    # Loop through the scan range for a given particle at it's pose
    position = particle.theta - (math.radians(30))
    for i in range(54):
        # init spot in list
        particle_scan_list[i] = 0

        # Find line between the two points in the scan 10 distance units away
        r = math.sqrt(1 + position**2)
        scan_pnt1 = [particle.x, particle.y]
        scan_pnt2 = [particle.x + 10/r, particle.y + (10 * position) / r]
        scan_line = find_line(scan_pnt1, scan_pnt2)

        # loop through each side in every obstacle
        for side in obstacle_sides:
            # Loop through each pair of points (scan line point -> obstacle side point)
            for point in scan_line:
                for side_point in side:
                    # If the point on the scan is equal to the point on the side with some error,
                    # add distance from original point in scan to point on side to distance list
                    if abs(point[0] - side_point[0]) < 0.25 and abs(point[1] - side_point[1]) < 0.25:
                        particle_scan_list[i] = math.hypot(side_point[0] - scan_line[0][0], side_point[1] - scan_line[0][1])
                    break
        if particle_scan_list[i] == 0:
            particle_scan_list[i] = 'nan'
        position = position + math.radians(1.125)

    return particle_scan_list


def find_line(pnt1, pnt2):
    """
    :param pnt1:
    :param pnt2:
    :return: list of points between two points
    """
    x_min = min(pnt1[0], pnt2[0])
    if x_min == pnt1[0]:
        current_point = [pnt1[0], pnt1[1]]
    else:
        current_point = [pnt2[0], pnt2[1]]
    x_max = max(pnt1[0], pnt2[0])
    y_min = min(pnt1[1], pnt2[1])
    y_max = max(pnt1[1], pnt2[1])

    x_slope = (x_max - x_min) / 20
    y_slope = (y_max - y_min) / 20
    line = list()
    # Loop from point to point adding values to the coordinate list
    while current_point[0] < x_max:
        line.append(current_point)
        current_point[0] += x_slope
        current_point[1] += y_slope
    return line


def resample(particles):
    return particles