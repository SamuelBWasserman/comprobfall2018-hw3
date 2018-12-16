import numpy as np
import scipy
import scipy.stats
import math
import random
import copy

NUM_PARTICLES = 500  # number of particles to sample
OBS_SIZE = 2  # dimentionality of the obstacles(2D in this case)
STATE_SIZE = 3
SAMPLE_THRESHOLD = NUM_PARTICLES / 1.5


class Particle:
    """Class to hold particle info
    Args:
        N_LM: Coordinates of the landmarks
    """

    def __init__(self, x, y, theta):
        self.w = 1.0 / NUM_PARTICLES
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0


def create_uniform_particles(x_range, y_range, theta_range):
    """ Creates a uniform distribution of particles for initialization purposes
    Args:
        x_range: Range in the x-coordinate of the world. List of [min, max]
        y_range: Range in the y-coordinate of the world. List of [min, max]
        theta_range: Range in the theta values for the robot. List of [min, max]. Should be -pi to pi.
    Returns:
        particles: A list of particles
    """
    # create a list of initial random guesses for particle location
    particles_x = np.random.uniform(x_range[0], x_range[1], size=NUM_PARTICLES)
    particles_y = np.random.uniform(y_range[0], y_range[1], size=NUM_PARTICLES)
    particles_theta = np.random.uniform(theta_range[0], theta_range[1], size=NUM_PARTICLES)

    # create a list of particles
    particles = list()
    for n in range(NUM_PARTICLES):
        particles.append(Particle(particles_x[n], particles_y[n], particles_theta[n]))

    return particles

def create_initial_particles(x_coord, y_coord, theta_range):
    """ Initialize particle states at initial coordinates and randomly sample heading of robot
    Args:
        x_coord: Range in the x-coordinate of the world. List of [min, max]
        y_coord: Range in the y-coordinate of the world. List of [min, max]
        theta_range: Range in the theta values for the robot. List of [min, max]. Should be -pi to pi.
    Returns:
        particles: A list of particles
    """
    # create a list of initial random guesses for particle location
    particles_x = np.full((NUM_PARTICLES,), fill_value=x_coord)
    particles_y = np.full((NUM_PARTICLES,), fill_value=y_coord)
    particles_theta = np.random.uniform(theta_range[0], theta_range[1], size=NUM_PARTICLES)

    # create a list of particles
    particles = list()
    for n in range(NUM_PARTICLES):
        particles.append(Particle(particles_x[n], particles_y[n], particles_theta[n]))

    return particles


def run_filter(particles, u, z, obs, scan_noise):
    """ Runs the particle filtering algorithm
    Args:
        particles: An initial or prev list of particles of size NUM_PARTICLES
        u: The control at a time t
        z: The observation at time t
        obs: A list of obstacles which is a list of vertex coordinates of the obstacle
        scan_nosie: The scan noise parameter
    Retuns:
        particles: A list of resamples particles
    """
    # perform the prediction step
    particles = dynamics_prediction(particles, u)

    # perform observation update
    particles = observation_update(particles, z, obs, scan_noise)

    # perform resampling update
    particles = resample(particles)

    # TODO: Maybe add an estimate function that computes an estimate of the particle's pose


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


def observation_update(particles, z, obs, noise, def_nan_val=10.0):
    """ Performs the observation update given a list of observations
    Args:
        particles: A list of particles of size NUM_PARTICLES
        z: A list of observations from -30 to 30 degrees at increments of 1.125 degrees. Length is 54.
        obs: A list of obstacles which is represented as a list of coordinates of its vertices
        noise: The scan noise parameter
        def_nan_val: A default value to set the 'nan' values to
    """

    # sanity check
    if len(z) != 54:
        raise Exception("Error: Observation must be length 54!")

    # normalization factor
    eta = 0

    # deal with 'nan' values in the actual z
    for i in range(len(z)):
        if str(z[i]) == 'nan':
            z[i] = def_nan_val

    # create the noise covariance matrix
    Q = np.zeros((len(z), len(z)))
    np.fill_diagonal(Q, math.pow(noise, 2))

    # loop through all observation values
    for iz in range(NUM_PARTICLES):
        # compute each particle's estimated z_t
        est_obs = compute_obs(particles[iz], obs, noise)

        # remove 'nan'
        for i in range(len(est_obs)):
            if est_obs[i] == 'nan':
                est_obs[i] = def_nan_val

        est_obs = np.array(est_obs)

        # build a distribution where z_t is the mean and the std is the std is the scan noise parameter
        # then calulcate probability of the particle's z_t on the distribution
        weight = scipy.stats.multivariate_normal.pdf(est_obs, mean=z, cov=Q)


        eta += weight

        # set new particle weights
        particles[iz].w = weight

    # normalize all particle weights
    for n in range(NUM_PARTICLES):
        particles[n].w = particles[n].w / eta

    # sanity check for normalization
    w = 0
    for n in range(NUM_PARTICLES):
        w += particles[n].w

    if w != 1.0:
        raise Exception("Weight w does not sum to one! The sum of the weights is:", w)
        
    return particles


def compute_obs(particle, obs, scan_noise):
    """ Computes the observation for a particle.
    Args:
        particle: A `Particle` to get the observation for
        obs: A list of obstacles represented as a list of coordinates
        scan_noise: The scan noise parameter
    Returns:
        z_hat: An estimate of the particle's z_t
    """
    particle_scan_list = list()
    for i in range(54):
        particle_scan_list.append(0)
    obstacle_sides = list()
    for obstacle in obs:
        obstacle_iter = iter(obstacle)
        # Save first corner and skip to second corner upon entering the loop
        prev_corner = obstacle[0]
        next(obstacle_iter)
        # Loop through corners and build lines for each side of the polygonal obstacle
        for corner in obstacle_iter:
            obstacle_sides.append(find_line(prev_corner, corner))
            prev_corner = corner

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
                        particle_scan_list[i] = math.hypot(side_point[0] - scan_line[0][0], side_point[1] - scan_line[0][1]) + np.random.normal(0, scan_noise)
                        break
                if particle_scan_list[i] is not 'nan':
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
    current_point = [pnt1[0], pnt1[1]]
    x_slope = float((pnt2[0] - pnt1[0]) / float(10))
    y_slope = float((pnt2[1] - pnt1[1]) / float(10))
    line = list()
    # Loop from point to point adding values to the coordinate list
    while abs(pnt2[0] - current_point[0]) > 0.5 and abs(pnt2[1] - current_point[1]) > 0.5:
        new_pnt = [current_point[0], current_point[1]]
        line.append(new_pnt)
        current_point[0] += x_slope
        current_point[1] += y_slope
    # append last pnt
    line.append(current_point)
    return line


def resample(particles):
    """ Generates a new list of particles by resampling based on particles with high probability
    Args:
        particles: A list of particles
    Returns:
        new_particles or particles: A new list of particles if we resample otherwise the old list
    """
    # store all particle weights in a list
    weights = list()
    for n in range(NUM_PARTICLES):
        weights.append(particles[n].w)
    weights = np.array(weights)

    # calculate effective sample size
    neff = 0
    for w in weights:
        neff += math.pow(w, 2)
    neff = 1.0 / neff

    # resample only if below some sample threshold
    if neff < SAMPLE_THRESHOLD:
        # perform multinomial resampling
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1. # avoid roundoff error
        # indices correspond to the index of weights with high probability
        indices = np.searchsorted(cumsum, np.random.uniform(size=NUM_PARTICLES))

        # use indices to create new particle set
        new_particles = copy.deepcopy(particles)
        for i in indices:
            new_particles[i].x = particles[i].x
            new_particles[i].y = particles[i].y
            new_particles[i].theta = particles[i].theta
            # set the new weights of the particle
            new_particles[i].w = particles[i].w

            # new_particles[i].w = 1.0 / NUM_PARTICLES # TODO: Dunno if we have to change this

        return new_particles

    return particles

def get_estimate(particles):
    max_particle = particles[0]
    for particle in particles:
        if particle.w > max_particle.w:
            max_particle = particle
    return max_particle
