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

    def __init__(self, NUM_OBS):
        self.w = 1.0 / NUM_PARTICLES
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        # (x,y) positions of the obstacles.
        self.obs = np.zeros((NUM_OBS, OBS_SIZE))  # (# of obs, 2)
        self.obsP = np.zeros((NUM_OBS * OBS_SIZE, OBS_SIZE))  # (# of obs*2, 2)


def run_filter(particles, u, z):
    """ Runs the particle filtering algorithm
    Args:
        particles: A list of particles of size NUM_PARTICLES
        u: The control at a time t
        z: The observation at time t
    Retuns:
        particles: A list of resamples particles
    """

    # perform the prediction step
    particles = dynamics_prediction(particles, u)

    # perform observation update
    particles = observation_update(particles, z)

    # perform resampling update
    particles = resample(particles)

    return particles


def dynamics_prediction(particles, u):
    """ Sample a new for each particle using motion model and odometry
    Args:
        particles: A list of particles of size NUM_PARTICLES
        u: A numpy vector of size 2 corresponding to (phi, dist)
    """

    for i in range(NUM_PARTICLES):
        x_prev = np.zeros((STATE_SIZE, 1))
        x_prev[0, 0] = particles[i].x
        x_prev[1, 0] = particles[i].y
        x_prev[2, 0] = particles[i].theta
        x_pred = motion_model(x_prev, u)
        particles[i].x = x_pred[0, 0]
        particles[i].y = x_pred[1, 0]
        particles[i].theta = x_pred[2, 0]

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


def observation_update(particles, z):
    """ Performs the observation update given a list of observations
    Args:
        particles: A list of particles of size NUM_PARTICLES
        z: A list of observations from -30 to 30 degrees at increments of 1.125 degrees. Length is 54.
    """

    # sanity check
    if len(z) != 54:
        raise Exception("Error: Observation must be length 54!")

    # loop through all observation values
    for iz in range(len(z)):
        print("test")
    return particles


def resample(particles):
    return particles