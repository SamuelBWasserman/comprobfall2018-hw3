import numpy as np
import math
import scipy.stats
from numpy.random import uniform
from numpy.random import randn
from numpy.random import random
from numpy.linalg import norm

NUM_PARTICLES = 100 # number of particles to sample
OBS_SIZE = 2 # dimentionality of the obstacles(2D in this case)
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
        self.obs = np.zeros((NUM_OBS, OBS_SIZE)) # (# of obs, 2)
        self.obsP = np.zeros((NUM_OBS*OBS_SIZE, OBS_SIZE)) # (# of obs*2, 2)

def run_filter(N, u, z, obs, state=None):
    """ Runs the particle filtering algorithm
    Args:
        particles: A list of particles of size NUM_PARTICLES
        u: The control at a time t
        z: The observation at time t
    Retuns:
        particles: A list of resamples particles
    """
    if state is not None:
        particles = create_gaussian_particles(mean=state, std=(5, 5, np.pi/4), N=N)
    else:
        particles = create_uniform_particles((-10, 10), (-10, 10), (-1*np.pi, np.pi), N)
    weights = np.ones(N) / N
    obs = np.array(obs)
    pose = np.array([0., 0.])
    # perform the prediction step
    dynamics_prediction(particles, u)

    # perform observation update
    observation_update(weights, z, R=0.01)

    # perform resampling update
    if neff(weights) < N/2:
        indexes = which_resample(weights)
        resample_from_index(particles, weights, indexes)
    # particles = resample(particles)
    mean, var = estimate(particles, weights)

    return mean

def create_uniform_particles(x_range, y_range, heading_range, n):
    particles = np.empty((n, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=n)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=n)
    particles[:, 2] = uniform(heading_range[0], heading_range[1], size=n)
    particles[:,2] %= 2 * np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N,3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


def dynamics_prediction(particles, u):
    """ Sample a new for each particle using motion model and odometry
    Args:
        particles: A list of particles of size NUM_PARTICLES
        u: A numpy vector of size 2 corresponding to (phi, dist)
    """
    N = len(particles)
    # Move heading
    particles[:, 2] += u[0] + (randn(N) * 0.1)
    particles[:, 2] %= 2 * np.pi

    # Move a distance
    dist = (u[1] * 1) + (randn(N) * 0.1)
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist


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
    xp[2,0] = xp[2,0] + u[0]

    # get new x coordinate
    xp[0,0] = xp[0,0] + u[1]*math.cos(xp[2,0])

    # get new y coordinate
    xp[1,0] = xp[1,0] + u[1]*math.sin(xp[2,0])

    return xp

def observation_update(weights, z, R):
    """ Performs the observation update given a list of observations
    Args:
        particles: A list of particles of size NUM_PARTICLES
        z: A list of observations from -30 to 30 degrees at increments of 1.125 degrees. Length is 54.
    """
    for i, obstacle_dist in enumerate(z):
        if obstacle_dist.data == "nan":
            distance = float(999)
            weights *= scipy.stats.norm(distance, R).pdf(distance)
        else:
            distance = float(obstacle_dist.data)
            weights *= scipy.stats.norm(distance, R).pdf(float(z[i].data))


    weights += 1.e-300
    weights /= sum(weights)

def which_resample(weights):
    N = len(weights)

    positions = (np.arange(N) + random()) / N

    indexes = np.zeros(N, 'i')
    cum_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cum_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def neff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))

def estimate(particles, weights):
    """Return the mean and variance of weighted particles"""
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos-mean)**2, weights=weights, axis=0)
    return mean, var
