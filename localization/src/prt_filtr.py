import numpy as np
import scipy
import scipy.stats
import math
import random
import copy

NUM_PARTICLES = 100  # number of particles to sample
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
        particle = Particle(particles_x[n], particles_y[n], particles_theta[n])
        particle.x = particles_x[n]
        particle.y = particles_y[n]
        particle.theta = particles_theta[n]
        particles.append(particle)
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
    particles_x = np.full((NUM_PARTICLES,), fill_value=x_coord + np.random.normal(0, 0.001), dtype=np.float64)
    particles_y = np.full((NUM_PARTICLES,), fill_value=y_coord + np.random.normal(0, 0.001), dtype=np.float64)
    particles_theta = np.full((NUM_PARTICLES,), fill_value=0.0 , dtype=np.float64)


    # create a list of particles
    particles = list()
    for n in range(NUM_PARTICLES):
        particle = Particle(particles_x[n], particles_y[n], particles_theta[n])
        particle.x = particles_x[n]
        particle.y = particles_y[n]
        particle.theta = particles_theta[n]
        particles.append(particle)

    return particles


def run_filter(particles, u, z, obs, scan_noise, world_corners):
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
    particles = observation_update(particles, z, obs, scan_noise, world_corners)

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
    xp[2, 0] = u[0]

    # get new x coordinate
    xp[0, 0] = xp[0, 0] + u[1] * math.cos(xp[2, 0])

    # get new y coordinate
    xp[1, 0] = xp[1, 0] + u[1] * math.sin(xp[2, 0])

    return xp


def observation_update(particles, z, obs, noise, corners, def_nan_val=0.00):
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
            z[i] = np.random.normal(def_nan_val, noise)

    # loop through all observation values
    for iz in range(NUM_PARTICLES):
        # compute each particle's estimated z_t
        est_obs = compute_new_obs(particles[iz], obs, noise, corners)
        est_obs = np.array(est_obs)

        for i in range(len(est_obs)):
            if str(est_obs[i]) == 'nan':
                est_obs[i] = np.random.normal(def_nan_val, noise)

        # build a distribution where z_t is the mean and the std is the std is the scan noise parameter
        # then calculate probability of the particle's z_t on the distribution
        weight = scipy.stats.norm.pdf(est_obs, loc=est_obs, scale=54*[noise])
        weight = np.linalg.norm(weight)
        weight = 1.0

        eta += weight

        # set new particle weights
        particles[iz].w = weight

    # normalize all particle weights
    for n in range(NUM_PARTICLES):
        particles[n].w = particles[n].w / eta
        
    return particles

def compute_new_obs(particle, obs, scan_noise, corners):
    """ Computes the observation for a particle. A remake of the above
    Args:
        particle: A `Particle` to get the observation for
        obs: A list of obstacles represented as a list of coordinates
        scan_noise: The scan noise parameter
    Returns:
        z_hat: An estimate of the particle's z_t
    """
    # initialize the particle list to all zeros
    particle_scan_list = np.zeros(shape=(54,), dtype=np.float64)

    # create obstacle edge list which is a list of edges
    obs_edge_list = list() # a list of the edges which is a list of 2 coordinates for every obs
    for o in obs:
        for v in range(len(o)):
            if v == len(o)-1:
                break
            obs_edge_list.append([list(o[v]), list(o[v+1])])

    # add world edges too
    for i in range(len(corners)):
        if i == len(corners)-1:
            break
        obs_edge_list.append([list(corners[i]), list(corners[i+1])])


    # set current angle and get distance estimates
    curr_pos = particle.theta - math.radians(30)
    scan_start = [particle.x, particle.y]
    for i in range(54):
        # Get the endpoints of the scan line
        scan_endpt = [particle.x + 10.0*math.cos(curr_pos), particle.y + 10.0*math.sin(curr_pos)]

        # set min scan distance
        min_scan_dist = 10.0
        for e in obs_edge_list:
            # check to see if scan line intersects this edge
            inter_coord = does_intersect([scan_start, scan_endpt], e, corners)


            if type(inter_coord) != bool:
                dist = distance(scan_start, inter_coord) #+ np.random.normal(0, scan_noise)

                # set new min_scan_dist
                if dist < min_scan_dist and dist > 0.45:
                    min_scan_dist = dist


        if min_scan_dist == 10.0:
            particle_scan_list[i] = 'nan'
        else:
            particle_scan_list[i] = min_scan_dist

        # get the next angle
        curr_pos += math.radians(1.125)

    return particle_scan_list

def does_intersect(line1, line2, bounds):
    """ Computes the coordinates of intersection if the two lines interect or False otherwise.
    Args:
        line1: A list of list which corresponds to points on the line
        line2: A list of list which corresponds to points on the line
        bounds: A list of list which corresponds to
    Returns:
        inter: Coordinate of intersection or False
    """
    # convert to numpy arrays
    line1 = np.array(line1, dtype=np.float64)
    line2 = np.array(line2, dtype=np.float64)

    try:
        t, s = np.linalg.solve(np.array([line1[1] - line1[0], line2[0] - line2[1]]).T, line2[0] - line1[0])
        inter = (1 - t) * line1[0] + t * line1[1]

        # check to see if value is in the bounds of the world
        # Get bounds of world
        min_x = bounds[2][0]
        max_x = bounds[0][0]
        min_y = bounds[2][1]
        max_y = bounds[0][1]
        # discard inter points outside the world
        if inter[0] < min_x or inter[0] > max_x:
            inter = False
            return inter
        if inter[1] < min_y or inter[1] > max_y:
            inter = False
            return inter

    except np.linalg.linalg.LinAlgError:
        inter = False

    return inter


def distance(pt1, pt2):
    return np.linalg.norm(pt1-pt2)


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
        print("RESAMPLING. NEFF=", neff)
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

            new_particles[i].w = 1.0 / NUM_PARTICLES

        return new_particles

    print("NO RESAMPLING. NEFF=", neff)
    return particles

def get_estimate(particles):
    # get list of all particle pose
    particle_x = list()
    particle_y = list()
    particle_w = list()
    for n in range(NUM_PARTICLES):
        particle_x.append(particles[n].x)
        particle_y.append(particles[n].y)
        particle_w.append(particles[n].w)

    # get mean and std of list
    x_mean = np.mean(particle_x)
    x_var = np.std(particle_x)
    y_mean = np.mean(particle_y)
    y_var = np.std(particle_y)

    # create mean vector and covariance matrix
    pos_mean = np.array([x_mean, y_mean])
    pos_cov = np.zeros(shape=(2,2), dtype=np.float64)
    pos_cov[0,0] = x_var
    pos_cov[1,1] = y_var

    # compute estimate
    prediction = np.random.normal(loc=pos_mean, scale=[x_var, y_var])

    return prediction

def get_max_prob_estimate(particles):
    """ Gets the state with highest probability
    Args:
        particles: A list of particles
    Returns:
        prediction: A prediction list.
    """
    weights = list()
    robot_states = list()
    for n in range(NUM_PARTICLES):
        weights.append(particles[n].w)
        robot_states.append([particles[n].x, particles[n].y])

    idx = weights.index(max(weights))

    sidx = np.random.choice(range(len(robot_states)), p=weights)

    #for n in range(NUM_PARTICLES):
        #print("WEIGHTS", particles[n].w, particles[n].x, particles[n].y)

    return [particles[sidx].x, particles[sidx].y]
