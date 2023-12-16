# This script contains all functions needed to solve the assignment. They are imported and run in the assignment2results.ipynb file

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# -- Parameters, data import
DIR_PATH = "C:/Users/tobia/Desktop/Skole/KU_inverse_problems/Assignments/Assignment 2/"
DIR_PATH_IMAGE = DIR_PATH + "images/"

# Import data
data = np.genfromtxt(DIR_PATH + "dataM.txt")

# Constants
mu0 = 1.256632e-6  # Unit: N / A**2. Vacuum permeability. 
h = 2e-2  # Unit: m. Height the magnetic field is measured at.
d_obs = data[:, 1] * 1e-9  # Unit: Tesla. Measured magnetic field.
std_d_obs = 25e-9  # Unit: Tesla. # Standard deviation of measured magnetic field data. 
x_band_obs = data[:, 0] * 1e-2  # Unit: m. Position of magnetic field measurements.
std_magnetization = 0.025  # Unit: A / m. Standard deviation of magnetization.


def norm2(x):
    """Euclidean norm of a 1darray x"""
    return np.sqrt(np.sum(x**2))


def find_nearest(x, val):
    """Find the value in the array x that is closest to the given value val.

    Args:
        x (1darray): Data values
        val (float): Value we want an index in x close to.

    Returns:
        int: Index of the value in x closest to val
    """
    idx = np.abs(x - val).argmin()
    return idx


def magnetization_distribution():
    """Draw a random number from a Gaussian around 0 with standard deviation std_magnetization=.0.025 A / m"""
    return np.random.normal(loc=0, scale=std_magnetization)  # Units: A / m


def find_stripe_widths(stripe_edges):
    """An array of stripe widths. Does not include their positions.

    Args:
        stripe_edges (1darray): Location of edges.
    """
    edges = np.nonzero(stripe_edges)[0]  # List of edge indices.
    width = edges[1:] - edges[:-1]  # Width is distance between edge indices
    return width


def find_width_positions(stripe_edges):
    """An array of stripe widths. ALl non-zero locations in the array are the width values corresponding to the edge at the same index.

    Args:
        stripe_edges (1darray): Location of edges.
    """
    edges = np.nonzero(stripe_edges)[0]  # List of edge indices.
    width = edges[1:] - edges[:-1]  # Width is distance between edge indices
    stripe_width = 1 * stripe_edges
    stripe_width[edges[:-1]] = width  # Width from one edge to its right neigbour. Exclude right end point as it has no right neighbour.
    return stripe_width


def change_magnetization_update_method(magnetization, stripe_edges):
    """Update the magnetization values in a random stripe.

    Args:
        magnetization (1darray): Magnetization array
        stripe_edges (1darray): Array of all the stripe edges
        stripe_width (1darray): Array of all the strip widths, going from left to right 
    """
    magnetization = magnetization.copy()
    stripe_edges = stripe_edges.copy()
    # Get random stripe to change magnetization in
    edges_coord_edge = np.nonzero(stripe_edges[:-1])[0]  # Get the edge indicis in a list
    random_edge_idx = np.random.choice(a=edges_coord_edge)  # Choose a random edge from the list
    stripe_width = find_width_positions(stripe_edges)  # Find the stripe widths
    width_of_chosen_edge = int(stripe_width[random_edge_idx])
    
    # Update magnetization of the chosen edge
    magnetization[random_edge_idx: random_edge_idx + width_of_chosen_edge] = magnetization_distribution()
    
    return magnetization, stripe_edges
    

def change_boundary_update_method(magnetization, stripe_edges):
    """Attempt to either add or remove a boundary. If unsuccesful, instead update magnetization.

    Args:
        magnetization (1darray): Magnetization array
        stripe_edges (1darray): Stripe edges position array
    """
    magnetization = magnetization.copy()
    stripe_edges = stripe_edges.copy()
    # Pick random location and update its boundary
    p_boundary = 0.125
    rng = np.random.uniform(low=0, high=1)
    random_band_idx = np.random.randint(low=1, high=stripe_edges.size-1)  # Do not include end points. Should always be edges.    
    add_edge = p_boundary > rng and stripe_edges[random_band_idx] == 0
    remove_edge = p_boundary < rng and stripe_edges[random_band_idx] == 1
    
    if add_edge:
        # Add edge and calculate distance to its to neighbouring edges such that can add new magnetizations to hose
        stripe_edges[random_band_idx] = 1  # Add edge
        stripe_edges_edge_coord = np.nonzero(stripe_edges)[0]  # All edges in no 0's and 1's coordinate system. Do not include dtype
        random_band_idx_edge_coord = np.where(stripe_edges_edge_coord==random_band_idx)[0]  # Newly added edge in no 0's and 1's coordinate system.
        width_left = random_band_idx_edge_coord - stripe_edges_edge_coord[random_band_idx_edge_coord - 1]
        width_right = stripe_edges_edge_coord[random_band_idx_edge_coord + 1] - random_band_idx_edge_coord
        width_left = width_left[0]
        width_right = width_right[0]
                
        # Update magnetization
        magnetization[random_band_idx - width_left : random_band_idx] = magnetization_distribution() # Left
        magnetization[random_band_idx : random_band_idx + width_right] = magnetization_distribution()  # Right
                
    elif remove_edge:
        # Find distance to neighbours of the point that will be removed.
        # The left neighbour will then have a magnetization value from itselt plus the width of the previous neighbours.
        stripe_edges_edge_coord = np.nonzero(stripe_edges)[0]
        random_band_idx_edge_coord = np.where(stripe_edges_edge_coord==random_band_idx)[0]
        width_left = random_band_idx_edge_coord - stripe_edges_edge_coord[random_band_idx_edge_coord - 1]
        width_right = stripe_edges_edge_coord[random_band_idx_edge_coord + 1] - random_band_idx_edge_coord        
        width_left = width_left[0]
        width_right = width_right[0]

        # Update edges
        stripe_edges[random_band_idx] = 0
        
        # Update magnetization
        magnetization[random_band_idx - width_left: random_band_idx + width_right] = magnetization_distribution()
        
    #else:  # In case neither of the above actions were taken, perform the magnetization update instead
     #   change_magnetization_update_method(magnetization, stripe_edges)
     
    return magnetization, stripe_edges
                

def G_matrix(N_bands):
    x_band = np.arange(-N_bands/2, N_bands/2)
    G = (-mu0 / (2 * np.pi) 
         * ((x_band_obs[:, None] - x_band[None, :]) ** 2 - h ** 2) 
         / ((x_band_obs[:, None] - x_band[None, :]) ** 2 + h ** 2) ** 2
    )
    return G


def prior_initialization(N_bands=200, illustrate_find_eps=False):
    """Use Tikhonov regularization to initialize MCMC prior updates

    Args:
        N_bands (int): Number of bands
    """
    # Calculate terms without epsilon for speeding up the loop
    G = G_matrix(N_bands)   
    d = 1 * d_obs
    GTG = G.T @ G
    GTd = G.T @ d
    I = np.eye(GTG.shape[0])
    # Loop values
    eps_vals = np.linspace(1e-6, 1e1, 100)
    loss_arr = np.zeros_like(eps_vals)
    # Find m and store loss
    for i, eps in enumerate(eps_vals):
        m = np.linalg.inv(GTG + eps ** 2 * I) @ GTd  # Tikhonov regularization expression
        misfit = norm2(d - G @ m) ** 2 
        loss = misfit - d.size * std_d_obs ** 2  # Should be zero for optimal epsilon
        loss_arr[i] = loss
    
    # Find optimal eps by loss intersection with 0
    loss_close_zero_idx = find_nearest(loss_arr, val=0)
    eps_optimal = eps_vals[loss_close_zero_idx]
    m_optimal = np.linalg.inv(GTG + eps_optimal ** 2 * I) @ GTd
    
    # Illustrate epsilon optial
    if illustrate_find_eps:
        fig, ax = plt.subplots(dpi=200)
        ax.plot(eps_vals, loss_arr, "-", label="Cost")
        ax.axhline(0, ls="dashed", lw="1", color="grey")
        ax.axvline(eps_optimal, ls="dashed", lw="1", color="grey", label=r"$\hat{\varepsilon}$")
        ax.set(xlabel=r"$\varepsilon$", ylabel=r"Cost $C(\varepsilon)$", title=r"Finding the optimal $\varepsilon$")
        ax.legend()
        fig.tight_layout()
        plt.savefig(DIR_PATH_IMAGE + f"optimal_eps{eps_optimal:.4f}.png")

    print("Optimal Tikhonov epsilon = ", eps_optimal)
    return m_optimal, eps_optimal


def prior_update_step(magnetization, stripe_edges):
    m = magnetization.copy()
    edges = stripe_edges.copy()
    # Choose update method with 50% chance
    change_magnetization = np.random.uniform(low=0, high=1) > 0.5
    if change_magnetization:
        m_new, edges_new = change_magnetization_update_method(m, edges)
    else:  # Change boundary
        m_new, edges_new = change_boundary_update_method(m, edges)
    return m_new, edges_new


def sample_prior_data(N_bands, N_step):
    # Iinitialization
    # magnetization, _ = prior_initialization(N_bands)
    # mag_nbor_diff = magnetization[1:] - magnetization[:-1]
    # stripe_edges = np.ones_like(magnetization)
    # stripe_edges[1:] = np.where(mag_nbor_diff==0, 0, 1)
    
    # Initialization bad
    magnetization = np.zeros(N_bands)
    stripe_edges = np.zeros_like(magnetization)
    stripe_edges[5] = 1
    stripe_edges[-5] = 1  # Create two edges

    # End points always an edge
    stripe_edges[0] = 1
    stripe_edges[-1] = 1
    
    # Update
    for _ in range(N_step):
        magnetization, stripe_edges = prior_update_step(magnetization, stripe_edges)
        
    return magnetization





def find_data_distribution(data):
    """The rho_D(data) distribution

    Args:
        data (1darray): Data values. Length equal to d_obs

    Returns:
        float: rho_D(data) given data
    """
    return np.exp(-0.5 * np.sum((data - d_obs) ** 2) / std_d_obs ** 2)


def find_likelihood(N_bands, model_parameters):
    G = G_matrix(N_bands)
    forward_solution = G @ model_parameters
    L = find_data_distribution(forward_solution)
    return L


def find_null_information(N_bands): # OBS NEED TO BE CHANGED
    return np.ones(N_bands)


def MCMC_posterior(N_bands, N_steps, burn_in_steps):
    # Initialize parameter values
    m_old, _ = prior_initialization(N_bands)
    mag_nbor_diff = m_old[1:] - m_old[:-1]
    stripe_edges_old = np.ones_like(m_old)
    stripe_edges_old[1:] = np.where(mag_nbor_diff==0, 0, 1)
    
    # End points always an edge
    stripe_edges_old[0] = 1
    stripe_edges_old[-1] = 1
    
    # Store values
    m_hist = np.empty((m_old.size, N_steps))
    m_hist[:, 0] = m_old
    
    # MCMC Iteration
    for i in tqdm(range(1, N_step)):
        m_old = m_hist[:, i-1]
        m_proposed, stripe_edges_proposed = prior_update_step(m_old, stripe_edges_old)
        accept_ratio = find_likelihood(N_bands, m_proposed) / find_likelihood(N_bands, m_old)
        
        if accept_ratio > np.random.uniform(low=0, high=1):
            m_hist[:, i] = m_proposed
            stripe_edges_old = stripe_edges_proposed
        else:
            m_hist[:, i] = m_old
    
    return m_hist[:, burn_in_steps:]


# -- Illustration functions --
def illustrate_prior_data(N_bands, N_step):
    x = np.linspace(-50, 50, N_bands)
    m = sample_prior_data(N_bands, N_step=N_step, layout="constrained")
    fig, ax = plt.subplots(dpi=165)
    ax.plot(x, m)
    ax.set(xlabel="x", ylabel="m", title=f"Magnetization prior distribution, N_step = {N_step}")
    plt.show()
    

def illustrate_one_model_parameter(N_bands, N_step, N_repeat):
    """Histogram of the first four model parameter"""
    # Get values
    model_arr = np.empty((N_repeat, 4))
    for i in tqdm(range(N_repeat)):
        m = sample_prior_data(N_bands, N_step)
        m04 = m[0:4]  # First four parameter values
        model_arr[i, :] = m04
    m0 = model_arr[:, 0]
    m1 = model_arr[:, 1]
    m2 = model_arr[:, 2]
    m3 = model_arr[:, 3]
    
    # Histogram values
    N_bins = int(np.sqrt(N_repeat))
    m0_count, m0_bins = np.histogram(m0, bins=N_bins)
    m1_count, m1_bins = np.histogram(m1, bins=N_bins)
    m2_count, m2_bins = np.histogram(m2, bins=N_bins)
    m3_count, m3_bins = np.histogram(m3, bins=N_bins)
    
    # Plot values
    fig, ax = plt.subplots(dpi=150, ncols=2, nrows=2, layout="constrained")
    # m0 ax00
    ax[0, 0].stairs(m0_count, m0_bins)
    ax[0, 0].set(title=r"m_0", xlabel="Magnetization values", ylabel="Frequency")
    # m1 ax01
    ax[0, 1].stairs(m1_count, m1_bins)
    ax[0, 1].set(title=r"m_1", xlabel="Magnetization values", ylabel="Frequency")
    # m2 ax10
    ax[1, 0].stairs(m2_count, m2_bins)
    ax[1, 0].set(title=r"m_2", xlabel="Magnetization values", ylabel="Frequency")
    # m3 ax11
    ax[1, 1].stairs(m3_count, m3_bins)
    ax[1, 1].set(title=r"m_3", xlabel="Magnetization values", ylabel="Frequency")
    # Figure title
    fig.suptitle("Histogram for the first four model parameters")
    plt.show()


def illustrate_burn_in(N_bands, N_steps):
    # Look at the five first model values and see when they converge i.e. the burn in time
    m = MCMC_posterior(N_bands, N_steps, burn_in_steps=0)
    m_first_five = m[:5, :]
    
    fig, ax = plt.subplots(dpi=150)
    ax.plot(np.arange(N_steps), m_first_five.T)
    ax.set(xlabel="Steps", ylabel="Magnetization", yscale="log", title="Burn in period for MCMC")
    ax.legend([r"$m_0$", r"$m_1$", r"$m_2$", r"$m_3$", r"$m_4$", ])
    plt.show()


if __name__ == "__main__":
    N_step = 1000_000
    N_bands = 200
    N_repeat = 100
    illustrate_burn_in(N_bands, N_step)
    print("Finished!")
    
    
    
# Old, probably wrong code
def find_prior_distribution(model_parameters, N_bands, N_step):  # OBS PROBABLY WRONG!
    """The rho_M(model_parameters) distribution

    Args:
        x (1darray): Array of random variable values
        N_bands (int): Number of bands in setup. 200
        N_step (int): Number of MCMC sampling steps.

    Returns:
        float: rho_M(model_parameters) given model_parameters
    """
    m_prior = sample_prior_data(N_bands, N_step)
    std = std_magnetization
    rho_M = np.exp(-0.5 * ((model_parameters - m_prior) / std) ** 2)
    return rho_M

def find_posterior(N_bands, model_parameters):
    rho_m = find_prior_distribution(model_parameters)
    likelihood = find_likelihood(N_bands, model_parameters)
    mu = find_null_information(N_bands)
    posterior = rho_m * likelihood / mu
    return posterior