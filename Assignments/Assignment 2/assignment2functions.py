# This script contains all functions needed to solve the assignment. They are imported and run in the assignment2results.ipynb file

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import circulant


# -- Parameters, data import
DIR_PATH = "C:/Users/tobia/Desktop/Skole/KU_inverse_problems/Assignments/Assignment 2/"
DIR_PATH_IMAGE = DIR_PATH + "images/"

# Import data
data = np.genfromtxt(DIR_PATH + "dataM.txt")

# Constants
N_bands = 200
mu0 = 1.25663706212e-6  # Unit: N / A**2. Vacuum permeability. 
h = 2e-2  # Unit: m. Height the magnetic field is measured at.
d_obs = data[:, 1] * 1e-9  # Unit: Tesla. Measured magnetic field.
std_d_obs = 25 * 1e-9  # Unit: Tesla. # Standard deviation of measured magnetic field data. 
x_band_obs = data[:, 0] * 1e-2  # Unit: m. Position of magnetic field measurements.
x_band = np.linspace(-50, 50, N_bands) * 1e-2  # Unit: m.
dx_band = 0.5e-2  # Unit: m
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
    """Draw a random number from a Gaussian around 0 with standard deviation std_magnetization=0.025 A / m"""
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
    Returns:
        (1darray, 1darray): Magnetization and stripe edges arrays after the update has been performed.
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
    """Attempt to either add or remove a boundary.

    Args:
        magnetization (1darray): Magnetization array
        stripe_edges (1darray): Stripe edges position array
        
    Returns:
        (1darray, 1darray): Magnetization and stripe edges arrays after the update has been performed.
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
                

def G_matrix():
    G = dx_band * (- mu0 / (2 * np.pi) * 
                   ((x_band_obs[:, None] - x_band[None, :]) ** 2 - h ** 2) 
                   / ((x_band_obs[:, None] - x_band[None, :]) ** 2 + h ** 2) ** 2 )
    return G


def prior_initialization(illustrate_find_eps=False):
    """Use Tikhonov regularization to initialize MCMC prior updates

    Args:
        N_bands (int): Number of bands
    returns:
        (1darray, float): magnetization, optimal epsilon
    """
    # Calculate terms without epsilon for speeding up the loop
    G = G_matrix()   
    d = 1 * d_obs
    GTG = G.T @ G
    GTd = G.T @ d
    I = np.eye(GTG.shape[0])
    # Loop values
    eps_vals = np.linspace(1e-7, 1e-5, 100)
    loss_arr = np.zeros_like(eps_vals)
    # Find m and store loss
    for i, eps in tqdm(enumerate(eps_vals)):
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
    """Sample from the prior by updating the bands using one of three methods. 

    Args:
        magnetization (1darray): Array with the magnetization values at the N_bands x positions
        stripe_edges (1darray): Location of the stripe edges

    Returns:
        (1darray, 1darray): Magnetization and stripe edges arrays after the update has been performed.
    """
    m = magnetization.copy()
    edges = stripe_edges.copy()
    # Choose update each method with 50% chance
    change_magnetization = np.random.uniform(low=0, high=1) > 0.5
    if change_magnetization:
        m_new, edges_new = change_magnetization_update_method(m, edges)
    else:  # Change boundary
        m_new, edges_new = change_boundary_update_method(m, edges)
    return m_new, edges_new


def sample_prior_data(N_bands, N_step, burn_in_steps, skip_factor=1, save_for_animate=False):
    """Returns magnetization """
    # Iinitialization
    magnetization, _ = prior_initialization()
    mag_nbor_diff = magnetization[1:] - magnetization[:-1]
    stripe_edges = np.ones_like(magnetization)
    stripe_edges[1:] = np.where(mag_nbor_diff==0, 0, 1)
    
    # End points always an edge
    stripe_edges[0] = 1
    stripe_edges[-1] = 1
    
    m_hist = np.empty((N_bands, N_step))
    m_hist[:, 0] = magnetization
    m_save = []
    # Update
    for i in tqdm(range(1, N_step)):
        m = m_hist[:, i-1]
        m_new, stripe_edges = prior_update_step(m, stripe_edges)
        m_hist[:, i] = m_new
        
        if (i > burn_in_steps) and (i % skip_factor) == 0:
            m_save.append(m_new)
    if save_for_animate:
        return m_save
    else:
        return m_hist[:, burn_in_steps:]


def data_distribution(data):
    """The rho_D(data) distribution

    Args:
        data (1darray): Data values. Length equal to d_obs

    Returns:
        float: rho_D(data) given data
    """
    rho_d = np.exp(-0.5 * np.sum((data - d_obs) ** 2) / std_d_obs ** 2)    
    return rho_d


def likelihood(model_parameters):
    G = G_matrix()
    forward_solution = G @ model_parameters  # OBS right now forward_solution is way to small in magnitude
    L = data_distribution(forward_solution)    
    return L


def null_information(N_bands):
    # mu_d(m) = k sqrt(det(g_d(d)))  - ignore k, cancels out in acceptance ratio
    metric_matrix = np.eye(N_bands)
    mu = np.sqrt(np.linalg.det(metric_matrix))
    return mu


def acceptance_ratio(model_parameters_proposed, model_parameters_old):
    L_new = likelihood(model_parameters_proposed)
    L_old = likelihood(model_parameters_old)
    misfit_new = -np.log(L_new)
    misfit_old = -np.log(L_old)
    acceptance = np.exp(misfit_old - misfit_new)
    return np.min((1, acceptance)), misfit_new


def MCMC_posterior(N_bands, N_step, burn_in_steps):
    """Sample from posterior using MCMC

    Args:
        N_bands (_type_): _description_
        N_step (_type_): _description_
        burn_in_steps (_type_): _description_

    Returns:
        (1darray, 1darray): magnetization history, missfit history
    """
    # Initialize magnetization
    m_old, _ = prior_initialization()
    mag_nbor_diff = m_old[1:] - m_old[:-1]
    # Initialize edges
    stripe_edges_old = np.ones_like(m_old)
    stripe_edges_old[1:] = np.where(mag_nbor_diff==0, 0, 1)
    stripe_edges_old[0] = 1  # End points always an edge
    stripe_edges_old[-1] = 1
    
    # Store values
    m_hist = np.empty((N_bands, N_step))
    m_hist[:, 0] = m_old
    missfit_list = []
    # MCMC Iteration
    for i in tqdm(range(1, N_step)):
        m_old = m_hist[:, i-1]
        m_proposed, stripe_edges_proposed = prior_update_step(m_old, stripe_edges_old)
        accept_ratio, misfit_proposed = acceptance_ratio(m_proposed, m_old)
        
        if accept_ratio > np.random.uniform(low=0, high=1):
            m_hist[:, i] = m_proposed
            stripe_edges_old = stripe_edges_proposed
            missfit_list.append(misfit_proposed)
        else:
            m_hist[:, i] = m_old
    
    return m_hist[:, burn_in_steps:], missfit_list


# -- Illustration functions --
def illustrate_four_prior_parameters(N_bands, N_step, burn_in_steps):  # PRIOR
    """Histogram of the prior four model parameter. Should be Gaussianly distributed"""    
    # Get m values
    m = sample_prior_data(N_bands, N_step, burn_in_steps)
    m0 = m[0, :]
    m5 = m[5, :]
    m20 = m[20, :]
    m40 = m[40, :]
    
    # Histogram values
    N_bins = int(np.sqrt(m0.size))
    m0_count, m0_bins = np.histogram(m0, bins=N_bins)
    m5_count, m5_bins = np.histogram(m5, bins=N_bins)
    m20_count, m20_bins = np.histogram(m20, bins=N_bins)
    m40_count, m40_bins = np.histogram(m40, bins=N_bins)
    bin_width_arr = m0_bins[1:] - m0_bins[:-1]
    frequency = bin_width_arr[0] / N_bins
    freq_str_scientific_notation = np.format_float_scientific(frequency, precision=2, trim="-")
    
    # Plot values
    fig, ax = plt.subplots(dpi=150, ncols=2, nrows=2, layout="constrained")
    # m0 ax00
    ax[0, 0].stairs(m0_count, m0_bins)
    #ax[0, 0].plot(x_vals, y_pdf, "-")
    ax[0, 0].set(ylabel=f"Frequency ({freq_str_scientific_notation})", title=r"$m_0$")
    # m1 ax01
    ax[0, 1].stairs(m5_count, m5_bins)
    ax[0, 1].set(title=r"$m_5$")
    # m2 ax10
    ax[1, 0].stairs(m20_count, m20_bins)
    ax[1, 0].set(xlabel="Magnetization values", title=r"$m_{20}$")
    # m3 ax11
    ax[1, 1].stairs(m40_count, m40_bins)
    ax[1, 1].set(xlabel="Magnetization values", ylabel=f"Frequency ({freq_str_scientific_notation})", title=r"$m_{40}$")
    # Figure title
    suptitle = r"Four prior sample parameters, $N_{step}$ = " + str(N_step)
    fig.suptitle(suptitle)
    plt.show()


def illustrate_burn_in(misfit):
    """Plot misfit over iterations to find the required steps to avoid burn in

    Args:
        misfit (1d array like): Misfit values from MCMC posterior sampling.
    """
    fig, ax = plt.subplots(dpi=150)
    ax.plot(misfit)
    ax.set(xlabel="Steps", ylabel="Misfit of accepted pertubations", title="Burn in period for MCMC")
    plt.show()


def illustrate_auto_correlation(misfit_vals, k_max):
    """Calculate and plot autocorrelation using misfit values of accepted MCMC steps.

    Args:
        misfit_vals (1d arraylike): Misfit values from MCMC sampling steps that were accepted
        k_max (int): The max correlation length investigated
    """
    # Calculate autocorrelation
    # Shift misfit such that it oscillates around 0 by subtracting mean of data
    misfit = np.array(misfit_vals)
    misfit = misfit[int(misfit.size()/2):]  # Discard burnin data
    misfit_mean = np.mean(misfit)
    misfit_shifted = misfit - misfit_mean
    misfit_roll_matrix = circulant(misfit_shifted)  # Creates square matrix where each row is a cummulatively +1 rolled vector of the input vector. 
    misfit_roll_matrix = misfit_roll_matrix[:, :k_max]  # No need to have correlation length close to the total number of misfit values
    autocorrelation = misfit_shifted @ misfit_roll_matrix
    
    fig, ax = plt.subplots(dpi=200)
    ax.plot(autocorrelation)
    ax.set(xlabel="Correlation length", ylabel="Autocorrelation (unnormalized)", title="Finding correlation of MCMC random walk")
    plt.show()


def illustrate_G_matrix():
    """Illustrate the G matrix using imshow"""
    fig, ax = plt.subplots()
    G = G_matrix()
    ax.imshow(G, cmap="bone")
    plt.show()


def illustrate_single_sample_posterior(mcmc_magnetization_output):
    """Illustrate a single sample of the MCMC posterior.

    Args:
        mcmc_magnetization_output (1darray): Magnetization values from MCMC sampling
    """
    # Get data
    m = mcmc_magnetization_output[:, -1]  # Only sample a single one, choose the last.
    G = G_matrix()
    Gm = G @ m
    
    # Plot
    fig, ax = plt.subplots()
    ax.plot(x_band_obs, Gm, label="posterior")
    ax.plot(x_band_obs, d_obs, label="d_obs")
    ax.legend()
    plt.show()


def illustrate_posterior_mean_and_std(mcmc_magnetization_output, skip_elements):  # Exercise 5
    """Show the Gaussian distribution of each position's magnetization value. Each point is the mean and the errorbar shows the size of one standard deviation.

    Args:
        mcmc_magnetization_output (1darray): Magnetization values from MCMC sampling
        skip_elements (int): 
    """
    # Gaussianly distributed 
    m = mcmc_magnetization_output[:, ::skip_elements]
    m_mean = np.mean(m, axis=1)
    m_std = np.std(m, axis=1)
    
    fig, ax = plt.subplots(dpi=200, ncols=1, nrows=2, layout="constrained")
    
    # ax0 mean w/errorbar
    ax[0].plot(x_band, m_mean, "k")
    ax[0].errorbar(x_band, m_mean, yerr=m_std, fmt="k,", alpha=0.4)
    ax[0].set(ylabel="Mean Magnetization", title="Magnetization parameter distribution")
    
    # ax1 std
    ax[1].plot(x_band, m_std, "k")
    ax[1].set(xlabel="x", ylabel="Std", title="Standard deviation distribution")
    plt.show()
    

if __name__ == "__main__":
    N_step = 1000_000
    N_bands = 200
    # -- PRIOR --
    #illustrate_d_gm()
    #illustrate_G_matrix()
    #illustrate_four_prior_parameters(N_bands, N_step, burn_in_steps=2000)
    
    # -- Posterior --
    #illustrate_burn_in(N_bands, N_step)
    #illustrate_single_sample_posterior(N_bands, N_step, burn_in_steps=1000)
    # illustrate_posterior_mean_and_std(N_bands, N_step, burn_in_steps=10_000)
    print("Finished!")
    