# This script contains all functions needed to solve the assignment. They are imported and run in the assignment2results.ipynb file

import numpy as np
import matplotlib.pyplot as plt

# -- Parameters, data import
DIR_PATH = "C:/Users/tobia/Desktop/Skole/KU_inverse_problems/Assignments/Assignment 2/"
DIR_PATH_IMAGE = DIR_PATH + "images/"

# Import data
data = np.genfromtxt(DIR_PATH + "dataM.txt")

# Constants
mu0 = 1.256632e-6  # Vacuum permeability. Unit: N / A**2
h = 2e-2  # Unit: m
d_obs = data[:, 1] * 1e-9  # Unit: Tesla. Magnetic field
std_d_obs = 25e-9  # Unit: Tesla
x_band_obs = data[:, 0] * 1e-2  # Unit: m 


# -- Functions --


def norm2(x):
    return np.sqrt(np.sum(x**2))


def find_nearest(x, val):
    """Find the value in the array x that is closest to the given value val

    Args:
        x (1darray): Data values
        val (float): Value we want an index in x close to.

    Returns:
        int: Index of the value in x closest to val
    """
    idx = np.abs(x - val).argmin()
    return idx


def magnetization_distribution():
    return np.random.normal(loc=0, scale=std_d_obs)  # OBS UNITS??!?!?


def find_width_full_array(stripe_edges):
    """An array of stripe widths. ALl non-zero locations in the array are the width values

    Args:
        stripe_edges (1darray): Location of edges.
    """
    edges = np.nonzero(stripe_edges)[0]
    width = edges[1:] - edges[:-1]  # Width is distance between edge indices
    stripe_width = 1 * stripe_edges
    stripe_width[edges[:-1]] = width  # Width to the right, so right edge does not have a distance
    return stripe_width


def find_stripe_edges(magnetization):
    """Finds the stripe edges by calculating the difference between each neighbouring magnetization value. If it is non-zero there is an edge.

    Args:
        magnetization (1darray): Magnetization values.

    Returns:
        1darray: Stripe edges. Ones mean an edge, zeros mean not edge.
    """
    stripe_edges = np.ones_like(magnetization)
    nbor_difference = magnetization[1:] - magnetization[:-1]
    stripe_edges[1:] = np.where(nbor_difference==0, 0, 1)
    stripe_edges[-1] = 1  # End points always have an edge
    return stripe_edges


def change_magnetization_update_method(magnetization):
    """Update the magnetization values in a random stripe.

    Args:
        magnetization (1darray): Magnetization array
    """
    # Get random stripe to change magnetization in
    stripe_edges = find_stripe_edges(magnetization)
    edges_coord_edge = np.nonzero(stripe_edges[:-1])[0]  # Do not include right end point
    random_edge_idx = np.random.choice(a=edges_coord_edge)
    stripe_width = find_width_full_array(stripe_edges)
    width_of_chosen_edge = int(stripe_width[random_edge_idx])
    
    # Update magnetization
    magnetization[random_edge_idx: random_edge_idx + width_of_chosen_edge] = magnetization_distribution()
    

def change_boundary_update_method(magnetization):
    stripe_edges = find_stripe_edges(magnetization)
    # Pick random location and update its boundary
    p_boundary = 0.125
    rng = np.random.uniform(low=0, high=1)
    # Loop through edges until one is found where it can perform the selected action (add or remove)
    #add_edge = False
    #remove_edge = False
    loop_attempts = 0  # Max attemps
    #while (not add_edge) and (not remove_edge) and (loop_attempts < 100): 
    random_band_idx = np.random.randint(low=1, high=stripe_edges.size-1)  # Do not include end points. Should always be edges.    
    add_edge = p_boundary > rng and stripe_edges[random_band_idx] == 0
    remove_edge = p_boundary < rng and stripe_edges[random_band_idx] == 1
    loop_attempts += 1

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
                

def find_stripe_widths(stripe_edges):
    """An array of stripe widths

    Args:
        stripe_edges (1darray): Location of edges.
    """
    edges = np.nonzero(stripe_edges)[0]
    width = edges[1:] - edges[:-1]  # Width is distance between edge indices
    return width


def prior_initialization(N_bands=200, illustrate_find_eps=False):
    """Use Tikhonov regularization to initialize MCMC prior updates

    Args:
        N_bands (int): Number of bands
    """
    # Find G matrix
    x_band = np.arange(-N_bands/2, N_bands/2)
    N_obs = d_obs.size    
    G = (-mu0 / (2 * np.pi) 
         * ((x_band_obs[:, None] - x_band[None, :]) ** 2 - h ** 2) 
         / ((x_band_obs[:, None] - x_band[None, :]) ** 2 + h ** 2) ** 2
    )
           
    # Calculate terms without epsilon for speeding up the loop
    GTG = G.T @ G
    GTd = G.T @ d_obs
    I = np.eye(GTG.shape[0])
    
    # Loop values
    eps_vals = np.linspace(1e-5, 1e1, 250)
    loss_arr = np.zeros_like(eps_vals)
    # Find m and store loss
    for i, eps in enumerate(eps_vals):
        m = np.linalg.inv(GTG + eps ** 2 * I) @ GTd  # Tikhonov regularization expression
        misfit = norm2(d_obs - G @ m) ** 2 
        loss = misfit - N_obs * std_d_obs ** 2  # Should be zero for optimal epsilon
        loss_arr[i] = loss
    
    # Optimal epsilon and m
    # Find minimal eps by loss intersection with 0
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

    return m_optimal, eps_optimal


def sample_prior(N_bands, N_step):
    # Iinitialization
    #stripe_edges = np.ones(N_bands)  # Indices with an edge. Assumes width=1 initially, thus all stripes are 1 band.
    #magnetization, _ = prior_initialization(N_bands)
    
    # Random initialization
    magnetization = np.random.uniform(-1e-3, 1e-3, size=N_bands)
    
    # Update
    for _ in range(N_step):
        # Choose update method randomly
        change_magnetization = np.random.uniform(low=0, high=1) > 0.5
        if change_magnetization:
            change_magnetization_update_method(magnetization)
        else:  # Change boundary
            change_boundary_update_method(magnetization)
        
    return magnetization



if __name__ == "__main__":
    N_step=100_000
    m = sample_prior(N_bands=200, N_step=N_step)
    fig, ax = plt.subplots(dpi=165)
    ax.plot(np.arange(-50, 50, 0.5), m)
    ax.set(xlabel="x", ylabel="m", title=f"Magnetization prior distribution, N = {N_step}")
    plt.show()
    print("Finished!")