# Code strucutre:
    # 1. Imports
    # 2. Parameters and data
    # 3. Functions
    # 4. Running the code


# -- 1. Imports -- 
import numpy as np
import matplotlib.pyplot as plt

# -- Parameters and data --
np.random.seed(100)  # To compare results with other students
SAVE_DIR = "Assignments/Assignment 1/images/"  # NOTE unique to my system

# Detector setup
Nx = 14  # x values go from 0 to 13 i.e. 14 possible values
Ny = 12
N_detector = 12

# Time data - the t's are the length of the hypotenuse of the ray inside the new medium
t1 = np.sqrt(2)  
t2 = np.sqrt(8)
t3 = np.sqrt(18)
speed_diff_reciprocal = 5  # s = 1 / 0.2, where 0.2 = 5.2 m/s - 5 m/s

# Box and delta
time_data = np.array([
    0, 0, 0, 0, 0, t1, t2, t3, t3, t3, t3, t3,  # Rays from the left, detector left->right
    t3, t3, t3, t2, t1, 0, 0, 0, 0, 0, 0, 0,  # Rays from the right, detector left->right
]) * speed_diff_reciprocal

# Single square box
time_data_delta = np.array([  
    0, 0, 0, 0, 0, t1, 0, 0, 0, 0, 0, 0,
    0, 0, t1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]) * speed_diff_reciprocal


# -- 3. Functions --
def norm2(x):
    """Euclidian norm of array x, |x|_2"""
    return np.sqrt(np.sum(x**2))


def gaussian_noise(x, mu=0):
    """Calculate Gaussian noise with the standard deviation given in the exercise"""
    std = 1 / 18 * norm2(x)
    return np.random.normal(loc=mu, scale=std, size=x.size)


def time_observed(time):
    """Add Gaussian noise to time"""
    time_noise = time + gaussian_noise(time)
    return time_noise


def g_matrix():
    """Define G matrix. Picks out the squares that each ray (both directions) passes through. Only contains 1's and 0's, not velocities.
    First 12 rows are the left rays, last 12 are the right rays. The columns contain the flattened Nx times Ny matrix"""
    length = Nx * Ny
    G_detect1 = np.zeros((N_detector, length))
    G_detect2 = 1 * G_detect1
    for i in range(N_detector):
        G_detect1[i, :] = np.flip(np.eye(Ny, Nx, k=Ny-i), axis=1).flatten()
        G_detect2[i, :] = np.eye(Ny, Nx, k=1+i).flatten()
    G = np.vstack((G_detect1, G_detect2))
    return G


def illustrate_G():
    """Illustrate and save the G matrix"""
    G = g_matrix()
    fig, ax = plt.subplots(dpi=200)
    ax.imshow(G)
    ax.set(ylabel="Detector", xlabel="xy flattened")
    fig.tight_layout()
    plt.savefig(SAVE_DIR + "G_matrix.png")


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


def tikhonov(time, illustrate_find_eps=False):
    """Calculate the optimal m value using Tikhonov regularization with the constraint on epsilon: ||Gm - d_obs|| ~= variance(t_obs) * N_data 

    Args:
        time (1darray): Time values
        illustrate_find_eps (bool, optional): If true, draw the plot that shows the epsilon value that gets the optimal m value. Defaults to False.

    Returns:
        (float, float): m_optimal, epsilon_optional
    """
    # Load values
    std = 1 / 18 * norm2(time)
    time_obs = time_observed(time)
    G = g_matrix()
    # Calculate terms without epsilon for speeding up the loop
    GTG = G.T @ G
    GTd = G.T @ time_obs
    I = np.eye(GTG.shape[0])
    
    # Loop values
    eps_vals = np.linspace(1e-5, 1e1, 200)
    loss_arr = np.zeros_like(eps_vals)
    # Find m and store loss
    for i, eps in enumerate(eps_vals):
        m = np.linalg.inv(GTG + eps ** 2 * I) @ GTd  # Tikhonov regularization expression
        misfit = norm2(time_obs - G @ m) ** 2 
        loss = misfit - time_obs.size * std ** 2  # Should be zero for optimal epsilon
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
        plt.savefig(SAVE_DIR + f"optimal_eps{eps_optimal:.4f}.png")
    return m_optimal, eps_optimal


def illustrate_solution(time):
    m_optimal, eps_optimal = tikhonov(time)
    fig, ax = plt.subplots(dpi=200)
    im = ax.imshow(m_optimal.reshape(Ny, Nx))
    ax.set(title=fr"$\epsilon=${eps_optimal:.3f}")
    plt.colorbar(im)
    fig.tight_layout()
    plt.savefig(SAVE_DIR + f"solution_eps{eps_optimal:.4f}.png")


# -- 4. Run the code --
if __name__ == "__main__":
    # OBS Somewhere I update arrays that should not be updated (And I really can't find where...), so only one of the following lines can be run at a time!
    
    tikhonov(time_data, illustrate_find_eps=True)  # Plot the loss vs epsilon graph
    #illustrate_G()  # Plot the G matrix
    #illustrate_solution(time_data)  # Plot the s values for the box setup
    #illustrate_solution(time_data_delta)  # Plot the s values for the delta square setup
    
    pass  # In case all four lines above are commented out