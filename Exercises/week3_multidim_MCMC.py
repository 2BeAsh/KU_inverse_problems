import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def norm2(x):
    return np.sqrt(np.sum(x**2))


def gaussian_five_dim(x_vec):
    b1 = np.array([2, -1, 0, 0, 0.2])
    b2 = np.array([-1, 0.5, 0, 0.1, 0])
    b3 = np.array([-3, 2, -0.1, 0, 0])
    term1 = 0.1 * np.exp(-0.5 * norm2(x_vec - b1) ** 2 / 1.2 ** 2)
    term2 = 0.6 * np.exp(-0.5 * norm2(x_vec - b2) ** 2 / 0.9 ** 2)
    term3 = 0.3 * np.exp(-0.5 * norm2(x_vec - b3) ** 2 / 1. ** 2)
    return term1 + term2 + term3


def q(x_old):
    """Proposal distribution. Uniform vector with magnitude 0.5 or less is added to the old point"""
    vector = np.random.normal(size=x_old.size)  # 100 is arbitrary large number
    radius = 0.5
    on_the_ball = vector / norm2(vector) * radius 
    Dx_proposed = on_the_ball * np.random.uniform(0, 1)
    
    #Dx_proposed = np.random.uniform(-0.5, 0.5, x_old.size)
    #while norm2(Dx_proposed) > 0.5:
    #    Dx_proposed = np.random.uniform(-0.5, 0.5, x_old.size)
    return x_old + Dx_proposed


def sample_h_MCMC(steps):
    x0 = np.array([8, 8, 8, 8, 8])
    x_hist = np.empty((steps, x0.size))
    x_hist[0] = 1 * x0
    h_hist = np.empty(steps)
    for i in range(1, steps):
        x_old = x_hist[i-1, :]
        x_proposed = q(x_old)
        x_proposed = np.clip(a=x_proposed, a_min=-10, a_max=10)  # x values must be inside [-10, 10]
        # Since the proposal distribution Q(x, x0) = Q(x0, x) null dist cancels out
        dist_ratio = gaussian_five_dim(x_proposed) / gaussian_five_dim(x_old)
        if dist_ratio > 1 or dist_ratio > np.random.uniform():
            x_hist[i, :] = x_proposed
            h_hist[i] = gaussian_five_dim(x_proposed)
        else:
            x_hist[i, :] = x_old
            h_hist[i] = gaussian_five_dim(x_old)
            
    return x_hist, h_hist


def illustrate_log_h_transient(steps):
    fig, ax = plt.subplots(dpi=150)
    _, h_hist = sample_h_MCMC(steps)
    ax.plot(h_hist, "--.", label="h")
    ax.set(xlabel="Steps", ylabel="log h(x)", yscale="log")
    ax.legend()
    plt.show()
    
    
def illustrate_x1_histogram(steps):
    fig, ax = plt.subplots(dpi=150)
    x_hist, _ = sample_h_MCMC(steps)
    x1 = x_hist[:, 0]
    bins = int(np.sqrt(steps))
    ax.hist(x1, bins=bins, label="x1")
    ax.set(xlabel="Steps", ylabel="x1 frequency", title=f"Data points = {steps}, Bins = {bins}")
    ax.legend()
    plt.show()

def illustrate_x1x2_histogram(steps):
    fig, ax = plt.subplots(dpi=150)
    x_hist, _ = sample_h_MCMC(steps)
    transient = 1000
    x1 = x_hist[transient:, 0]
    x2 = x_hist[transient:, 1]
    bins = int(np.sqrt(x2.size))
    correlation = estimate_covariance_of_h(steps, x_vals=x_hist)
    ax.hist2d(x1, x2, bins=bins, range=[[-10, 10], [-10, 10]])
    ax.set(xlabel="x1", ylabel="x2")
    ax.set_title(f"Correlation = {correlation:.3f}, Points = {x2.size}, bins = {bins}", fontsize=10)
    plt.show()
    
    
def estimate_mean_of_h(steps):
    transient = 1000
    x_hist, _ = sample_h_MCMC(steps)
    x_hist = x_hist[transient:, :]
    mean = np.mean(x_hist, axis=0)
    print(f"The mean of h(x) ~ {mean}")


def estimate_covariance_of_h(steps, x_vals=None):
    transient = 1000
    if x_vals.any() == None:
        x_hist, _ = sample_h_MCMC(steps)
    else:
        x_hist = x_vals
    x_hist = x_hist[transient:, :]
    x1 = x_hist[:, 0]
    x2 = x_hist[:, 1]
    covariance_term = (x1 - x1.mean()) * (x2 - x2.mean())
    covariance_x1x2 = np.mean(covariance_term)
    correlation = covariance_x1x2 / (np.std(x1) * np.std(x2))
        
    print(f"Covariance x1 and x2 = {covariance_x1x2:.3f}")
    print(f"Correlation = {correlation:.3f}")
    return correlation

    
if __name__ == "__main__":
    steps = 300_000
    #illustrate_log_h_transient(steps)
    #illustrate_x1_histogram(steps)
    illustrate_x1x2_histogram(steps)
    #estimate_mean_of_h(steps=100_000)
    #estimate_covariance_of_h(steps)