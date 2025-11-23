import torch 
import numpy as np
import matplotlib.pyplot as plt

def get_sigma_scheduled(sigma_1, sigma_L, L, schedule_type="geometric"):
    if schedule_type == "geometric":
        r = (sigma_L / sigma_1)**(1/(L - 1))
        sigmas = [sigma_1 * (r**i) for i in range(L)]
    elif schedule_type == "linear":
        sigmas = torch.linspace(sigma_1, sigma_L, L).tolist()
    elif schedule_type == "quadratic":
        t = torch.linspace(0, 1, L)
        sigmas = (sigma_1 + (sigma_L - sigma_1) * (t**2)).tolist()
    elif schedule_type == "sqrt":
        t = torch.linspace(0, 1, L)
        sigmas = (sigma_1 + (sigma_L - sigma_1) * torch.sqrt(t)).tolist()
    elif schedule_type == "sigmoid":
        sigmoid = torch.sigmoid(torch.linspace(-6, 6, L))
        norm_sigmoid = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
        sigmas = (sigma_1 + (sigma_L - sigma_1) * norm_sigmoid).tolist()
    else:
        raise ValueError(f"{schedule_type} is not supported.")
    return sigmas

def plot_sigma_over_schedulers(sigma_1, sigma_L, L, schedule_types, use_log=False):
    schedule_types = ['geometric', 'linear', 'quadratic', 'sqrt', 'sigmoid']
    for schedule_type in schedule_types:
        sigmas = get_sigma_scheduled(sigma_1, sigma_L, L, schedule_type=schedule_type)
        sigmas = np.log(sigmas) if use_log else sigmas
        plt.plot(torch.arange(1, L+1), sigmas, label=schedule_type)
    plt.title("Noise Schedule Over Time for Different Schedules")
    plt.xlabel("Level index i")
    plt.ylabel("Noise schedules")
    plt.legend()
    if use_log:
        plt.savefig("results/sigma_log_comparison.png")
    else:
        plt.savefig("results/sigma_comparison.png")
    plt.show()

def log_sigma_over_schedulers(sigma_1, sigma_L, L, schedule_types):
    for schedule_type in schedule_types:
        sigmas = get_sigma_scheduled(sigma_1, sigma_L, L, schedule_type=schedule_type)
        print(f"Schedule type: {schedule_type} - {[np.round(sigma,3) for sigma in sigmas]}")

if __name__ == "__main__":
    sigma_1=1.0
    sigma_L=0.01
    L=10
    schedule_types = ['geometric', 'linear', 'quadratic', 'sqrt', 'sigmoid']


    
