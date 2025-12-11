# This file is just a placeholder to illustrate how to set up different noise schedules
# The code for running training on different schedules was ran in a Google Colab notebook
# TODO: Merge this part of the code to training_code.py file

import argparse
from noise_schedulers import get_sigma_scheduled

def load_data(allowed_classes=[1,2,3], split="train", sigma_1=1.0, sigma_L=0.01, L=10, batch_size=32, schedule_type='geometric'):
    # Function to illustrate loading under specific noise schedule
    sigmas = get_sigma_scheduled(sigma_1, sigma_L, L, schedule_type)
    return None, sigmas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule_type", type=str, choices=["geometric", "linear", "sqrt", "quadratic", "sigmoid"], help="Noise schedule type to use")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    schedule_type = args.schedule_type
    batch_size = args.batch_size

    dataloader, sigmas = load_data(batch_size=batch_size, schedule_type=schedule_type, allowed_classes=[0,1,2])

    # Remaining of the training loop
    