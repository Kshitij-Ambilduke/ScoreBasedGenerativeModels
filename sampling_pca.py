import torch
from ncsn_model import CondRefineNetDilated
import math 
from tqdm import tqdm
from sklearn.decomposition import PCA
import joblib
import time

model = CondRefineNetDilated(input_channels=1, L=10, ngf=64)
model.load_state_dict(torch.load("author_code/PCA/SAVED_MODEL_1_fmnist.pt"))
model.eval()

sigma_1 = 1.0
sigma_L = 0.01
L = 10
r = (sigma_L / sigma_1)**(1/(L - 1))
sigmas = [sigma_1 * (r**i) for i in range(L)]

T=100
eps = 2.0e-5
# x_t = torch.randn(1, 1, 28, 28)
x_t = torch.rand(1, 1, 8, 8) * 1.0

images = []

start_time = time.time()
for i in tqdm(range(L)):
    alpha = eps*(sigmas[i]**2/sigmas[-1]**2)
    for j in tqdm(range(T)):
        z = torch.randn(1, 1, 8, 8)
        with torch.no_grad(): 
            score = model(x_t, torch.tensor([i]))
            x_t = x_t + (alpha)*score + math.sqrt(alpha*2)*z
    images.append(x_t.squeeze().cpu().numpy())

# xt = torch.clamp(x_t, 0.0, 1.0)
PCA_model = joblib.load("author_code/PCA/PCA_64_012.joblib")
images = [PCA_model.inverse_transform(img.flatten()).reshape(28, 28) for img in images]

end_time = time.time()
print("---------------------------------------------------------------------")
print(f"Sampling completed in {end_time - start_time:.2f} seconds.")
print("---------------------------------------------------------------------")

images = [torch.clamp(torch.tensor(img), 0.0, 1.0) for img in images]

# plot all images in a grid
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for idx, ax in enumerate(axs.flatten()):
    ax.imshow(images[idx], cmap='gray')
    ax.axis('off')
    ax.set_title(f"Step {idx+1}")
plt.tight_layout()
plt.show()


