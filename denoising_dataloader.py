import torch
import datasets
from torchvision import transforms
import pandas as pd
import torch
import random
from noise_schedulers import get_sigma_scheduled

# TO DO: Add sigma to the training sample. So the new sample will be (original_image, noisy_image, sigma_index)

class MakeNoisySamples:
    def __init__(self, sigmas):
        self.sigmas = sigmas
    
    def __call__(self, original_img):
        noisy_imgs = [torch.randn_like(original_img) * self.sigmas[i] for i in range(len(self.sigmas))]
        noisy_imgs = [original_img + noisy_imgs[i] for i in range(len(noisy_imgs))]
        return [(original_img, noisy_imgs[i], self.sigmas.index(self.sigmas[i]), self.sigmas[i]) for i in range(len(noisy_imgs))] # list of tuples (original, noisy)

class ScoreGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_name="mnist", 
                 split="train", 
                 sigma_1=1.0,
                 sigma_L=0.01,
                 L=10,
                 allowed_label_classes=None,
                 schedule_type="geometric"):
        
        super().__init__()
        self.dataset_name = dataset_name

        # r = (sigma_L / sigma_1)**(1/(L - 1))
        # sigmas = [sigma_1 * (r**i) for i in range(L)]
        sigmas = get_sigma_scheduled(sigma_1, sigma_L, L, schedule_type)
        
        if self.dataset_name == "mnist":
            dataset_name = "ylecun/mnist"
            data_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    MakeNoisySamples(sigmas)
                ]
            )

        data = datasets.load_dataset(dataset_name)
        data = pd.DataFrame(data[split])
        if allowed_label_classes:
            data = data[data["label"].isin(allowed_label_classes)]

        data = data["image"].tolist()
        images = [data_transforms(img) for img in data] # list of lists of tuples (original, noisy)
        
        # making a single list of tuples (original, noisy)
        # from list[list[(original, noisy)]]
        self.orig_noisy_pairs = []
        for i in range(len(images)):
            for j in range(len(images[i])):
                self.orig_noisy_pairs.append(images[i][j])

        random.shuffle(self.orig_noisy_pairs)
    
    def __len__(self):
        return len(self.orig_noisy_pairs)

    def __getitem__(self, index):
        return {
            "original": self.orig_noisy_pairs[index][0],
            "noisy": self.orig_noisy_pairs[index][1],
            "sigma_index": self.orig_noisy_pairs[index][2],
            "sigma_value": self.orig_noisy_pairs[index][3]  
        }

class MyCollate:
    def __init__(self):
        pass

    def __call__(self, batch):

        original_images = torch.cat([batch[i]["original"] for i in range(len(batch))], dim=0)
        noisy_images = torch.cat([batch[i]["noisy"] for i in range(len(batch))], dim=0)
        sigma_indices = torch.tensor([batch[i]["sigma_index"] for i in range(len(batch))])
        sigma_values = torch.tensor([batch[i]["sigma_value"] for i in range(len(batch))])

        return {
            "original": original_images.unsqueeze(1),
            "noisy": noisy_images.unsqueeze(1),
            "sigma_index": sigma_indices,
            "sigma_value": sigma_values
        }     


# ---------------- Testing the dataloader ---------------- ##
# dataset = ScoreGenerationDataset(dataset_name="mnist", split="train", sigma_1=1.0, sigma_L=0.01, L=10, allowed_label_classes=[5,8])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=MyCollate())
# batch = next(iter(dataloader))  
# print(batch["original"].shape)
# print(batch["noisy"].max(), batch["noisy"].min())
# print(batch["original"].max(), batch["original"].min())

# ----------------- Plot the images --------------------- ##
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(4, 8, figsize=(12, 6))
# for i in range(4):
#     for j in range(8):
#         axs[i, j].imshow(batch["noisy"][i*8 + j].squeeze(), cmap='gray')
#         axs[i, j].axis('off')
#         axs[i, j].set_title(f"σ_idx: {batch['sigma_index'][i*8 + j].item()}, σ: {batch['sigma_value'][i*8 + j].item():.2f}")

# fig1, axs1 = plt.subplots(4, 8, figsize=(12, 6))

# for i in range(4):
#     for j in range(8):
#         axs1[i, j].imshow(batch["original"][i*8 + j].squeeze(), cmap='gray')
#         axs1[i, j].axis('off')
#         axs1[i, j].set_title("Original")
# plt.show()
# --------------------------------------------------------------- #