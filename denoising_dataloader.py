import torch
import datasets
from torchvision import transforms
import pandas as pd
import torch
import random

# TO DO: Add sigma to the training sample. So the new sample will be (original_image, noisy_image, sigma_index)

class MakeNoisySamples:
    def __init__(self, sigmas):
        self.sigmas = sigmas
    
    def __call__(self, original_img):
        noisy_imgs = [torch.randn_like(original_img) * self.sigmas[i] for i in range(len(self.sigmas))]
        noisy_imgs = [original_img + noisy_imgs[i] for i in range(len(noisy_imgs))]
        return [(original_img, noisy_imgs[i], self.sigmas.index(self.sigmas[i])) for i in range(len(noisy_imgs))] # list of tuples (original, noisy)

class ScoreGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_name="mnist", 
                 split="train", 
                 sigma_1=1.0,
                 sigma_L=0.01,
                 L=10,
                 allowed_label_classes=None):
        
        super().__init__()
        self.dataset_name = dataset_name

        r = (sigma_L / sigma_1)**(1/(L - 1))
        sigmas = [sigma_1 * (r**i) for i in range(L)]

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
            "sigma_index": self.orig_noisy_pairs[index][2]
        }

class MyCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        # print(len(batch))
        # for i in range(len(batch)):
        #     print(batch[i])
        #     break

        original_images = torch.cat([batch[i]["original"] for i in range(len(batch))], dim=0)
        noisy_images = torch.cat([batch[i]["noisy"] for i in range(len(batch))], dim=0)
        sigma_indices = torch.tensor([batch[i]["sigma_index"] for i in range(len(batch))])

        return {
            "original": original_images.unsqueeze(1),
            "noisy": noisy_images.unsqueeze(1),
            "sigma_index": sigma_indices
        }     


# # ---------------- Testing the dataloader ---------------- ##
# dataset = ScoreGenerationDataset(dataset_name="mnist", split="train", sigma_1=1.0, sigma_L=0.01, L=10, allowed_label_classes=[5,8])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=MyCollate())
# batch = next(iter(dataloader))  
# print(batch["original"].shape)

# # ----------------- Plot the images --------------------- ##
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(4, 8, figsize=(12, 6))
# for i in range(4):
#     for j in range(8):
#         axs[i, j].imshow(batch["noisy"][i*8 + j].squeeze(), cmap='gray')
#         axs[i, j].axis('off')
#         axs[i, j].set_title(f"σ_idx: {batch['sigma_index'][i*8 + j].item()}")

# fig1, axs1 = plt.subplots(4, 8, figsize=(12, 6))

# for i in range(4):
#     for j in range(8):
#         axs1[i, j].imshow(batch["original"][i*8 + j].squeeze(), cmap='gray')
#         axs1[i, j].axis('off')
#         axs1[i, j].set_title("Original")
# plt.show()
# # --------------------------------------------------------------- #