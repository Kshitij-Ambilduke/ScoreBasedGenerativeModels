from ncsn_model import CondRefineNetDilated
from denoising_dataloader import ScoreGenerationDataset, MyCollate
import torch

dataset = ScoreGenerationDataset(dataset_name="mnist", split="train", sigma_1=1.0, sigma_L=0.01, L=10, allowed_label_classes=[5,8])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=MyCollate())
model = CondRefineNetDilated(input_channels=1, L=10, ngf=64)

# TO DO: Write loss score matching loss
# TO DO: Add training loop 