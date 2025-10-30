from ncsn_model import CondRefineNetDilated
from denoising_dataloader import ScoreGenerationDataset, MyCollate
import torch

dataset = ScoreGenerationDataset(dataset_name="mnist", split="train", sigma_1=1.0, sigma_L=0.01, L=10, allowed_label_classes=[5,8])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=MyCollate())
model = CondRefineNetDilated(input_channels=1, L=10, ngf=64)

# TO DO: Write loss score matching loss
# TO DO: Add training loop 

def load_data(allowed_classes=[1,2,3], split="train", sigma_1=1.0, sigma_L=0.01, L=10, batch_size=32):
    dataset = ScoreGenerationDataset(dataset_name="mnist", 
                                     split=split, 
                                     sigma_1=sigma_1, 
                                     sigma_L=sigma_L, 
                                     L=L, 
                                     allowed_label_classes=allowed_classes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=MyCollate())
    r = (sigma_L / sigma_1)**(1/(L - 1))
    sigmas = [sigma_1 * (r**i) for i in range(L)]
    return dataloader, sigmas

def main():
    dataloader, sigmas = load_data()
    model = CondRefineNetDilated(input_channels=1, L=10, ngf=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for batch in dataloader:

        original = batch["original"]
        noisy = batch["noisy"]
        sigma_index = batch["sigma_index"]

        optimizer.zero_grad()
        outputs = model(noisy, sigma_index)
        sigmas_batch = torch.tensor([sigmas[i] for i in sigma_index])
        sigmas_batch = sigmas_batch.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        loss = sigmas_batch*outputs + (noisy - original)/sigmas_batch
        loss = (loss**2).sum(dim=[1, 2, 3])
        loss = 0.5 * loss.mean()        
        loss.backward()
        optimizer.step()
        
        print("Loss:", loss.item())

if __name__ == "__main__":
    main()
        