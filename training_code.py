from ncsn_model import CondRefineNetDilated
from denoising_dataloader import ScoreGenerationDataset, MyCollate
import torch
from tqdm import tqdm

# dataset = ScoreGenerationDataset(dataset_name="mnist", split="train", sigma_1=1.0, sigma_L=0.01, L=10, allowed_label_classes=[5,8])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=MyCollate())
# model = CondRefineNetDilated(input_channels=1, L=10, ngf=64)

# TO DO: Write loss score matching loss
# TO DO: Add training loop 

def load_data(allowed_classes=[0,1,2], split="train", sigma_1=1.0, sigma_L=0.01, L=10, batch_size=128):
    dataset = ScoreGenerationDataset(dataset_name="fmnist", 
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
    total_steps = 10000
    print(total_steps)
    dataloader, sigmas = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CondRefineNetDilated(input_channels=1, L=10, ngf=64)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    steps = 0
    epoch_steps = 0
    steps_list = []
    while steps<total_steps:
        epoch_loss = 0.0
        for batch in tqdm(dataloader, total=len(dataloader)):
            steps += 1 
            original = batch["original"].to(device)
            noisy = batch["noisy"].to(device)
            sigma_index = batch["sigma_index"].to(device)

            optimizer.zero_grad()
            outputs = model(noisy, sigma_index)
            sigmas_batch = torch.tensor([sigmas[i] for i in sigma_index])
            sigmas_batch = sigmas_batch.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(device)

            loss = sigmas_batch*outputs + (noisy - original)/sigmas_batch
            loss = (loss**2).sum(dim=[1, 2, 3])
            loss = 0.5 * loss.mean()        
            loss.backward()
            optimizer.step()

            print("Loss:", loss.item())
            epoch_loss += loss.item()
            steps_list.append(steps)
            
        epoch_loss /= len(dataloader)
        epoch_steps += 1
        torch.save(model.state_dict(), f"SAVED_MODEL_{epoch_steps}_fmnist.pt")

        print(f"Epoch Steps: {epoch_steps}, Average Loss: {epoch_loss}")
if __name__ == "__main__":
    main()