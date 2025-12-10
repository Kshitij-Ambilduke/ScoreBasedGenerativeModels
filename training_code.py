from ncsn_model import CondRefineNetDilated
from denoising_dataloader import ScoreGenerationDataset, MyCollate
import torch
from tqdm import tqdm
import argparse

def load_data(allowed_classes=[0,1,2], split="train", sigma_1=1.0, sigma_L=0.01, L=10, batch_size=128, PCA_components=None,
                 PCA_model_save_path=None):
    dataset = ScoreGenerationDataset(dataset_name="fmnist", 
                                     split=split, 
                                     sigma_1=sigma_1, 
                                     sigma_L=sigma_L, 
                                     L=L, 
                                     allowed_label_classes=allowed_classes,
                                     PCA_components=PCA_components,
                                     PCA_model_save_path=PCA_model_save_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=MyCollate())
    r = (sigma_L / sigma_1)**(1/(L - 1))
    sigmas = [sigma_1 * (r**i) for i in range(L)]
    return dataloader, sigmas

def main():
    parser = argparse.ArgumentParser(description='Train conditional denoising model')
    parser.add_argument('--total-steps', type=int, default=10000, help='Total training steps')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--split', type=str, default='train', help='Dataset split')
    parser.add_argument('--allowed-classes', type=int, nargs='+', default=[0,1,2], help='Allowed label classes')
    parser.add_argument('--sigma-1', type=float, default=1.0, help='Sigma 1')
    parser.add_argument('--sigma-L', type=float, default=0.01, help='Sigma L')
    parser.add_argument('--L', type=int, default=10, help='Number of noise levels')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ngf', type=int, default=64, help='Number of generator feature maps')
    parser.add_argument('--save-prefix', type=str, default='SAVED_MODEL', help='Model save filename prefix')
    parser.add_argument('--device', type=str, default=None, help='Device to use, overrides auto-detect')
    parser.add_argument('--PCA-components', type=int, default=None, help='Number of PCA components to use (if any)')
    parser.add_argument('--PCA-model-save-path', type=str, default=None, help='Path to save PCA model (if PCA is used)')

    args = parser.parse_args()

    total_steps = args.total_steps
    print(f"Total steps: {total_steps}")

    dataloader, sigmas = load_data(allowed_classes=args.allowed_classes,
                                   split=args.split,
                                   sigma_1=args.sigma_1,
                                   sigma_L=args.sigma_L,
                                   L=args.L,
                                   batch_size=args.batch_size)

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CondRefineNetDilated(input_channels=1, L=args.L, ngf=args.ngf)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    epoch_steps = 0
    steps_list = []
    while steps < total_steps:
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

            loss = sigmas_batch * outputs + (noisy - original) / sigmas_batch
            loss = (loss ** 2).sum(dim=[1, 2, 3])
            loss = 0.5 * loss.mean()
            loss.backward()
            optimizer.step()

            print("Loss:", loss.item())
            epoch_loss += loss.item()
            steps_list.append(steps)

        epoch_loss /= len(dataloader)
        epoch_steps += 1
        save_name = f"{args.save_prefix}_{epoch_steps}_fmnist.pt"
        torch.save(model.state_dict(), save_name)

        print(f"Epoch Steps: {epoch_steps}, Average Loss: {epoch_loss}")
if __name__ == "__main__":
    main()