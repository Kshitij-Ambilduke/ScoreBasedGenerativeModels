import argparse
import torch
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from ncsn_model import CondRefineNetDilated


def parse_args():
    p = argparse.ArgumentParser(description="Run sampling with CondRefineNetDilated")
    p.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for sampling")
    p.add_argument("--model-path", "-m", type=str, default="author_code/mnist/SAVED_MODEL_5.pt", help="Path to model .pt file")
    p.add_argument("--L", type=int, default=10, help="Number of levels")
    p.add_argument("--sigma-1", type=float, default=1.0, help="Sigma at level 1")
    p.add_argument("--sigma-L", type=float, default=0.01, help="Sigma at level L")
    p.add_argument("--T", type=int, default=100, help="Inner steps per level")
    p.add_argument("--eps", type=float, default=2.0e-5, help="Epsilon constant")
    p.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    p.add_argument("--out", "-o", type=str, default=None, help="Optional output image file to save the samples (e.g. out.png)")
    p.add_argument("--show", action="store_true", help="Show the plot interactively (default: False)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")

    batch_size = args.batch_size

    # instantiate model
    model = CondRefineNetDilated(input_channels=1, L=args.L, ngf=64)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # build sigmas schedule
    sigma_1 = args.sigma_1
    sigma_L = args.sigma_L
    L = args.L
    r = (sigma_L / sigma_1) ** (1 / (L - 1))
    sigmas = [sigma_1 * (r ** i) for i in range(L)]

    T = args.T
    eps = args.eps

    x_t = torch.rand(batch_size, 1, 28, 28).to(device)

    batch_snapshots = []

    start_time = time.time()

    for i in tqdm(range(L), desc="Levels"):
        alpha = eps * (sigmas[i] ** 2 / sigmas[-1] ** 2)
        labels = torch.full((batch_size,), i, device=device, dtype=torch.long)

        for j in range(T):
            z = torch.randn_like(x_t)
            with torch.no_grad():
                score = model(x_t, labels)
                x_t = x_t + (alpha) * score + math.sqrt(alpha * 2) * z
        batch_snapshots.append(x_t.detach().cpu().clone())

    end_time = time.time()
    print(f"Sampling completed in {end_time - start_time:.2f} seconds.")

    # plotting
    fig, axs = plt.subplots(batch_size, L, figsize=(L * 1.5, batch_size * 1.5), squeeze=False)

    for b in range(batch_size):  # Row: The specific image in the batch
        for l in range(L):  # Column: The level of refinement
            img_tensor = batch_snapshots[l][b]
            img = torch.clamp(img_tensor, 0.0, 1.0).squeeze().numpy()

            ax = axs[b, l]
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            if b == 0:
                ax.set_title(f"Level {l+1}", fontsize=10)

    plt.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved figure to {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()