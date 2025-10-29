import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class CondInstanceNorm2d_plus(nn.Module):
    """
    input: x = B x C x H x W
    """
    def __init__(self, L, C):
        super().__init__()
        self.C = C
        self.gamma_embeddings = nn.Embedding(L, C) # we have a single scalar GAMMA for each channel 
                                                   # which is chosen based on the index of noise level
        self.beta_embeddings = nn.Embedding(L, C)
        self.alpha_embeddings = nn.Embedding(L, C)
        
        self.instance_norm = nn.InstanceNorm2d(C, affine=False)
    
    def forward(self, x, sigma_index):
        """
        x : B x C x H x W
        sigma_index : B
        """

        gamma = self.gamma_embeddings(sigma_index)  # B x C : gamma for each channel for each example in batch
        gamma = gamma.view(-1, self.C, 1, 1)        # B x C x 1 x 1
        
        beta = self.beta_embeddings(sigma_index)
        beta = beta.view(-1, self.C, 1, 1)          # B x C x 1 x 1

        alpha = self.alpha_embeddings(sigma_index)
        alpha = alpha.view(-1, self.C, 1, 1)        # B x C x 1 x 1

        mu = torch.mean(x, dim=(2,3))               # B x C : mean pixel value per channel 
        m = torch.mean(mu, dim=-1)                  # B : mean of all the means
        v = torch.var(mu, dim=-1)                   # B : Variance of all the means
        standard_mu = (mu-m)/(torch.sqrt(v)+1e-6)   # B x C

        x = self.instance_norm(x)   # B x C x H x W
        x = gamma*x                 # (B x C x 1 x 1) . (B x C x H x W) = (B x C x H x W)
        x = beta+x                  # (B x C x 1 x 1) . (B x C x H x W) = (B x C x H x W)
        x = alpha*standard_mu + x   # (B x C x 1 x 1) . (B x C x H x W) = (B x C x H x W)

        return x
        
class ChainedResidualPooling(nn.Module):
    def __init__(self, num_blocks, num_layers_per_block, in_channels, L):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block

        self.norm_layers = nn.ModuleList()
        for i in range(num_blocks):
            self.norm_layers.append(CondInstanceNorm2d_plus(L, in_channels))
        
        self.CRP_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block_layers = nn.ModuleList()
            for j in range(num_layers_per_block):
                block_layers.append(
                    nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    conv3x3(in_channels=in_channels, out_channels=in_channels)
                    )
                )
            self.CRP_blocks.append(block_layers)

        self.act = nn.ELU()

    def forward(self, x, sigma_index):
        x = self.act(x)
        path = x
        for i in range(self.num_blocks):
            path = self.norm_layers[i](path, sigma_index)
            for layer in self.CRP_blocks[i]:
                path = layer(path)
                x = x + path
        return x
    

# TO DO: Test the code till now
        