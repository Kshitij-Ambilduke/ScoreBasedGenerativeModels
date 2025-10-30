import torch
import torch.nn as nn
from functools import partial

def conv3x3(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def dilated_conv3x3(in_planes, out_planes, dilation, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)

####################### REFINE NET BACKBONE STARTS HERE ####################################

class CondInstanceNorm2d_plus(nn.Module):
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
        m = torch.mean(mu, dim=-1).view(-1,1)       # B x 1 : mean of all the means
        v = torch.var(mu, dim=-1).view(-1,1)        # B x 1: Variance of all the means
        standard_mu = (mu-m)/(torch.sqrt(v)+1e-6)   # B x C
        standard_mu = standard_mu.view(standard_mu.shape[0],standard_mu.shape[1],1,1)

        x = self.instance_norm(x)   # B x C x H x W
        x = gamma*x                 # (B x C x 1 x 1) . (B x C x H x W) = (B x C x H x W)
        x = beta+x                  # (B x C x 1 x 1) . (B x C x H x W) = (B x C x H x W)
        x = alpha*standard_mu + x   # (B x C x 1 x 1) . (B x C x H x W) = (B x C x H x W)

        return x


class CondChainedResidualPooling(nn.Module):
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
    

class CondResidualConvUnit(nn.Module):
    def __init__(self, num_blocks, num_layers_per_block, in_channels, L):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block

        self.norm_layers = nn.ModuleList()
        for i in range(num_blocks):
            self.norm_layers.append(CondInstanceNorm2d_plus(L=L, C=in_channels))
        
        self.RCU = nn.ModuleList()
        for i in range(num_blocks):
            blocks = nn.ModuleList()
            for j in range(num_layers_per_block):
                blocks.append(
                    nn.Sequential(
                        nn.ELU(),
                        conv3x3(in_channels=in_channels, out_channels=in_channels)
                    )
                )
            self.RCU.append(blocks)
    
    def forward(self, x, sigma_index):
        for i in range(self.num_blocks):
            residual = x
            x = self.norm_layers[i](x, sigma_index)
            for layer in self.RCU[i]:
                x = layer(x)
            x = x + residual
        
        return x


class CondMultiResFusion(nn.Module):
    def __init__(self, in_planes, num_layers, L, output_C):
        super().__init__()
        
        self.in_planes = in_planes #list of Channels per plane
        self.num_layers = num_layers
        self.output_C = output_C

        self.norm_layers = nn.ModuleList()
        for i in range(len(in_planes)):
            self.norm_layers.append(CondInstanceNorm2d_plus(L=L, C=in_planes[i]))

        self.MSF_layers = nn.ModuleList() #one per input -- ModuleList([ModuleList(), ModuleList(), ...])
        for i in range(len(in_planes)):
            blocks = nn.ModuleList()
            if num_layers==1: #if only one layer then in_channels --> out_channels directly
                    blocks.append(
                        nn.Sequential(
                            conv3x3(in_channels=in_planes[i], out_channels=output_C)
                        )
                    )
            else: #else (in_channels -> out_channels) -> (out_channels -> out_channels)^(n-1)
                blocks.append(
                        nn.Sequential(
                            conv3x3(in_channels=in_planes[i], out_channels=output_C)
                        )
                    )
                for j in range(num_layers-1):
                    blocks.append(
                        nn.Sequential(
                            conv3x3(in_channels=output_C, out_channels=output_C)
                        )
                    )
            self.MSF_layers.append(blocks)
                                    
    def forward(self, x_s, sigma_index, output_size):
        sum_x = torch.zeros(x_s[0].shape[0], self.output_C, *output_size, device=x_s[0].device)
        for i in range(len(self.in_planes)):
            x_s[i] = self.norm_layers[i](x_s[i], sigma_index)
            for layer in self.MSF_layers[i]:
                x_s[i] = layer(x_s[i])
            x_s[i] = nn.functional.interpolate(x_s[i], output_size, mode='bilinear', align_corners=True)
            sum_x = sum_x + x_s[i]
        return sum_x
    
        
class CondRefineNet(nn.Module):
    def __init__(self, in_planes, 
                 num_RCU_blocks, num_layers_per_RCU_block,
                 num_layers_per_MSF_block, output_C_MSF,
                 num_CRP_blocks,num_layers_per_CRP_block,
                 L):
        super().__init__()

        self.in_planes = in_planes
        #in_planes = list of Channels for each of the plane
        self.AdaptiveConv = nn.ModuleList()
        for i in range(len(in_planes)):
            self.AdaptiveConv.append(
                CondResidualConvUnit(num_blocks=num_RCU_blocks,
                                     num_layers_per_block=num_layers_per_RCU_block,
                                     L=L,
                                     in_channels=in_planes[i])
            )
        
        self.MRF = CondMultiResFusion(in_planes=in_planes,
                                      num_layers=num_layers_per_MSF_block,
                                      L=L,
                                      output_C=output_C_MSF)
        
        self.CRP = CondChainedResidualPooling(num_blocks=num_CRP_blocks,
                                              num_layers_per_block=num_layers_per_CRP_block,
                                              in_channels=output_C_MSF,
                                              L=L)

        self.OutputConv = CondResidualConvUnit(num_blocks=2,
                                               num_layers_per_block=num_layers_per_RCU_block,
                                               in_channels=output_C_MSF,
                                               L=L)

    def forward(self, x_s, sigma_index, output_shape):
        h_s = []
        
        # Adaptive Convolutional layer
        for i in range(len(x_s)):
            h = self.AdaptiveConv[i](x_s[i], sigma_index)
            h_s.append(h)
        
        # Multi-resolution fusion
        if len(self.in_planes)==1:
            h = h_s[0]
        else:
            h = self.MRF(h_s, sigma_index, output_shape)

        # Chained residual pooling
        h = self.CRP(h, sigma_index)

        # Output Convolution
        h = self.OutputConv(h, sigma_index)
    
        return h

################ REFINE NET BACKBONE ENDS HERE #####################

################ FEATURE EXTRACTION RESNET STARTS HERE ###############

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        else:
            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            )

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output

class ConditionalResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, resample=None, act=nn.ELU(),
                 normalization=CondInstanceNorm2d_plus, adjust_padding=False, dilation=None):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(C=input_dim, L=num_classes)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
            else:
                self.conv1 = nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1)
                self.normalize2 = normalization(C=input_dim, L=num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(C=output_dim, L=num_classes)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
                self.normalize2 = normalization(C=output_dim, L=num_classes)
                self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)
        
        self.normalize1 = normalization(C=input_dim, L=num_classes)

    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output

class CondRefineNetDilated(nn.Module):
    def __init__(self,
                 input_channels=1,
                 L=10,
                 ngf=64):
        super().__init__()

        self.norm = CondInstanceNorm2d_plus
        self.ngf = ngf 
        self.num_classes = L
        self.act = act = nn.ELU()
        # self.act = act = nn.ReLU(True)

        self.begin_conv = nn.Conv2d(input_channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(C=ngf, L=self.num_classes)

        self.end_conv = nn.Conv2d(ngf, input_channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2)]
        )

        # if config.data.image_size == 28:
        self.res4 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                        normalization=self.norm, adjust_padding=True, dilation=4),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                        normalization=self.norm, dilation=4)]
        )
        
        self.refine1 = CondRefineNet(in_planes=[2 * self.ngf],
                                      num_RCU_blocks=2,
                                      num_layers_per_RCU_block=2,
                                      num_layers_per_MSF_block=1,
                                      output_C_MSF=2 * self.ngf,
                                      num_CRP_blocks=2,
                                      num_layers_per_CRP_block=2,
                                      L=self.num_classes)
        
        self.refine2 = CondRefineNet(in_planes=[2 * self.ngf, 2 * self.ngf],
                                      num_RCU_blocks=2,
                                      num_layers_per_RCU_block=2,
                                      num_layers_per_MSF_block=1,
                                      output_C_MSF=2 * self.ngf,
                                      num_CRP_blocks=2,
                                      num_layers_per_CRP_block=2,
                                      L=self.num_classes)
        
        self.refine3 = CondRefineNet(in_planes=[2 * self.ngf, 2 * self.ngf],
                                      num_RCU_blocks=2,
                                      num_layers_per_RCU_block=2,
                                      num_layers_per_MSF_block=1,
                                      output_C_MSF=self.ngf,
                                      num_CRP_blocks=2,
                                      num_layers_per_CRP_block=2,
                                      L=self.num_classes)

        self.refine4 = CondRefineNet(in_planes=[self.ngf, self.ngf],
                                    num_RCU_blocks=2,
                                    num_layers_per_RCU_block=2,
                                    num_layers_per_MSF_block=1, 
                                    output_C_MSF=self.ngf,
                                    num_CRP_blocks=2,
                                    num_layers_per_CRP_block=2,
                                    L=self.num_classes)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            x = m(x, y)
        return x

    def forward(self, x, y):
        output = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, output, y) # resnet layer 1 o/p = B x ngf x H x W = layer1
        layer2 = self._compute_cond_module(self.res2, layer1, y) # resnet layer 2 o/p = B x 2ngf x H/2 x W/2 = layer2
        layer3 = self._compute_cond_module(self.res3, layer2, y) # resnet layer 3 o/p = B x 2ngf x H/4 x W/4 = layer3
        layer4 = self._compute_cond_module(self.res4, layer3, y) # resnet layer 4 o/p = B x 2ngf x H/8 x W/8 = layer4

        ref1 = self.refine1([layer4], y, layer4.shape[2:]) #RefineNet 1: input: layer4 (B x 2ngf x H/8 x W/8) output: ref1 (B x 2ngf x H/8 x W/8)
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[2:]) #RefineNet 2: input: layer3 (B x 2ngf x H/4 x W/4), ref1 (B x 2ngf x H/8 x W/8) output: ref2 (B x 2ngf x H/4 x W/4)

        ref3 = self.refine3([layer2, ref2], y, layer2.shape[2:]) #RefineNet 3: input: layer2 (B x 2ngf x H/2 x W/2), ref2 (B x 2ngf x H/4 x W/4) output: ref3 (B x ngf x H/2 x W/2)
        output = self.refine4([layer1, ref3], y, layer1.shape[2:]) #RefineNet 4: input: layer1 (B x ngf x H x W), ref3 (B x ngf x H/2 x W/2) output: output (B x ngf x H x W)

        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)
        return output

# ## ---------------- TO DO: Test the code till now -------------------
# x = torch.randn(4,1,28,28)  # batch_size=4, input_channels=1, height=32, width=32
# y = torch.randint(0,10,(4,)) # batch_size=4, num_classes=10
# # print("Y:", y.shape)

# model = CondRefineNetDilated(input_channels=1, L=10, ngf=64)
# out = model(x,y)
# print(out.shape)  # Expected output shape: (4, 1, 32, 32)
## ------------------------------------------------------------------
        