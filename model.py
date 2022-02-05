import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Conv2dMod(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-8, groups=1):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels//groups, kernel_size, kernel_size, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight) # initialize weight
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.groups = groups

    def forward(self, x, y):
        # x: (batch_size, input_channels, H, W) 
        # y: (batch_size, output_channels)
        # self.weight: (output_channels, input_channels, kernel_size, kernel_size)
        N, C, H, W = x.shape
        
        # reshape weight
        w1 = y[:, None, :, None, None]
        w1 = w1.swapaxes(1, 2)
        w2 = self.weight[None, :, :, :, :]
        # modulate
        weight = w1 * w2

        # demodulate
        d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
        weight = weight * d
        # weight: (batch_size, output_channels, input_channels, kernel_size, kernel_size)
        
        # reshape
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N, *ws)
        
        
        # padding
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='replicate')
        
        # convolution
        x = F.conv2d(x, weight, stride=1, padding=0, groups=N * self.groups)
        x = x.reshape(N, self.output_channels, H, W)

        return x

class PointwiseBias(nn.Module):
    """Some Information about Noise"""
    def __init__(self, channels):
        super(PointwiseBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float32))

    def forward(self, x):
        # x: (batch_size, channels, H, W)
        # self.noise: (channels)
        
        # add bias
        bias = self.bias.repeat(x.shape[0], 1, x.shape[2], x.shape[3]).reshape(x.shape)
        x = x + bias
        return x

class Conv2dNeXtMod(nn.Module):
    def __init__(self, input_channels, output_channels, style_dim):
        super(Conv2dNeXtMod, self).__init__()
        self.depthwise = Conv2dMod(input_channels, input_channels, kernel_size=7, groups=input_channels)
        self.pontwise  = Conv2dMod(input_channels, output_channels, kernel_size=1)
        self.affine_dw = nn.Linear(style_dim, input_channels)
        self.affine_pw = nn.Linear(style_dim, output_channels)
        self.bias      = PointwiseBias(output_channels)
    
    def forward(self, x, y):
        x = self.depthwise(x, self.affine_dw(y))
        x = self.pontwise(x, self.affine_pw(y))
        x = self.bias(x)
        return x

class ToRGB(nn.Module):
    def __init__(self, channels, style_dim):
        super(ToRGB, self).__init__()
        self.conv   = Conv2dMod(channels, 3, 1)
        self.affine = nn.Linear(style_dim, 3)

    def forward(self, x, y):
        x = self.conv(x, self.affine(y))

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, latent_channels, output_channel, style_dim):
        super(GeneratorBlock, self).__init__()

class Blur(nn.Module):
    def __init__(self):
        super(Blur, self).__init__()
        self.kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32)
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel[None, None, :, :]
    def forward(self, x):
        shape = x.shape
        # padding
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # reshape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        # convolution
        x = F.conv2d(x, self.kernel.to(x.device), stride=1, padding=0, groups=x.shape[1])
        # reshape
        x = x.reshape(shape)
        return x

class HighPass(nn.Module):
    def __init__(self):
        super(Blur, self).__init__()
        identity = torch.tensor([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]])
        self.kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32)
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel[None, None, :, :]
        self.kernel = identity - self.kernel
    def forward(self, x):
        shape = x.shape
        # padding
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # reshape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        # convolution
        x = F.conv2d(x, self.kernel.to(x.device), stride=1, padding=0, groups=x.shape[1])
        # reshape
        x = x.reshape(shape)
        return x
