import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing

from tqdm import tqdm

class Conv2dMod(nn.Module):
    """Some Information about Conv2dMod"""
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-8):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight) # initialize weight
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.eps = eps

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
        weight = w2 * (w1 + 1)

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
        x = F.conv2d(x, weight, stride=1, padding=0, groups=N)
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

class Conv2dModBlock(nn.Module):
    def __init__(self, input_channels, output_channels, style_dim, kernel_size=3):
        super(Conv2dModBlock, self).__init__()
        self.conv = Conv2dMod(input_channels, output_channels, kernel_size=kernel_size)
        self.affine    = nn.Linear(style_dim, output_channels)
        self.bias      = PointwiseBias(output_channels)
    
    def forward(self, x, y):
        x = self.conv(x, self.affine(y))
        x = self.bias(x)
        return x

class ToRGB(nn.Module):
    def __init__(self, channels, style_dim):
        super(ToRGB, self).__init__()
        self.conv   = Conv2dMod(channels, 3, 1)
        self.affine = nn.Linear(style_dim, 3)

    def forward(self, x, y):
        x = self.conv(x, self.affine(y))
        return x


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

class EqualLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EqualLinear, self).__init__()
        self.weight = torch.randn(output_dim, input_dim)
        self.bias = torch.zeros(output_dim)
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class MappingNetwork(nn.Module):
    def __init__(self, style_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()
        self.seq = nn.Sequential(*[EqualLinear(style_dim, style_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(style_dim)
    def forward(self, x):
        return self.seq(self.norm(x))

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, latent_channels, output_channels, style_dim, num_latent_layers=0, kernel_size=3, activation=nn.LeakyReLU, upscale=True):
        super(GeneratorBlock, self).__init__()
        self.conv1 = nn.conv_in = Conv2dModBlock(input_channels, latent_channels, style_dim, kernel_size=kernel_size)
        self.act1 = activation()
        self.conv2 = nn.conv_in = Conv2dModBlock(latent_channels, output_channels, style_dim, kernel_size=kernel_size)
        self.act2 = activation()
        self.to_rgb = ToRGB(output_channels, style_dim)
        if upscale:
            self.upscale = nn.Upsample(scale_factor=2)
        else:
            self.upscale = nn.Identity()

    def forward(self, x, y):
        x = self.upscale(x)
        x = self.conv1(x, y)
        x = self.act1(x)
        x = self.conv2(x, y)
        x = self.act1(x)
        rgb = self.to_rgb(x, y)
        return x, rgb

class Generator(nn.Module):
    def __init__(self, initial_channels=512, style_dim=512):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList([])
        self.last_channels = initial_channels
        self.const = nn.Parameter(torch.randn(initial_channels, 4, 4))
        self.upscale = nn.Upsample(scale_factor=2)
        self.style_dim = style_dim

        self.add_layer(initial_channels, upscale=False)

    def forward(self, y):
        if type(y) != list:
            y = [y] * len(self.layers)
        x = self.const.repeat(y[0].shape[0], 1, 1, 1)
        rgb_out = None
        for i in range(len(self.layers)):
            x, rgb = self.layers[i](x, y[i])
            if rgb_out == None:
                rgb_out = rgb
            else:
                rgb_out = self.upscale(rgb_out) + rgb
        return rgb_out

    def add_layer(self, channels, upscale=True):
        self.layers.append(GeneratorBlock(self.last_channels, (self.last_channels + channels)//2, channels, self.style_dim, upscale=upscale))
        self.last_channels = channels

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, latent_channels, output_channels, downscale=True, activation=nn.LeakyReLU):
        super(DiscriminatorBlock, self).__init__()
        self.from_rgb = nn.Conv2d(3, input_channels, 1, 1, 0)
        self.conv1 = nn.Conv2d(input_channels, latent_channels, 3, 1, 1)
        self.act1  = activation()
        self.conv2 = nn.Conv2d(latent_channels, output_channels, 3, 1, 1)
        self.act2  = activation()
        self.res   = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        if downscale:
            self.downscale = nn.AvgPool2d(kernel_size=2)
        else:
            self.downscale = nn.Identity()

    def forward(self, x):
        r = self.res(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x += r
        x = self.downscale(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, initial_channels=512):
        super(Discriminator, self).__init__()
        self.last_channels = initial_channels
        self.layers = nn.ModuleList([])
        self.pool4x = nn.AvgPool2d(kernel_size=4)
        self.fc1 = nn.Linear(initial_channels + 1, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.add_layer(initial_channels, downscale=False)
     
    def forward(self, rgb):
        x = self.layers[0].from_rgb(rgb)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.pool4x(x)
        x = x.reshape(x.shape[0], -1)
        sigma = torch.std(x, dim=0, keepdim=False).mean().repeat(x.shape[0], 1) # Minibatch Std.
        x = torch.cat([x,sigma], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def add_layer(self, channels, downscale=True):
        self.layers.insert(0, DiscriminatorBlock(self.last_channels, (self.last_channels + channels)//2, channels, downscale=downscale))
        self.last_channels = channels

class GAN(nn.Module):
    def __init__(self, initial_channels=512, style_dim=512):
        super(GAN, self).__init__()
        self.mapping_network = MappingNetwork(style_dim)
        self.generator = Generator(initial_channels, style_dim)
        self.discriminator = Discriminator(initial_channels)
        self.style_dim = style_dim
        self.initial_channels = initial_channels
    
    def train_epoch(self, dataloader, optimizer, device, dtype=torch.float32):
        for i, real in tqdm(enumerate(dataloader)):
            N = real.shape[0]
            D, M, G = self.generator, self.mapping_network, self.discriminator
            
            # train generator
            M.zero_grad()
            G.zero_grad()
            z = torch.randn(N, self.tyle_dim, device=device, dtype=dtype)
            w = self.mapping_network(z)

            fake = G(w)
            generator_loss = -D(fake).mean()
            generator_loss.backward()

            # train discriminator
            fake = fake.detach()
            discriminator_fake_loss = -torch.minimum(-D(fake)-1, torch.zeros(N, 0))
            discriminator_real_loss = -torch.minimum(D(real)-1, torch.zeros(N, 0))
            discriminator_loss = discriminator_fake_loss + discriminator_real_loss
            discriminator_loss.backward()

            # update parameter
            optimizer.step()

            tqdm.write(f"G: {generator_loss.item():.4f} D{generator_loss.item():.4f}")

    def train_resolution(self, dataset, num_epoch, device, dtype=torch.float32):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=multiprocessing.cpu_count())
        for i in range(num_epoch):
            self.train_epoch(dataloader, optimizer, device, dtype=dtype)

    def train(self, dataset,  num_epoch=100, batch_size=48, max_resolution=1024, device=torch.device('cpu'), dtype=torch.float32):
        resolution = -1
        while resolution <= max_resolution:
            num_layers = len(self.generator.layers)
            bs = batch_size // (2**(num_layers-1))
            ch = self.initial-channels // (2 ** (num_layers-1))
            if bs < 4:
                bs = 4
            if ch < 12:
                ch = 12
            self.train_resolution(dataset, num_epoch, device, dtype)
            resolution = 4 * (2 ** (num_layers-1))
            if resolution > max_resolution:
                break
            self.generator.add_layer(ch)
            self.discriminator.add_layer(ch)
