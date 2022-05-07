import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import random

class Conv2dMod(nn.Module):
    """Some Information about Conv2dMod"""
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-8, groups=1, demodulation=True):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels // groups, kernel_size, kernel_size, dtype=torch.float32))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # initialize weight
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.demodulation = demodulation
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
        weight = w2 * (w1 + 1)

        # demodulate
        if self.demodulation:
            d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d
        # weight: (batch_size, output_channels, input_channels, kernel_size, kernel_size)
        
        # reshape
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N * self.groups, *ws)
        
        
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
    def __init__(self, channels):
        super(ToRGB, self).__init__()
        self.conv   = nn.Conv2d(channels, 3, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
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
        super(HighPass, self).__init__()
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

class NoiseInjection(nn.Module):
    """Some Information about NoiseInjection"""
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.conv = nn.Conv2d(channels, 1, 1, 1, 0)
        
    def forward(self, x):
        gain = self.conv(x)
        noise = torch.randn(*x.shape).to(x.device) * gain
        return x + noise

class EqualLinear(nn.Module):
    def __init__(self, input_dim, output_dim, lr_mul=0.1):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.lr_mul = lr_mul
    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, self.bias *  self.lr_mul)

class MappingNetwork(nn.Module):
    def __init__(self, style_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()
        self.seq = nn.Sequential(*[nn.Sequential(EqualLinear(style_dim, style_dim), nn.LeakyReLU(0.2)) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(style_dim)
    def forward(self, x):
        return self.seq(self.norm(x))

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, latent_channels, output_channels, style_dim, num_latent_layers=0, kernel_size=3, upscale=True, p_drop=0):
        super(GeneratorBlock, self).__init__()
        self.conv1 = Conv2dModBlock(input_channels, latent_channels, style_dim, kernel_size=kernel_size)
        self.act1 = nn.LeakyReLU(0.2)
        self.noise1 = NoiseInjection(latent_channels)
        self.conv2 = Conv2dModBlock(latent_channels, output_channels, style_dim, kernel_size=kernel_size)
        self.act2 = nn.LeakyReLU(0.2)
        self.noise2 = NoiseInjection(output_channels)
        self.to_rgb = ToRGB(output_channels)
        self.p_drop = p_drop
        if upscale:
            self.upscale = nn.Upsample(scale_factor=2)
        else:
            self.upscale = nn.Identity()

    def forward(self, x, y):
        x = self.upscale(x)
        if self.p_drop <= random.random() or not self.training:
            x = self.conv1(x, y)
            x = self.act1(x)
            x = self.noise1(x)
        if self.p_drop <= random.random() or not self.training:
            x = self.conv2(x, y)
            x = self.act2(x)
            x = self.noise2(x)
        rgb = self.to_rgb(x)
        return x, rgb

class Generator(nn.Module):
    def __init__(self, initial_channels=512, style_dim=512, depth_dropout_probability=0.0):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList([])
        self.last_channels = initial_channels
        self.const = nn.Parameter(torch.randn(initial_channels, 4, 4))
        self.upscale = nn.Upsample(scale_factor=2)
        self.style_dim = style_dim
        self.blur = Blur()
        self.add_layer(initial_channels, upscale=False, p_drop=depth_dropout_probability)

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
                rgb_out = self.blur(self.upscale(rgb_out)) + rgb
        rgb_out = torch.tanh(rgb_out)
        return rgb_out

    def add_layer(self, channels, upscale=True, p_drop=0.0):
        self.layers.append(GeneratorBlock(self.last_channels, (self.last_channels + channels)//2, channels, self.style_dim, upscale=upscale, p_drop=p_drop))
        self.last_channels = channels

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, latent_channels, output_channels, downscale=True, p_drop=0.0):
        super(DiscriminatorBlock, self).__init__()
        self.from_rgb = nn.Conv2d(3, input_channels, 1, 1, 0)
        self.conv1 = nn.Conv2d(input_channels, latent_channels, 3, 1, 1, padding_mode='replicate')
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(latent_channels, output_channels, 3, 1, 1, padding_mode='replicate')
        self.act2 = nn.LeakyReLU(0.2)
        self.res = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        self.p_drop = p_drop
        if downscale:
            self.downscale = nn.AvgPool2d(kernel_size=2)
        else:
            self.downscale = nn.Identity()

    def forward(self, x):
        r = self.res(x)
        if self.p_drop <= random.random():
            x = self.conv1(x)
            x = self.act1(x)
        if self.p_drop <= random.random():
            x = self.conv2(x)
            x = self.act2(x)
        x = (x + r)
        x = self.downscale(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, initial_channels=512, depth_dropout_probability=0.0):
        super(Discriminator, self).__init__()
        self.last_channels = initial_channels
        self.layers = nn.ModuleList([])
        self.pool4x = nn.AvgPool2d(kernel_size=4)
        self.fc1 = nn.Linear(initial_channels + 1, 128)
        self.act1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)
        self.downscale = nn.Sequential(Blur(), nn.AvgPool2d(kernel_size=2))

        self.add_layer(initial_channels, downscale=False, p_drop=depth_dropout_probability)
     
    def forward(self, rgb):
        x = self.layers[0].from_rgb(rgb)
        for i in range(len(self.layers)):
            if i == 1:
                x += self.layers[1].from_rgb(self.downscale(rgb))
            x = self.layers[i](x)
        mb_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1) # Minibatch Std.
        x = self.pool4x(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, mb_std], dim=1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

    def add_layer(self, channels, downscale=True, p_drop=0.0):
        self.layers.insert(0, DiscriminatorBlock(channels, (self.last_channels + channels)//2, self.last_channels, downscale=downscale, p_drop=p_drop))
        self.last_channels = channels

class GAN(nn.Module):
    def __init__(self, initial_channels=512, style_dim=512):
        super(GAN, self).__init__()
        self.mapping_network = MappingNetwork(style_dim)
        self.generator = Generator(initial_channels, style_dim)
        self.discriminator = Discriminator(initial_channels)
        self.style_dim = style_dim
        self.initial_channels = initial_channels
    
    def train_epoch(self, dataloader, optimizers, device, dtype=torch.float32,augment_func=nn.Identity):
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        self.mapping_network = self.mapping_network.to(device)
        opt_m, opt_g, opt_d = optimizers
        self.to(device)
        for i, real in enumerate(dataloader):
            if real.shape[0] < 2:
                continue
            real = real.to(device).to(dtype)
            N = real.shape[0]
            G, M, D = self.generator, self.mapping_network, self.discriminator
            G.train()
            M.train()
            D.train()
            T = augment_func
            L = len(G.layers)
            # train generator
            M.zero_grad()
            G.zero_grad()
            z1 = torch.randn(N, self.style_dim, device=device, dtype=dtype)
            z2 = torch.randn(N, self.style_dim, device=device, dtype=dtype)
            mid = random.randint(1, L)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            w = [w1] * mid + [w2] * (L-mid)
            fake = G(w)

            generator_loss = -D(fake).mean()
            generator_loss.backward()
            opt_m.step()
            opt_g.step()

            # train discriminator
            fake = fake.detach()
            D.zero_grad()
            discriminator_fake_loss = -torch.minimum(-D(T(fake))-1, torch.zeros(N, 1).to(device)).mean()
            discriminator_real_loss = -torch.minimum(D(T(real))-1, torch.zeros(N, 1).to(device)).mean()
            discriminator_loss = discriminator_fake_loss + discriminator_real_loss
            discriminator_loss.backward()

            # update parameter
            opt_d.step()
            
            tqdm.write(f"Batch: {i} G: {generator_loss.item():.4f} D: {discriminator_loss.item():.4f}")

    def train_resolution(self, dataset, num_epoch, batch_size, device, dtype=torch.float32, result_dir='./results/', model_path='./model.pt', augment_func=nn.Identity()):
        optimizers = (
                optim.Adam(self.mapping_network.parameters(), lr=1e-5),
                optim.Adam(self.generator.parameters(), lr=1e-5),
                optim.Adam(self.discriminator.parameters(), lr=1e-5),
                )
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
        for i in tqdm(range(num_epoch)):
            self.train_epoch(dataloader, optimizers, device, dtype=dtype, augment_func=augment_func)
            torch.save(self, model_path)

            # save image
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            path = os.path.join(result_dir, f"{i}.jpg")
            img = self.generator(self.mapping_network(torch.randn(1, self.style_dim, dtype=dtype, device=device)))
            img = img.cpu().detach().numpy() * 127.5 + 127.5
            img = img[0].transpose(1, 2, 0)
            img = img.astype(np.uint8)
            img = Image.fromarray(img, mode='RGB')
            img.save(path)


    def train(self, dataset,  num_epoch=100, batch_size=32, max_resolution=1024, device=None, dtype=torch.float32, augment_func=nn.Identity()):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.set_dtype(dtype=dtype)
        resolution = -1
        while resolution <= max_resolution:
            num_layers = len(self.generator.layers)
            bs = batch_size // (2 ** max(num_layers-3, 0))
            ch = self.initial_channels // (2 ** max(num_layers-3, 0))
            if bs < 4:
                bs = 4
            if ch < 12:
                ch = 12
            resolution = 4 * (2 ** (num_layers-1))
            dataset.set_size(resolution)
            self.to(device)
            print(f"Start training with batch size: {bs} ch: {ch}")
            self.train_resolution(dataset, num_epoch, bs, device, dtype, augment_func=augment_func)
            if resolution >= max_resolution:
                break
            self.generator.add_layer(ch)
            self.discriminator.add_layer(ch)
            self.set_dtype(dtype)

    def set_dtype(self, dtype=torch.float16):
        for param in self.parameters():
            param.data = param.data.to(dtype)

    def generate_random_image(self, num_images):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images = []
        for i in range(num_images):
            z1 = torch.randn(1, self.style_dim).to(device)
            z2 = torch.randn(1, self.style_dim).to(device)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            L = len(self.generator.layers)
            mid = random.randint(1, L)
            style = [w1] * mid + [w2] * (L-mid)
            image = self.generator(style)
            image = image.detach().cpu().numpy()
            images.append(image[0])
        return images

    def generate_random_image_to_directory(self, num_images, dir_path="./tests"):
        images = self.generate_random_image(num_images)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for i in range(num_images):
            image = images[i]
            image = np.transpose(image, (1, 2, 0))
            image = image * 127.5 + 127.5
            image = image.astype(np.uint8)
            image = Image.fromarray(image, mode='RGB')
            image = image.resize((1024, 1024))
            image.save(os.path.join(dir_path, f"{i}.png"))
