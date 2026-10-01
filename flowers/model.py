import torch
from torch.nn.utils.parametrizations import spectral_norm

def init_conv(alpha):
    def run(module):
        if isinstance(module, torch.nn.Conv2d) == False: return
        torch.nn.init.kaiming_normal_(module.weight, alpha)
        if module.bias is None: return
        torch.nn.init.normal_(module.bias, 0, 0.01)
    return run

def linear(in_dim, out_dim, use_norm=True):
    def norm(module):
        return spectral_norm(module) if use_norm else module
    return [
        norm(torch.nn.Linear(in_dim, out_dim)),
        torch.nn.LeakyReLU()]

def block_shuffle(in_dim, out_dim, scale, count_a, count_b):
    if scale < 0: 
        mid_dim = in_dim * 4
    elif scale > 0: 
        mid_dim = in_dim // 4
    else: 
        mid_dim = in_dim
    return [
        *[Residual(in_dim) for _ in range(count_a)],
        torch.nn.PixelShuffle(2) if scale > 0 else torch.nn.Identity(),
        torch.nn.PixelUnshuffle(2) if scale < 0 else torch.nn.Identity(),
        torch.nn.Conv2d(mid_dim, out_dim, 1),
        *[Residual(out_dim) for _ in range(count_b)]]

def block(in_dim, out_dim, scale, count_a, count_b):
    return [
        *[Residual(in_dim) for _ in range(count_a)],
        torch.nn.Upsample(scale_factor=2) if scale > 0 else torch.nn.Identity(),
        torch.nn.MaxPool2d(2) if scale < 0 else torch.nn.Identity(),
        torch.nn.Conv2d(in_dim, out_dim, 1),
        *[Residual(out_dim) for _ in range(count_b)]]

def block_critic(in_dim, out_dim, kernel):
    return [
        spectral_norm(torch.nn.Conv2d(in_dim, out_dim, kernel, 1, kernel // 2)),
        torch.nn.LeakyReLU(),
        spectral_norm(torch.nn.Conv2d(out_dim, out_dim, kernel, 2, kernel // 2)),
        torch.nn.LeakyReLU()]

class Residual(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim, 3, 1, "same"),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(dim, dim, 3, 1, "same"))
        self.beta = torch.nn.Parameter(torch.randn(1, dim, 1, 1) * 0.01)

    def forward(self, input):
        buffer = self.layers(input)
        buffer = buffer * self.beta + input
        return torch.nn.functional.leaky_relu(buffer)

class LatentGenerator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *linear(100, 256, False),
            *linear(256, 256, False),
            *linear(256, 256, False),
            torch.nn.Unflatten(1, (256, 1, 1)),
            torch.nn.ConvTranspose2d(256, 512, 4),
            *block(512, 256, 1, 4, 0),
            *block(256, 128, 1, 4, 0),
            *block(128, 64, 1, 4, 4),
            torch.nn.Conv2d(64, 8, 1))
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(256, 50, 4),
            torch.nn.Flatten(1))

    def forward(self, input):
        return self.layers(input)

class LatentCritic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            *block_critic(8, 64, 5),
            *block_critic(64, 128, 3),
            *block_critic(128, 256, 3))
        self.critic = torch.nn.Sequential(
            torch.nn.AvgPool2d(4),
            torch.nn.Flatten(),
            *linear(256, 256),
            *linear(256, 256),
            spectral_norm(torch.nn.Linear(256, 1)))

    def forward(self, input):
        features = self.backbone(input)
        return self.critic(features), features
        
class Autoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 1),
            *block_shuffle(32, 64, -1, 2, 0),
            *block_shuffle(64, 128, -1, 2, 0),
            *block_shuffle(128, 256, -1, 2, 2),
            torch.nn.Conv2d(256, 8, 1))
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(8, 256, 1),
            *block_shuffle(256, 128, 1, 4, 0),
            *block_shuffle(128, 64, 1, 4, 0),
            *block_shuffle(64, 32, 1, 4, 4),
            torch.nn.Conv2d(32, 3, 1))

    def forward(self, input):
        latent = self.encoder(input)
        return self.decoder(latent), latent

class Critic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            *block_critic(3, 32, 7),
            *block_critic(32, 64, 5),
            *block_critic(64, 128, 3),
            *block_critic(128, 256, 3),
            torch.nn.AvgPool2d(4),
            torch.nn.Flatten(),
            *linear(256, 256),
            *linear(256, 256),
            spectral_norm(torch.nn.Linear(256, 1))])

    def forward(self, input):
        buffer = input
        features = [input]
        for layer in self.layers[:-1]:
            buffer = layer(buffer)
            if isinstance(layer, torch.nn.Conv2d):
                features.append(buffer)
        return self.layers[-1](buffer).flatten(), features
