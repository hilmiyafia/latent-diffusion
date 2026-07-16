import torch
from blocks import AutoencoderDown, AutoencoderUp, Residual, Project
from torch.nn.utils.spectral_norm import spectral_norm

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.project1 = Project(3, 16)
        self.encoder1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.encoder2 = torch.nn.Sequential(
            AutoencoderDown(16, 32),
            AutoencoderDown(32, 64),
            AutoencoderDown(64, 128),
            AutoencoderDown(128, 256),
            *[Residual(256) for _ in range(2)])
        self.encoder3 = torch.nn.Conv2d(256, 8, 3, 1, 1)
        self.project2 = Project(256, 8)
        self.project3 = Project(8, 512)
        self.decoder1 = torch.nn.Conv2d(8, 512, 3, 1, 1)
        self.decoder2 = torch.nn.Sequential(
            *[Residual(512) for _ in range(4)],
            AutoencoderUp(512, 256),
            AutoencoderUp(256, 128),
            AutoencoderUp(128, 64),
            AutoencoderUp(64, 32))
        self.decoder3 = torch.nn.Conv2d(32, 3, 3, 1, 1)
        self.project4 = Project(32, 3)
        
    def encode(self, input):
        buffer = self.encoder1(input) + self.project1(input)
        buffer = self.encoder2(buffer)
        buffer = self.encoder3(buffer) + self.project2(buffer)
        return buffer.tanh()
    
    def decode(self, input):
        buffer = self.decoder1(input) + self.project3(input)
        buffer = self.decoder2(buffer)
        buffer = self.decoder3(buffer) + self.project4(buffer)
        return buffer.tanh()
                
    def forward(self, input):
        latent = self.encode(input)
        return self.decode(latent)

class Critic(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            spectral_norm(torch.nn.Conv2d(3, 32, 3, 1, 1)),
            spectral_norm(torch.nn.Conv2d(32, 32, 7, 2, 3)),
            spectral_norm(torch.nn.Conv2d(32, 64, 3, 1, 1)),
            spectral_norm(torch.nn.Conv2d(64, 64, 7, 2, 3)),
            spectral_norm(torch.nn.Conv2d(64, 128, 3, 1, 1)),
            spectral_norm(torch.nn.Conv2d(128, 128, 5, 2, 2)),
            spectral_norm(torch.nn.Conv2d(128, 256, 3, 1, 1)),
            spectral_norm(torch.nn.Conv2d(256, 256, 3, 2, 1)),
            spectral_norm(torch.nn.Conv2d(256, 1, 4))])
        
    def forward(self, input):
        buffer = input
        features = []
        for layer in self.layers[:-1]:
            buffer = torch.nn.functional.leaky_relu(layer(buffer))
            features.append(buffer.flatten(1))
        return self.layers[-1](buffer).squeeze(), torch.cat(features, 1)
