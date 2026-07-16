import math
import torch

class LayerNorm(torch.nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channel)

    def forward(self, input):
        return self.norm(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    
def block(channel):
    return [
        LayerNorm(channel),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(channel, channel, 3, 1, 1),
        LayerNorm(channel),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(channel, channel, 3, 1, 1)]

class Project(torch.nn.Module):

    def __init__(self, a, b):
        super().__init__()
        weight = torch.zeros((b, a), dtype=torch.float32)
        if a == b:
            weight.fill_diagonal_(1)
        elif b > a:
            scale = a / b
            for i in range(b):
                j = round((i + 0.5) * scale - 0.5)
                j = min(max(0, j), a - 1)
                weight[i, j] = 1.0
        else:
            scale = a / b
            for i in range(b):
                left = i * scale
                right = (i + 1) * scale
                i0 = int(math.floor(left))
                i1 = int(math.ceil(right))
                for j in range(i0, i1):
                    overlap = max(0.0, min(right, j + 1) - max(left, j))
                    if overlap > 0:
                        weight[i, j] = overlap / scale
        self.register_buffer("weight", weight[..., None, None])

    def forward(self, input):
        return torch.nn.functional.conv2d(input, self.weight)

class Residual(torch.nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.layers = torch.nn.Sequential(*block(channel))
                
    def forward(self, input):
        return self.layers(input) + input
    
class ResidualUp(torch.nn.Module):

    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.PixelShuffle(2),
            Project(channel_in // 4, channel_out))
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(channel_in, channel_out, 4, 2, 1),
            *block(channel_out))
                
    def forward(self, input):
        return self.layers(input) + self.project(input)
    
class ResidualDown(torch.nn.Module):

    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.PixelUnshuffle(2),
            Project(channel_in * 4, channel_out))
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(channel_in, channel_out, 4, 2, 1),
            *block(channel_out))
                
    def forward(self, input):
        return self.layers(input) + self.project(input)

class SelfAttention(torch.nn.Module):

    def __init__(self, channel, head=8, dim=32):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(channel),
            torch.nn.Linear(channel, 3 * head * dim, bias=False),
            torch.nn.Unflatten(-1, (3, head, dim)))
        self.project = torch.nn.Linear(head * dim, channel)
        
    def forward(self, input):
        _, _, height, width = input.shape
        query, key, value = self.layers(input.flatten(2).mT).permute(0, 3, 2, 1, 4).unbind(2)
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        output = self.project(output.permute(0, 2, 1, 3).flatten(2)).mT.unflatten(-1, (height, width))
        return output + input

def ReflowUp(channel_in, channel_out):
    return torch.nn.Sequential(
        ResidualUp(channel_in, channel_out),
        SelfAttention(channel_out),
        Residual(channel_out),
        Residual(channel_out))

def ReflowDown(channel_in, channel_out):
    return torch.nn.Sequential(
        Residual(channel_in),
        Residual(channel_in),
        SelfAttention(channel_in),
        ResidualDown(channel_in, channel_out))

def AutoencoderUp(channel_in, channel_out):
    return torch.nn.Sequential(
        ResidualUp(channel_in, channel_out),
        Residual(channel_out),
        Residual(channel_out),
        Residual(channel_out),
        Residual(channel_out))

def AutoencoderDown(channel_in, channel_out):
    return torch.nn.Sequential(
        Residual(channel_in),
        Residual(channel_in),
        ResidualDown(channel_in, channel_out))
