import os
import torch
import torchvision
import torchvision.transforms.v2

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, path, count=-1, flip=True, multiplier=1, size=-1):
        self.files = [f"{path}/{file}" for file in os.listdir(path)]
        self.files.sort()
        if 0 < count < len(self.files):
            self.files = self.files[:count]
        self.multiplier = multiplier
        self.augment = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.ConvertImageDtype(),
            torchvision.transforms.v2.RandomHorizontalFlip() if flip 
            else torch.nn.Identity(),
            torchvision.transforms.v2.RandomCrop((size, size)) if size > 0
            else torch.nn.Identity()])

    def __len__(self):
        return int(len(self.files) * self.multiplier)
    
    def __getitem__(self, index):
        index = index % len(self.files)
        image = torchvision.io.read_image(self.files[index])
        return self.augment(image) * 1.8 - 0.9

class LatentDataset(torch.utils.data.Dataset):

    def __init__(self, path, count=-1, multiplier=1):
        self.files = [f"{path}/{file}" for file in os.listdir(path)]
        self.files.sort()
        if 0 < count < len(self.files):
            self.files = self.files[:count]
        self.multiplier = multiplier

    def __len__(self):
        return int(len(self.files) * self.multiplier)
    
    def __getitem__(self, index):
        index = index % len(self.files)
        return torch.load(self.files[index])
    