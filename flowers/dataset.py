import os
import torch
import torchvision

class Latent(torch.utils.data.Dataset):

    def __init__(self, path, count=-1):
        self.files = [f"{path}/{file}" for file in os.listdir(path)]
        self.files.sort()
        if count > 0:
            self.files = self.files[:count]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return torch.load(self.files[index])

class Dataset(torch.utils.data.Dataset):

    def __init__(self, path, val=False, count=-1):
        self.files = [f"{path}/{file}" for file in os.listdir(path)]
        self.files.sort()
        if count > 0:
            self.files = self.files[:count]
        if val:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(256),
                torchvision.transforms.ConvertImageDtype(torch.float32)])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop((64, 64)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ConvertImageDtype(torch.float32)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = torchvision.io.read_image(self.files[index])
        return self.transform(image)

if __name__ == "__main__":
    dataset = Dataset("../flowers")
    image = dataset[0]
    print(image, image.shape)