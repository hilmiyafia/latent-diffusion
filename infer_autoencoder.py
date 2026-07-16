import os
import tqdm
import torch
import torchvision
from train_autoencoder import Trainer2

if __name__ == "__main__":
    BATCH_SIZE = 8
    PATH = "../datasets/celeb"
    model = Trainer2.load_from_checkpoint(r"lightning_logs\version_1\checkpoints\epoch=399-step=100000.ckpt").model
    os.makedirs("latents", exist_ok=True)
    with torch.no_grad():
        for file in tqdm.tqdm(os.listdir(PATH)):
            data = torchvision.io.read_image(f"{PATH}/{file}").cuda() * 1.8 / 255 - 0.9
            latent = model.encode(data[None])[0].cpu().detach()
            torch.save(torch.clone(latent), f"latents/{file}.pt")
