import os
import torch
import tqdm
from train import Adversarial
from dataset import Dataset

if __name__ == "__main__":
    os.makedirs("latents", exist_ok=True)
    dataset = Dataset(r"E:\Hilmi\Documents\Projects\datasets\flowers", val=True)
    model = Adversarial.load_from_checkpoint(r"autoencoder\lightning_logs\version_0\checkpoints\epoch=99-step=102300.ckpt").model

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataset)):
            latent = model.encoder(data[None].cuda()).cpu()[0]
            torch.save(latent, f"latents/image_{i}.pt")