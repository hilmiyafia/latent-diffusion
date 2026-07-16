import torch
import lightning
import torchvision
from model_reflow import Model
from dataset import LatentDataset
from train_autoencoder import Trainer2 as Autoencoder

class Trainer(lightning.LightningModule):

    def __init__(self, decode=None, time_step=100):
        super().__init__()
        self.model = Model()
        self.ema = Model()
        self.decode = decode
        self.save_hyperparameters("time_step")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), 2e-4)
    
    def training_step(self, batch, _):
        with torch.no_grad():
            noise = self.make_noise(batch)
            time = torch.randn(len(batch), 1, 1, 1, device=batch.device).sigmoid()
            mixed = batch * time + noise * (1 - time)
            velocity = batch - noise
        output = self.model(mixed, time)
        loss = (velocity - output).square().mean()
        self.log("loss", loss, True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            beta = 0.999
            params = dict(self.ema.named_parameters())
            for name, param in self.model.named_parameters():
                if name in params:
                    params[name].lerp_(param, 1 - beta)
            buffers = dict(self.ema.named_buffers())
            for name, buffer in self.model.named_buffers():
                if name in buffers:
                    buffers[name].copy_(buffer)
    
    def validation_step(self, batch, _):
        shift = 3
        generator = torch.Generator(batch.device)
        generator.manual_seed(0)
        buffer = self.make_noise(batch, generator)
        times = torch.linspace(0.0, 1.0, self.hparams["time_step"])
        times = (shift * times) / (1.0 + (shift - 1.0) * times)
        speed = times[1:] - times[:-1]
        for i in range(self.hparams["time_step"] - 1):
            time = torch.ones(len(batch), 1, 1, 1).to(batch.device) * times[i]
            output = self.ema(buffer, time)
            buffer = buffer + output * speed[i]
        result = (self.decode(buffer) + 0.9) / 1.8
        result = (255 * result).clamp(0, 255).to(torch.uint8)
        result = torchvision.utils.make_grid(result, 4, 0)
        number = str(self.global_step).rjust(5, "0")
        torchvision.io.write_jpeg(result.cpu().detach(), f"Validation {number}.jpg")

    def make_noise(self, reference, generator=None):
        device = reference.device
        b, c, h, w = reference.shape
        noise = (
            torch.randn(b, c, h, w, generator=generator, device=device) 
            + torch.randn(b, c, 1, 1, generator=generator, device=device) * 0.01)
        return noise
    
if __name__ == "__main__":
    EPOCHS = 400
    BATCH_SIZE = 8
    TIME_STEP = 100
    LATENT_PATH = "latents"

    loader_train = torch.utils.data.DataLoader(
        LatentDataset(LATENT_PATH), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        drop_last=True, 
        persistent_workers=True)
    loader_val = torch.utils.data.DataLoader(
        LatentDataset(LATENT_PATH, BATCH_SIZE), 
        batch_size=BATCH_SIZE)

    autoencoder = Autoencoder.load_from_checkpoint(r"lightning_logs\version_1\checkpoints\epoch=399-step=100000.ckpt").model

    trainer = lightning.Trainer(max_epochs=EPOCHS)
    trainer.fit(
        Trainer(autoencoder.decode, TIME_STEP),
        loader_train,
        loader_val)
