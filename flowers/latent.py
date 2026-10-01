import torch
import lightning
import torchvision
from train import Adversarial
from model import LatentGenerator, LatentCritic, init_conv
from dataset import Latent as Dataset

class LatentAdversarial(lightning.LightningModule):

    def __init__(self, decode=None):
        super().__init__()
        self.model = LatentGenerator()
        self.critic = LatentCritic()
        self.model.apply(init_conv(0.01))
        self.critic.apply(init_conv(0.01))
        self.decode = decode
        self.automatic_optimization = False

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(self.model.parameters(), 6e-5, [0.5, 0.9]),
            torch.optim.AdamW(self.critic.parameters(), 6e-5, [0.5, 0.9])]

    def training_step(self, batch, index):
        model_opt, critic_opt = self.optimizers()
        if index % 2 == 0:
            self.train_critic(critic_opt, batch)
        else:
            self.train_model(model_opt, batch)

    def train_critic(self, opt, real):
        self.toggle_optimizer(opt)
        with torch.no_grad():
            noise = torch.randn(real.shape[0], 100, device=real.device)
            fake = self.model(noise)
        loss_real = (1 - self.critic(real)[0]).square().mean()
        loss_fake = (1 + self.critic(fake)[0]).square().mean()
        self.log("real", loss_real.item(), True)
        self.log("fake", loss_fake.item(), True)
        opt.zero_grad()
        self.manual_backward(loss_real + loss_fake)
        opt.step()
        self.untoggle_optimizer(opt)

    def train_model(self, opt, real):
        self.toggle_optimizer(opt)
        noise = torch.randn(real.shape[0], 100, device=real.device)
        fake = self.model(noise)
        score, feature = self.critic(fake)
        decoded = self.model.decoder(feature)
        loss = (1 - score).square().mean()
        loss_info = (decoded - noise[:, :50]).square().mean()
        self.log("model", loss.item(), True)
        self.log("info", loss_info.item(), True)
        opt.zero_grad()
        self.manual_backward(loss + 10 * loss_info)
        opt.step()
        self.untoggle_optimizer(opt)

    def validation_step(self, batch, _):
        generator = torch.Generator(batch.device)
        generator.manual_seed(0)
        noise = torch.randn(batch.shape[0], 100, device=batch.device, generator=generator)
        buffer = self.decode(self.model(noise))
        image = torchvision.utils.make_grid(buffer, 4, 0)
        image = (image * 255).clamp(0, 255).to(torch.uint8).cpu()
        torchvision.io.write_jpeg(image, f"val_{self.global_step}.jpg")

if __name__ == "__main__":
    PATH = r"latents"
    AUTOENCODER = r"autoencoder\lightning_logs\version_0\checkpoints\epoch=99-step=102300.ckpt"
    autoencoder = Adversarial.load_from_checkpoint(AUTOENCODER).model.cuda()
    def decode(input):
        return autoencoder.decoder(input)
    train_data = torch.utils.data.DataLoader(
        Dataset(PATH),
        batch_size=8,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True)
    val_data = torch.utils.data.DataLoader(
        Dataset(PATH, 8),
        batch_size=8,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        drop_last=True)
    trainer = lightning.Trainer(
        max_epochs=100, 
        val_check_interval=100, 
        check_val_every_n_epoch=None)
    trainer.fit(
        LatentAdversarial(decode),
        train_data,
        val_data)