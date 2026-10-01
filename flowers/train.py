import torch
import lightning
import torchvision
from model import Critic, init_conv, Autoencoder as Model
from dataset import Dataset

class Adversarial(lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = Model()
        self.critic = Critic()
        self.model.apply(init_conv(0.01))
        self.critic.apply(init_conv(0.01))
        self.weight = torch.nn.Buffer(torch.as_tensor(1.0))
        self.alpha = torch.nn.Buffer(torch.as_tensor(1e-3))
        self.automatic_optimization = False

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(self.model.parameters(), 2e-4, [0.5, 0.9]),
            torch.optim.AdamW(self.critic.parameters(), 2e-4, [0.5, 0.9])]

    def training_step(self, batch, index):
        model_opt, critic_opt = self.optimizers()
        if index % 2 == 0:
            self.train_critic(critic_opt, batch)
        else:
            self.train_model(model_opt, batch)

    def train_critic(self, opt, real:torch.Tensor):
        self.toggle_optimizer(opt)
        with torch.no_grad():
            fake, _ = self.model(real)
        real_score, _ = self.critic(real)
        fake_score, _ = self.critic(fake)
        loss_real = (1 - real_score).clamp(min=0).mean()
        loss_fake = (1 + fake_score).clamp(min=0).mean()
        self.log("real", loss_real.item(), True)
        self.log("fake", loss_fake.item(), True)
        opt.zero_grad()
        self.manual_backward(loss_real + loss_fake)
        opt.step()
        self.untoggle_optimizer(opt)

    def loss_features(self, fakes, reals):
        loss = 0
        for fake, real in zip(fakes, reals):
            loss = loss + (fake - real).square().mean()
        return 0.5 * loss / len(fakes)

    def train_model(self, opt, real):
        self.toggle_optimizer(opt)
        fake, latent = self.model(real)
        with torch.no_grad():
            _, real_features = self.critic(real)
        fake_score, fake_features = self.critic(fake)
        loss_fake = (1 - fake_score).clamp(min=0).mean()
        loss_features = self.loss_features(fake_features, real_features)
        loss_latent = (latent.abs() - 1).clamp(min=0).mean()

        mean = fake_score.mean()
        grad_score = torch.autograd.grad(
            outputs=mean, 
            inputs=self.model.decoder[-1].weight, 
            grad_outputs=torch.ones_like(mean), 
            retain_graph=True, 
            create_graph=False, 
            only_inputs=True)[0].norm()
        grad_features = torch.autograd.grad(
            outputs=loss_features, 
            inputs=self.model.decoder[-1].weight, 
            grad_outputs=torch.ones_like(loss_features),
            retain_graph=True, 
            create_graph=False, 
            only_inputs=True)[0].norm()
        self.weight = self.weight * (1 - self.alpha) + (grad_features / (grad_score + 1e-8)) * (self.alpha)

        self.log("score", loss_fake.item(), True)
        self.log("features", loss_features.item(), True)
        self.log("latent", loss_latent.item(), True)
        self.log("grad/score", grad_score.item(), False)
        self.log("grad/feature", grad_features.item(), False)
        self.log("grad/weight", self.weight.item(), False)
        opt.zero_grad()
        self.manual_backward(loss_fake * self.weight + loss_features + loss_latent)
        opt.step()
        self.untoggle_optimizer(opt)

    def validation_step(self, batch, _):
        output, _ = self.model(batch)
        image = torch.stack((output, batch), 1)
        image = torchvision.utils.make_grid(image.flatten(0, 1), 4, 0)
        image = (image * 255).clamp(0, 255).to(torch.uint8).cpu()
        torchvision.io.write_jpeg(image, f"val_{self.global_step}.jpg")

if __name__ == "__main__":
    PATH = r"E:\Hilmi\Documents\Projects\datasets\flowers"
    train_data = torch.utils.data.DataLoader(
        Dataset(PATH),
        batch_size=8,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True)
    val_data = torch.utils.data.DataLoader(
        Dataset(PATH, True, 8),
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
        Adversarial(),
        train_data,
        val_data)