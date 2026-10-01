import torch
import lightning
import torchvision
from model_autoencoder import Model, Critic
from dataset import ImageDataset

class Trainer1(lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = Model()
 
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), 2e-4)
           
    def training_step(self, batch, _):
        output = self.model(batch)
        loss = (output - batch).abs().mean()
        self.log("loss", loss, True)
        return loss
        
    def validation_step(self, batch, _):
        output = (self.model(batch) + 0.9) / 1.8
        output = torchvision.utils.make_grid(output, 4, 0)
        output = (output * 255).clamp(0, 255).to(torch.uint8).detach().cpu()
        number = str(self.current_epoch).rjust(4, "0")
        torchvision.io.write_jpeg(output, f"Validation {number}.jpg")

class Trainer2(lightning.LightningModule):

    def __init__(self, model=None):
        super().__init__()
        self.model = Model() if model is None else model
        self.critic = Critic()
        self.automatic_optimization = False
 
    def configure_optimizers(self):
        return [
            torch.optim.AdamW(self.model.parameters(), 2e-4, [0.5, 0.9]),
            torch.optim.AdamW(self.critic.parameters(), 2e-4, [0.5, 0.9])]
           
    def training_step(self, batch, index):
        model_opt, critic_opt = self.optimizers()
        if index % 5 < 4 or self.current_epoch < 1:
            self.train_critic(critic_opt, batch)
        else:
            self.train_model(model_opt, batch)

    def train_critic(self, opt, real):
        self.toggle_optimizer(opt)
        opt.zero_grad()
        with torch.no_grad():
            fake = self.model(real)
        loss_fake = self.bce(self.critic(fake)[0], 0)
        loss_real = self.bce(self.critic(real)[0], 1)
        self.log("critic/fake", loss_fake, True)
        self.log("critic/real", loss_real, True)
        self.manual_backward(loss_fake + loss_real)
        opt.step()
        self.untoggle_optimizer(opt)

    def train_model(self, opt, real):
        self.toggle_optimizer(opt)
        opt.zero_grad()
        fake = self.model(real)
        with torch.no_grad():
            real_features = self.critic(real)[1]
        fake_score, fake_features = self.critic(fake)
        loss = (fake - real).abs().mean()
        loss_critic = self.bce(fake_score, 1)
        loss_features = (fake_features - real_features).abs().mean() * 10
        self.log("model/loss", loss, True)
        self.log("model/critic", loss_critic, True)
        self.log("model/features", loss_features, True)
        self.manual_backward(loss + loss_critic + loss_features)
        opt.step()
        self.untoggle_optimizer(opt)

    def bce(self, input, target):
        return torch.nn.functional.binary_cross_entropy_with_logits(input, torch.ones_like(input) * target)
        
    def validation_step(self, batch, _):
        output = (self.model(batch) + 0.9) / 1.8
        output = torchvision.utils.make_grid(output, 4, 0)
        output = (output * 255).clamp(0, 255).to(torch.uint8).detach().cpu()
        number = str(self.current_epoch).rjust(4, "0")
        torchvision.io.write_jpeg(output, f"Validation {number}.jpg")

if __name__ == "__main__":
    BATCH_SIZE = 8
    PATH = r"..\datasets\celeb"
    EPOCHS = [50, 400]
    PHASE = 1
    
    if PHASE == 0:
        model = Trainer1()
    elif PHASE == 1:
        model = Trainer2(
            Trainer1.load_from_checkpoint(
                r"lightning_logs\version_0\checkpoints\epoch=49-step=12500.ckpt").model)
    
    loader_train = torch.utils.data.DataLoader(
        ImageDataset(PATH, size=64), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        drop_last=True, 
        persistent_workers=True)
    loader_val = torch.utils.data.DataLoader(
        ImageDataset(PATH, BATCH_SIZE, False), 
        batch_size=BATCH_SIZE)

    trainer = lightning.Trainer(max_epochs=EPOCHS[PHASE])
    trainer.fit(
        model,
        loader_train,
        loader_val)
