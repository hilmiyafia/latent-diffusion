import torch
from blocks import ReflowDown, ReflowUp, Residual

class Model(torch.nn.Module):

    def __init__(self, time_dim=8):
        super().__init__()
        positions = 1000 * torch.pow(10000, -torch.arange(time_dim) / time_dim)[None, :, None, None]
        self.register_buffer("positions", positions)
        self.layer1 = torch.nn.Conv2d(8 + 2 * time_dim, 64, 1)
        self.layer2 = ReflowDown(64, 128)
        self.layer3 = ReflowDown(128, 256)
        self.layer4 = ReflowDown(256, 512)
        self.layer5 = torch.nn.Sequential(*[Residual(512) for _ in range(3)])
        self.layer6 = ReflowUp(1024, 256)
        self.layer7 = ReflowUp(512, 128)
        self.layer8 = ReflowUp(256, 64)
        self.layer9 = torch.nn.Conv2d(128, 8, 1)

    def forward(self, input, time):
        time = (time * self.positions).expand(*input.shape)
        buffer = torch.cat((input, time.sin(), time.cos()), 1)
        buffer1 = self.layer1(buffer)
        buffer2 = self.layer2(buffer1)
        buffer3 = self.layer3(buffer2)
        buffer4 = self.layer4(buffer3)
        buffer5 = self.layer5(buffer4)
        buffer6 = self.layer6(torch.cat((buffer5, buffer4), 1))
        buffer7 = self.layer7(torch.cat((buffer6, buffer3), 1))
        buffer8 = self.layer8(torch.cat((buffer7, buffer2), 1))
        buffer9 = self.layer9(torch.cat((buffer8, buffer1), 1))
        return buffer9
