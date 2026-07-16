import sys
import torch
from PIL import Image
from train_reflow import Trainer as Reflow
from train_autoencoder import Trainer2 as Autoencoder
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

def load_reflow(step=100, shift=3):
    reflow = Reflow.load_from_checkpoint(
        r"lightning_logs\version_2\checkpoints\epoch=399-step=100000.ckpt").ema
    autoencoder = Autoencoder.load_from_checkpoint(
        r"lightning_logs\version_1\checkpoints\epoch=399-step=100000.ckpt").model
    times = torch.linspace(0.0, 1.0, step)
    times = (shift * times) / (1.0 + (shift - 1.0) * times)
    speed = times[1:] - times[:-1]
    def generate():
        with torch.no_grad():
            output =  torch.randn(1, 8, 16, 16).cuda()
            for i in range(step - 1):
                time = torch.ones(1, 1, 1, 1).cuda() * times[i]
                output = output + reflow(output, time) * speed[i]
            output = (autoencoder.decode(output) + 0.9) / 1.8
            output = torch.permute(output[0], (1, 2, 0)).cpu().detach()
            output = (255 * output).clamp(0, 255).to(torch.uint8)
        return output.numpy()
    return generate

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rectified Flow")
        self.reflow = load_reflow()
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.image_label = QLabel()
        self.image_label.setFrameStyle(QLabel.Shape.StyledPanel | QLabel.Shadow.Sunken)
        self.image_label.setLineWidth(2)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.update_image)
        layout.addWidget(self.generate_button)
        self.update_image()

    def update_image(self):
        image = Image.fromarray(self.reflow())
        if image.mode == "RGB":
            format_type = QImage.Format.Format_RGB888 
        else:
            format_type = QImage.Format.Format_RGBA8888
        pixmap = QPixmap.fromImage(QImage(
            image.tobytes(), 
            image.width, 
            image.height, 
            image.width * len(image.getbands()), 
            format_type))
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
