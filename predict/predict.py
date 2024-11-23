import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

# Define the modified AlexNet model
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Adjusted for 32x32 input
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Adjusted input size for the linear classifier
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)  # Assuming two max pool layers reducing size to 4x4

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Define transformations for input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize input image to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensors
])

# Load the trained model
model = AlexNet()
model.load_state_dict(torch.load('../train/alexnet_cifar10.pth'))
model.eval()  # Set model to evaluation mode

# Function to predict class label for a single image
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image_tensor)
        _, predicted = output.max(1)
        return predicted.item()

# Example usage
image_path = '../images/image_2.png'  # Path to your image
predicted_class = predict_image(image_path)
print(f'Predicted class: {predicted_class}')
