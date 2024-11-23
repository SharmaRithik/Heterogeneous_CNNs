import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os

def create_save_directory(prune_percentage):
    save_dir = f'network_layers_pruned_{int(prune_percentage * 100)}percent'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def save_layer_info(model, save_dir):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Get the layer name parts
            if 'features' in name:
                layer_num = name.split('.')[-1]
                base_name = f'features_{layer_num}'
            else:
                base_name = 'classifier'
            
            # Save weights
            weight = module.weight.data.cpu().numpy()
            weight_file = os.path.join(save_dir, f'{base_name}_weight.txt')
            np.savetxt(weight_file, weight.reshape(weight.shape[0], -1))
            
            # Save biases if they exist
            if module.bias is not None:
                bias = module.bias.data.cpu().numpy()
                bias_file = os.path.join(save_dir, f'{base_name}_bias.txt')
                np.savetxt(bias_file, bias)

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
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
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pretrained model
    pretrained_model_path = 'alexnet_cifar10.pth'
    pretrained_model = AlexNet().to(device)
    pretrained_model.load_state_dict(torch.load(pretrained_model_path))
    pretrained_model.eval()

    # Define pruning percentage
    prune_percentage = 0.99

    # Prune the pretrained model
    print(f"Pruning model with {prune_percentage*100}% sparsity...")
    for name, module in pretrained_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data.cpu().numpy()
            threshold = np.percentile(np.abs(weight), prune_percentage * 100)
            mask = np.abs(weight) >= threshold
            module.weight.data *= torch.tensor(mask, dtype=torch.float).to(device)

    # Initialize the model, loss function, and optimizer
    model = pretrained_model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training...")
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Create save directory and save layer information
    save_dir = create_save_directory(prune_percentage)
    save_layer_info(model, save_dir)
    
    # Save the trained model
    model_save_path = os.path.join(save_dir, f'pruned_alexnet_cifar10_{prune_percentage}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model and layer information saved in directory: {save_dir}")

    # Evaluate the model on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main()
