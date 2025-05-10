# Your code here

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import time

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Set the paths to your train and test directories
train_dir = "train"  # Assuming the train folder is in the current directory
test_dir = "test"    # Assuming the test folder is in the current directory


# Define the transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# Loading the train and test datasets
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model
class FoodCNN(nn.Module):
    def __init__(self):
        super(FoodCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 91)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Assuming your model is called 'FoodCNN' and you already have it defined

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the model

model = FoodCNN().to(device)  # No need for 'debug=True' if it's not needed

# Weight Initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

StepLR = torch.optim.lr_scheduler.StepLR

# Learning Rate Scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Data Loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"Starting Epoch {epoch + 1}/{num_epochs}...")
    
    epoch_start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Running Accuracy: {100 * correct / total:.2f}%")

    # Step the scheduler
    scheduler.step()  # Update learning rate
    
    # Print Epoch Statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")
    
    # Evaluate the model on the test set after each epoch
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.2f}%\n")

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch {epoch + 1} Duration: {epoch_duration:.2f} seconds\n")
# Save the model
torch.save(model.state_dict(), "food_cnn_model.pth")
# Load the model
model.load_state_dict(torch.load("food_cnn_model.pth"))
# Set the model to evaluation mode
model.eval()
# Test the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
# Save the model
torch.save(model.state_dict(), 'model_final.pth')
