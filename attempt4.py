print("Importing libraries...")

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

print("Loading the dataset...")

# Define paths
train_dir = 'train'
test_dir = 'test'

"""
# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 
"""

train_transform = transforms.Compose([
    transforms.Resize((144, 144)),  # Slightly larger to allow cropping
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# Load datasets
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
test_dataset = ImageFolder(root=test_dir, transform=test_transform)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

num_classes = len(train_dataset.classes)
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes[:10]}...")

print ("CNN implementation")

#  deeper architecture and batch normalization
class FoodCNN(nn.Module):
    def __init__(self, train_dataset):
        super(FoodCNN, self).__init__()
        
        num_classes = len(train_dataset.classes)
        #  32 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #  64 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #  128 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 256 filters
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #  512 filters 
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

model = FoodCNN(train_dataset)
print(model)

print("Training the model...")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

def evaluation(model, test_loader, criterion, desc="Evaluation"):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
        
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc):
            images, labels = images.to(device), labels.to(device)
                
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    test_acc = 100 * correct / total
    test_loss = test_loss / len(test_loader)

    return test_acc, test_loss

device = torch.device("cuda")
model = model.to(device)

def training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=10):

    best_accuracy=0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            # Backward and optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        test_acc, test_loss = evaluation(model, test_loader, criterion, desc="Testing")
        print(f"Testing - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
        # Update learning rate based on validation loss
        scheduler.step(test_loss)
    

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model with accuracy: {best_accuracy:.2f}%")
    
    
    return model, best_accuracy

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    #lr = 0.001
    optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
    #optimizer = optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)
    model, accuracy = training(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs = 100)

print("Calculating accuracy on the test set...")

def hyperparameters(num_epochs, best_accuracy, optimizer, batch_size, device, image_size):
    print("\nHyperparameters:")
    print(f"Architecture: Improved CNN with 5 convolutional blocks and batch normalization")
    print(f"Image Size: {image_size[0]}x{image_size[1]}")
    
    # Extract optimizer name and its params dynamically
    opt_name = optimizer.__class__.__name__
    print(f"Optimizer: {opt_name}")
    
    for param_group in optimizer.param_groups:
        print(f"Initial Learning Rate: {param_group['lr']}")
        print(f"Weight Decay: {param_group.get('weight_decay', 'N/A')}")
        break  # only need the first param_group for this
    
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Device: {device}")

model.load_state_dict(torch.load("best_model.pth"))
final_acc, final_loss = evaluation(model, test_loader, criterion, desc="Final")
#print(f"\nFinal Model Performance - Test Accuracy: {final_acc:.2f}%")
hyperparameters(num_epochs=100, best_accuracy=accuracy, optimizer=optimizer, batch_size=batch_size, device=device, image_size=(128, 128))

