import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

print("Hyperparameters")

num_classes = 91
learning_rate = 0.001
num_epochs = 20
batch_size = 64

print(f"Hyperparameters: num_classes={num_classes}, learning_rate={learning_rate}, num_epochs={num_epochs}, batch_size={batch_size}")

print("Code block 1")

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to have mean=0.5 and std=0.5
   
])

# Load datasets
train_dataset = datasets.ImageFolder(root='train', transform=transform)
test_dataset = datasets.ImageFolder(root='test', transform=transform)

# Wrap in DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Optional: check class names
print(train_dataset.classes)

print("Defining the model")

class ConvolutionalNN(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionalNN, self).__init__()
        # Implement the CNN with the following layers --- SOLUTION
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_layer6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(512 * 16 * 16, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.5)


    # Progresses data across layers
    def forward(self, x):
       
        # Implement the forward pass --- SOLUTION
        
        # Convolutional layers with ReLU and MaxPooling
        x = self.relu1(self.conv_layer1(x))
        x = self.relu2(self.conv_layer2(x))
        x = self.max_pool1(x)
        x = self.dropout(x)

        x = self.relu1(self.conv_layer3(x))
        x = self.relu2(self.conv_layer4(x))
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.relu1(self.conv_layer5(x))
        x = self.relu2(self.conv_layer6(x))
        x = self.max_pool3(x)
        x = self.dropout3(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU and Dropout
        x = self.relu3(self.fc1(x))
        x = self.dropout4(x)
        x = self.relu3(self.fc2(x))
        x = self.dropout5(x)
        out = self.fc3(x)
        
        return out
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Code block 4, training using device: {device}')

# Move model to the device  
model = ConvolutionalNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # Move images to the correct device
        labels = labels.to(device)  # Move labels to the correct device

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model.ckpt')
# Load the model checkpoint
model.load_state_dict(torch.load('model.ckpt'))

# Test the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0   
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)  # Move images to the correct device
        labels = labels.to(device)  # Move labels to the correct device
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'model_final.pth')
