import torch
import os
import pandas as pd
from torchvision.io import read_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FacesDataset(Dataset):
    def __init__(self, images_csv_file, categories_csv_file, root_dir, transform=None):
        self.image_annotations = pd.read_csv(images_csv_file)
        self.categories = pd.read_csv(categories_csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, index):
        # Extract number from 123.jpg
        file_name = os.listdir(self.root_dir)[index]
        original_image_index = file_name.split('.')[0]
        
        name = self.image_annotations.iloc[int(original_image_index)]['Category']
        label = self.categories.loc[self.categories['Category'] == name].index[0]


        # Load the image
        img_path = os.path.join(self.root_dir, str(file_name))
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)

        return image, label


def main():

    # Define transforms for data normalization and augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the same dimensions
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = FacesDataset(images_csv_file='train.csv', categories_csv_file='category.csv', root_dir='./train-good', transform=transform)

    # Create data loaders
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers = 8)
    num_classes = 100

    # Initialize your model, criterion, and optimizer

    model = VGG16(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if GPU is available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 50  # or however many you'd like

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the appropriate device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), 'VGG16-custom-attempt#1.pth')


if __name__ == '__main__':
    main()
