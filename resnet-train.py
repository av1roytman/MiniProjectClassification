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
from torchvision.models import ResNet18_Weights


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
        original_image_index = file_name.split('_')[0]
        
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FacesDataset(images_csv_file='train.csv', categories_csv_file='category.csv', root_dir='./train-good3', transform=transform)

    # Create data loaders
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers = 8)
    num_classes = 100

    # Initialize your model, criterion, and optimizer

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # For normal training:
    for param in model.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)  # Assuming 100 classes (celebrities)

    # Added for further training
    model.load_state_dict(torch.load('resnet18-attempt2.pth'))

    # Check if GPU is available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 30  # or however many you'd like

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward pass + optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct_predictions / len(dataset)

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'resnet18-attempt3.pth')


if __name__ == '__main__':
    main()
