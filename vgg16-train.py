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
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    # Ensure the layer is an instance of nn.Linear before accessing in_features
    if isinstance(model.classifier, nn.Sequential):
        # Now we are sure classifier is Sequential, so indexing should be safe
        if isinstance(model.classifier[6], nn.Linear):
            num_features = int(model.classifier[6].in_features)
            model.classifier[6] = nn.Linear(num_features, num_classes)
        else:
            print("The target layer is not a linear layer.")
    else:
        print("The classifier is not a sequential module.")

    # model = VGG16(num_classes=num_classes)

    # model = CustomViTModel(num_classes=num_classes)

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
    torch.save(model.state_dict(), 'VGG16-attempt#1.pth')


if __name__ == '__main__':
    main()
