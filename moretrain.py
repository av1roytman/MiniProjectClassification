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


from transformers import ViTForImageClassification, ViTConfig

class CustomViTModel(nn.Module):
    def __init__(self, num_classes: int):
        super(CustomViTModel, self).__init__()
        # Load the configuration and modify it for the number of classes
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes)

        # Ensure that we are explicitly getting a ViTForImageClassification instance
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', config=config, ignore_mismatched_sizes=True)
        if not isinstance(model, ViTForImageClassification):
            raise TypeError("The loaded model is not a ViTForImageClassification instance.")

        self.vit = model

    def forward(self, x):
        outputs = self.vit(x)
        return outputs.logits  # Get the logits from the model outputs


class FacesDataset(Dataset):
    def __init__(self, images_csv_file, categories_csv_file, root_dir, transform=None):
        self.image_annotations = pd.read_csv(images_csv_file)
        self.categories = pd.read_csv(categories_csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, index):
        # Extract the base file name (e.g., '0_face_0.jpg') and the original image index (e.g., '0' from '0_face_0.jpg')
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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = FacesDataset(images_csv_file='train.csv', categories_csv_file='category.csv', root_dir='./train-cropped1', transform=transform)

    # Determine the lengths of the training and validation sets
    train_len = int(0.9 * len(dataset))  # 80% of the dataset for training
    val_len = len(dataset) - train_len  # 20% of the dataset for validation

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers = 8)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers = 8)

    # Initialize your model, criterion, and optimizer
    # model = AdvancedCNN()

    num_classes = 100
    # model = VGG16(num_classes=num_classes)

    model = CustomViTModel(num_classes=num_classes)

    # Load the model from .pth file
    model.load_state_dict(torch.load('customViT-model.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if GPU is available
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 25  # or however many you'd like

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

        # Validation phase
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Validation accuracy: {100 * correct / total:.2f}%')

    # Save the model
    torch.save(model.state_dict(), 'customViT-model.pth')


if __name__ == '__main__':
    main()