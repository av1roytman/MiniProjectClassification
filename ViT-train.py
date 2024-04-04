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
import re


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
    def __init__(self, images_csv_file, categories_csv_file, root_dirs, transform=None):
        self.image_annotations = pd.read_csv(images_csv_file)
        self.categories = pd.read_csv(categories_csv_file)
        self.root_dirs = root_dirs  # This is now a list of directories
        self.transform = transform

    def __len__(self):
        total = 0
        for dir in self.root_dirs:
            total += len(os.listdir(dir))
        return total

    def __getitem__(self, index):
        # Extract number from 123.jpg
        root_dir = self.root_dirs[index % len(self.root_dirs)]
    
        # Get the list of files in the directory
        files = os.listdir(root_dir)
        
        # Choose the file based on the index
        file_name = files[index % len(files)]

        original_image_index = re.split('_|\.', file_name)[0]
        
        name = self.image_annotations.iloc[int(original_image_index)]['Category']
        label = self.categories.loc[self.categories['Category'] == name].index[0]

        # Load the image
        img_path = os.path.join(self.root_dirs[index % len(self.root_dirs)], str(file_name))
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

    dataset = FacesDataset(images_csv_file='train.csv', categories_csv_file='category.csv', root_dirs=['./train-cropped1', './train-good', './train-good2', './train-good3'], transform=transform)


    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers = 8)

    num_classes = 100

    model = CustomViTModel(num_classes=num_classes)

    # Load the model from .pth file
    model.load_state_dict(torch.load('customViT-model-attempt5.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if GPU is available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 15  # or however many you'd like

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
    torch.save(model.state_dict(), 'customViT-model-attempt7-pretrainfromattempt5.pth')


if __name__ == '__main__':
    main()