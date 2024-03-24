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


class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(AdvancedCNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.25)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.25)

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_dropout1 = nn.Dropout(0.5)

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc_dropout2 = nn.Dropout(0.5)

        # Output Layer
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.fc_dropout1(x)
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.fc_dropout2(x)

        # Output layer
        x = self.fc3(x)

        return x


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        # Define the configuration for VGG16: 'M' denotes MaxPool layer
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.features = self._make_layers(cfg)

        # Classifier part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


from transformers import ViTForImageClassification

class CustomViTModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit: ViTForImageClassification = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.vit.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x.logits


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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if GPU is available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 15  # or however many you'd like

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
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
