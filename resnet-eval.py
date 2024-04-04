import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
from PIL import Image
import os
from torch.utils.data import Dataset
from model import AdvancedCNN
from torchvision import models
from torchvision.models import ResNet18_Weights

class FacesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(self.root_dir), key=lambda x: int(os.path.splitext(x)[0]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Get the filename from the sorted list
        file_name = self.filenames[index]

        # Load the image
        img_path = os.path.join(self.root_dir, str(file_name))
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Return the image and the filename without the extension
        return image, os.path.splitext(file_name)[0]

def main():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the same dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # Assuming you have a Dataset object for your test data
    test_dataset = FacesDataset(root_dir='./test-cropped', transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # Assuming you have a PyTorch model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Load the model from .pth file

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)  # Assuming 100 classes (celebrities)

    model.load_state_dict(torch.load('resnet18-attempt4.pth'))

    # model = AdvancedCNN()
    # model.load_state_dict(torch.load('model.pth'))

    model.eval()

    # Assuming you have a list of class names corresponding to the output classes of your model
    class_names = pd.read_csv('category.csv')

    # Initialize a list to store the predictions
    predictions = []
    filenames = []

    # Iterate over the test data and generate predictions
    with torch.no_grad():
        for inputs, file_names in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.tolist())
            filenames.extend(file_names)

    # Convert the predictions to class names
    predictions = [class_names['Category'][pred] for pred in predictions]

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame({
        'Id': filenames,
        'Category': predictions
    })
    df.to_csv('predictions-resnet-2-sets.csv', index=False)

if __name__ == '__main__':
    main()