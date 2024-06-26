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

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Assuming you have a PyTorch model
    model = CustomViTModel(num_classes=100)

    model = model.to(device)

    # Load the model from .pth file
    model.load_state_dict(torch.load('customViT-model-attempt7-pretrainfromattempt5.pth'))

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
            inputs = inputs.to(device)
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
    df.to_csv('predictions-pretrainfrom5.csv', index=False)

if __name__ == '__main__':
    main()