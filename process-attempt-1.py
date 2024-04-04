from facenet_pytorch import MTCNN
from PIL import Image
import torch
import os

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

# Specify the directory containing images
old = './train'
new = './train-good3'

if not os.path.exists(new):
    os.makedirs(new)

# Iterate over all files in the directory
count = 0

for filename in os.listdir(old):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check for common image file extensions
        file_num = filename.split('.')[0]
        image_path = os.path.join(old, filename)
        image = Image.open(image_path)

        # Convert to RGB
        image = image.convert('RGB')

        # Detect faces
        boxes, _ = mtcnn.detect(image)

        # Initialize a variable to count saved faces
        saved_faces = 0

        # Crop faces and save if they meet the size requirement
        if boxes is not None:
            for i, (x, y, x2, y2) in enumerate(boxes):
                if (x2 - x) > 30 and (y2 - y) > 30:  # Check if the face is at least 30x30 pixels
                    face = image.crop((x, y, x2, y2))
                    face.save(os.path.join(new, f'{file_num}_face_{i}.png'))
                    saved_faces += 1

        # If no faces were saved, resave the original image
        if saved_faces == 0:
            image.save(os.path.join(new, f'{file_num}_face_{saved_faces}.png'))

        count += 1
        if count % 100 == 0:
            print(f'Processed {count} images')

