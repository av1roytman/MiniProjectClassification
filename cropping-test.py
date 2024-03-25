from facenet_pytorch import MTCNN
from PIL import Image
import torch
import os

def crop_faces_torch(source_directory, target_directory, use_cuda=True, min_size=(50, 50)):
    device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    counter = 0

    for filename in os.listdir(source_directory):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(source_directory, filename)
            image = Image.open(image_path).convert('RGB')
            
            boxes, _ = mtcnn.detect(image)
            
            if boxes is not None:
                # Sort boxes by area (width * height)
                boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)

                # Only process the largest detected face
                box = boxes[0].astype(int)
                width, height = box[2] - box[0], box[3] - box[1]

                if width >= min_size[0] and height >= min_size[1]:
                    image = image.crop(box)

            # Save the cropped image or the original image if no faces were detected
            base_filename, file_extension = os.path.splitext(filename)
            image_filename = os.path.join(target_directory, f"{base_filename}{file_extension}")
            image.save(image_filename)

            counter += 1
            if counter % 100 == 0:
                print(f'Processed {counter} images')

# Use the function
source_dir = './train'
target_dir = './train-good'
crop_faces_torch(source_dir, target_dir)