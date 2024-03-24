from facenet_pytorch import MTCNN
from PIL import Image
import torch
import os

def crop_faces_torch(source_directory, target_directory, no_faces_directory=None, use_cuda=True, min_size=(50, 50)):
    device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    if no_faces_directory and not os.path.exists(no_faces_directory):
        os.makedirs(no_faces_directory)

    counter = 0

    for filename in os.listdir(source_directory):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(source_directory, filename)
            image = Image.open(image_path).convert('RGB')
            
            boxes, _ = mtcnn.detect(image)
            
            if boxes is None:
                if no_faces_directory:
                    image.save(os.path.join(no_faces_directory, filename))
                continue

            for i, box in enumerate(boxes):
                box = box.astype(int)
                width, height = box[2] - box[0], box[3] - box[1]

                if width >= min_size[0] and height >= min_size[1]:
                    cropped_image = image.crop(box)
                    base_filename, file_extension = os.path.splitext(filename)
                    face_filename = os.path.join(target_directory, f"{base_filename}_face_{i}{file_extension}")
                    cropped_image.save(face_filename)

        counter += 1
        if counter % 100 == 0:
            print(f'Processed {counter} images')


# Use the function
source_dir = './train'
target_dir = './train-cropped1'
no_faces_dir = './train-nofaces'  # Optional
crop_faces_torch(source_dir, target_dir, no_faces_dir)
