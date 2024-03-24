from mtcnn import MTCNN
from PIL import Image
import numpy as np
import os

def crop_faces_mtcnn(source_directory, target_directory):
    detector = MTCNN()

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for filename in os.listdir(source_directory):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(source_directory, filename)
            image = Image.open(image_path)
            
            # Convert black-and-white images to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_array = np.array(image)

            faces = detector.detect_faces(image_array)
            
            for i, face in enumerate(faces):
                x, y, width, height = face['box']
                cropped_image = image.crop((x, y, x + width, y + height))
                face_filename = os.path.join(target_directory, f"{filename}")
                cropped_image.save(face_filename)

# Use the function
source_dir = './train_small'
target_dir = './train_small-cropped'
crop_faces_mtcnn(source_dir, target_dir)
