import cv2
import os
from PIL import Image
import numpy as np

# Path to the input directory containing images to process
input_dir = "./train_small/"
# Path to the output directory where cropped faces will be saved
output_dir = "./train_small-cropped/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(input_dir, filename)
        
        # Attempt to read the image using OpenCV
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("Not a valid image")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            # If OpenCV fails, try opening the image with PIL and convert it to an OpenCV-friendly format
            try:
                pil_image = Image.open(path)
                pil_image = pil_image.convert('RGB')  # Convert to RGB
                open_cv_image = np.array(pil_image)
                # Convert RGB to BGR for OpenCV
                image = open_cv_image[:, :, ::-1].copy()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                continue

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Crop and save each face found in the image
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            output_path = os.path.join(output_dir, f"cropped_{filename}")
            cv2.imwrite(output_path, face)

print("Face detection and cropping completed.")


# import cv2
# import os

# # Path to the input directory containing images to process
# input_dir = "./train/"
# # Path to the output directory where cropped faces will be saved
# output_dir = "./train-cropped/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Load the pre-trained Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Process each image in the input directory
# for filename in os.listdir(input_dir):
#     if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
#         # Read the image
#         path = os.path.join(input_dir, filename)
#         image = cv2.imread(path)
#         if image is None:
#             print(f"Failed to load image at {path}")
#             continue
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection

#         # Detect faces in the image
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         # Crop and save each face found in the image
#         for (x, y, w, h) in faces:
#             face = image[y:y+h, x:x+w]
#             output_path = os.path.join(output_dir, f"cropped_{filename}")
#             cv2.imwrite(output_path, face)

# print("Face detection and cropping completed.")
