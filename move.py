import os
import cv2
import dlib
import shutil

# Move detectable faces to new directories which we'll use for the main script

# Path to face detector
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Initialize detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def detect_faces(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    return len(dets) > 0

def move_faces_with_detection(source_dir, target_dir):
    for img_file in os.listdir(source_dir):
        img_path = os.path.join(source_dir, img_file)
        if detect_faces(img_path):
            shutil.move(img_path, target_dir)

target_male_faces_path = config.males
target_female_faces_path = config.females

# Ensure target directories exist
os.makedirs(target_male_faces_path, exist_ok=True)
os.makedirs(target_female_faces_path, exist_ok=True)

# Process images
move_faces_with_detection(config.male_initial, target_male_faces_path)
move_faces_with_detection(config.female_initial, target_female_faces_path)
