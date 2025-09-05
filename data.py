import os
import cv2
import numpy as np
import mediapipe as mp

# ===== CONFIG =====
INPUT_IMAGE_DIR = "senetence/sign_language"     # Folder containing class-wise image folders
OUTPUT_NPY_DIR = "npy_data"     # Where .npy files will be saved
MAX_HANDS = 2

# ===== MEDIAPIPE SETUP =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=MAX_HANDS)
mp_drawing = mp.solutions.drawing_utils

# ===== PROCESS IMAGES =====
for class_name in os.listdir(INPUT_IMAGE_DIR):
    class_path = os.path.join(INPUT_IMAGE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(OUTPUT_NPY_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for file_name in os.listdir(class_path):
        if not (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.png') or file_name.lower().endswith('jpeg')):
            continue

        img_path = os.path.join(class_path, file_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Use only the first hand
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])  # shape: (21, 3)
            npy_filename = os.path.splitext(file_name)[0] + ".npy"
            np.save(os.path.join(output_class_path, npy_filename), keypoints)
            print(f"✅ Saved: {npy_filename}")
        else:
            print(f"❌ No hand found in: {file_name}")
