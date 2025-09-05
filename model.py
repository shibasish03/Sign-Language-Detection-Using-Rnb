import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import os
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ===== CONFIG =====
MODEL_PATH = "rnn_data.pt"
DATASET_PATH = "npy_data"  # To load class names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.sidebar.title(" RNN Sign Language Detection")
st.sidebar.write(f" Using device: {device}")

# ===== LOAD CLASS NAMES =====
class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
NUM_CLASSES = len(class_names)

# ===== RNN MODEL CLASS (must match training) =====
class KeypointRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(KeypointRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# ===== LOAD MODEL =====
model = KeypointRNN(input_size=3, hidden_size=128, num_layers=2, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===== MEDIAPIPE SETUP =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils


# ===== STREAMLIT WEBRTC VIDEO PIPELINE =====
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.prev_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        predicted_class = "No Hand"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract keypoints
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.append([lm.x, lm.y, lm.z])

                keypoints = np.array(keypoints, dtype=np.float32)  # (21, 3)
                keypoints_tensor = torch.tensor(keypoints).unsqueeze(0).to(device)  # (1, 21, 3)

                # Inference
                with torch.no_grad():
                    output = model(keypoints_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)
                    predicted_class = class_names[pred.item()]
                    confidence = conf.item()

                # Draw label
                h, w, _ = img.shape
                cv2.putText(img, f'{predicted_class} ({confidence:.2f})',
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img


st.title(" Real-Time Sign Language Detection (RNN + MediaPipe)")
st.write("Press **Start** below to activate your webcam and detect hand signs in real-time.")

webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
