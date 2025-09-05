# Sign-Language-Detection-Using-Rnb
Indian Sign Language (ISL) Gesture Recognition

This project focuses on Indian Sign Language (ISL) recognition using deep learning. The model is built with Recurrent Neural Networks (RNN/LSTM) to learn sequences of hand gestures (frames) and predict sentences in ISL.

📌 Project Overview

Dataset

given as zip in the repo

Contains:

Frames for sentence-level detection

Frames for word-level detection

Videos for sentence-level detection



Goal

Train an RNN-based model that can map sequences of gesture frames into sentences.

Future extension: Build a system that can translate ISL gestures into text/speech for real-time communication.

Why RNN?

RNNs (LSTM/GRU) are effective for temporal data like gesture sequences.

Unlike static classification, RNNs can learn sentence structure from gesture sequences.

⚙️ Tech Stack

Language: Python

Frameworks: PyTorch / NumPy / OpenCV

Model: RNN (LSTM-based)

Dataset: ISL-CSLTR (frames & videos)

🚀 Training Pipeline

Preprocessing

Extract frames / keypoints from gesture videos.

Normalize and convert into .npy sequences.

Model

Input: Gesture sequence (frames or keypoints)

Layers: LSTM → Fully Connected → Softmax

Output: Sentence prediction

Training

Optimizer: Adam

Loss Function: CrossEntropyLoss

Epochs: 100+

Batch Size: 100

Saving Model

Model weights saved as .pt or .h5 for reuse.

🖼️ Example

Input: Sequence of gesture frames

Output: "How are you?" (predicted sentence in ISL)

📌 Future Work

Improve accuracy using transformer models.

Add real-time webcam inference with MediaPipe + OpenCV.

Expand to ISL-to-speech translation system.
