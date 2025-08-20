😷 FaceMaskNet – Deep Learning Based Face Mask Detection
📌 Project Overview

This project implements a deep learning-based Face Mask Detection system that identifies whether individuals are wearing a mask or not. The model leverages Convolutional Neural Networks (CNNs) for image classification and can be deployed in real-time applications such as:

🏥 Healthcare & Hospitals – Ensuring mask compliance for staff, patients, and visitors.

🛡️ Security & Surveillance – Monitoring public spaces for safety compliance.

🛒 Retail & Workplaces – Enhancing safety in shops, offices, and factories.

🚔 Law Enforcement – Preventing theft and enhancing public safety by detecting mask-wearing individuals.

🎯 Objectives

Build an accurate mask vs no-mask classifier using deep learning.

Support real-time detection using webcam/ CCTV streams.

Enable deployment in healthcare, security, and workplace safety applications.

🏗️ System Architecture

The workflow is as follows:

Input:

Images or live video streams.

Preprocessing:

Face detection using OpenCV Haar Cascades / Dlib / MTCNN.

Resize & normalize input images for CNN.

Model (CNN / Transfer Learning):

Trained on datasets of masked and unmasked faces.

Outputs classification: Mask (1) or No Mask (0).

Output:

Real-time bounding box around detected faces with label (Mask / No Mask).

📊 Dataset

Face Mask Detection Dataset (publicly available or custom scraped).

Includes labeled images of:

People with masks.

People without masks.

Preprocessing:

Resize images to 224×224 pixels.

Normalize pixel values to [0,1].

Split: 80% training, 10% validation, 10% testing.

⚙️ Technical Details
Model Architecture

Option 1: Custom CNN

Conv2D → MaxPooling → Conv2D → MaxPooling → Dense → Softmax.

Option 2: Transfer Learning

Pre-trained models (MobileNetV2, ResNet50, VGG16).

Fine-tuned for binary classification.

Loss & Optimizer

Loss Function: Binary Cross Entropy (BCE).

Optimizer: Adam (lr=0.0001).

Evaluation Metrics

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

🚀 Training Procedure

Load and preprocess dataset.

Apply data augmentation (rotation, zoom, flip).

Train CNN / fine-tuned transfer learning model.

Evaluate on test set.

Deploy real-time detection using OpenCV.

📈 Results & Evaluation

Achieved 95%+ accuracy on test dataset (depending on model used).

Robust in real-time detection from webcam streams.

Handles multiple faces in one frame.

💻 Tech Stack

Language: Python

Frameworks: TensorFlow / Keras / PyTorch

Libraries: OpenCV, NumPy, Matplotlib, scikit-learn

Deployment: Flask / FastAPI (for web apps), OpenCV (for real-time video).
