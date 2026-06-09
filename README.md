# retinal_dis_classification

A deep learning-based web application for classifying retinal fundus images into multiple disease categories using a fine-tuned ResNet50 model. Built with TensorFlow and Streamlit.

---

## 🚀 Overview

This project aims to assist in early detection of retinal diseases through automated image classification. It uses a pre-trained **ResNet50** model, fine-tuned on a labeled dataset containing 8 retinal disease categories. The frontend provides an intuitive interface for users to upload an image and receive instant predictions.

---

## 📁 Dataset

- Source: [Kaggle - Retinal OCT C8](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8)
- Classes: 'AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL'
- Total Images: ~24,000 across training, validation, and testing sets

---

## 🧠 Model

- Base Model: `ResNet50` (pre-trained on ImageNet)
- Input Size: `224x224`
- Classification Head:
  - GlobalAveragePooling2D
  - Dense(256, ReLU) + Dropout
  - Dense(NUM_CLASSES, softmax)

- Techniques Used:
  - Transfer Learning
  - Data Augmentation
  - Model Checkpointing

---

## 🔍 Results

- Achieved ~66% validation accuracy (can be improved with more tuning or a larger dataset)
- Visualized results with confusion matrix and classification report
  ![image](https://github.com/user-attachments/assets/a4a3f0f8-3df7-4227-b05f-7ec10c763533)
  ![image](https://github.com/user-attachments/assets/4b6ac505-a781-405d-9131-bc3f1f9b3501)


---

## 💻 Web Application

Built using **Streamlit**, this frontend allows users to:
- Upload a retinal fundus image (`.jpg`, `.jpeg`, `.png`)
- See the uploaded image and predicted disease side by side
- Get confidence score for predictions

  ![Screenshot 2025-06-12 152934](https://github.com/user-attachments/assets/edeeadec-898d-4320-bc0b-3993d05769c9)



