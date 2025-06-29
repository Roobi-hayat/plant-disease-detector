AI-Powered Plant Disease Detection System

This system diagnoses plant diseases from leaf images using deep learning technology.
Technology Used:
•	TensorFlow for building the deep learning model
•	Streamlit for creating a user-friendly web interface

Plant Support:
•	Provides instant diagnosis with treatment and prevention suggestions
•	Supports 9 common plant types:
  Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato 
  
Dataset & Preprocessing
Dataset:
•	Source: PlantVillage dataset (available on Kaggle)
•	Contains 40,000 images
•	Covers 9 plant types and 24 disease classes
Preprocessing Steps:
•	Rescaling images to normalize pixel values (1/255)
•	Data augmentation using ImageDataGenerator
•	20% of data used for validation

Algorithm & Implementation:
Transfer Learning:
•	Used MobileNetV2 (pretrained on ImageNet) for feature extraction.
•	MobileNetV2 breaks down images into 1,280 distinct features, such as:
o	Texture variations (e.g., vein patterns)
o	Color gradients (e.g., chlorosis)
o	Shape abnormalities (e.g., necrotic areas)
Performance:
•	Achieved 96.4% validation accuracy

Backend:
Uses SQLite database to store treatment and prevention info

Group Members:

Roobi Hayat    4212/F21 "B"

Inza Shahid    4194/F21 "B"

Iqra Saleem    4172/F21 "B"
