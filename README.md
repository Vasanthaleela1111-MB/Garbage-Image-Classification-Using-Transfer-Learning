# ♻️ Garbage Image Classification Using Transfer Learning

## Overview

Garbage Image Classification Using Transfer Learning is a deep learning-based computer vision project that automatically classifies waste images into six categories: Cardboard, Glass, Metal, Paper, Plastic, and Trash.

The project utilizes multiple pretrained Convolutional Neural Network (CNN) architectures and transfer learning techniques to improve classification accuracy while reducing training time. A comparative analysis of five state-of-the-art deep learning models was conducted to identify the best-performing architecture for waste classification.

---

## Problem Statement

Efficient waste segregation is essential for improving recycling processes and reducing environmental pollution. Manual waste sorting is labor-intensive, time-consuming, and often inaccurate.

This project aims to develop an automated image classification system capable of identifying different types of waste materials using deep learning and computer vision techniques.

---

## Dataset

**Dataset:** Garbage Classification Dataset

### Waste Categories

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

---

## Data Preprocessing

The following preprocessing techniques were applied:

### Image Cleaning
- Blur Detection and Removal
- Noise Detection and Removal
- Invalid Image Filtering

### Image Processing
- Image Resizing (224 × 224)
- Pixel Normalization

### Data Augmentation
- Rotation
- Horizontal Flip
- Vertical Flip
- Zoom Transformation
- Width Shift
- Height Shift
- Shear Transformation
- Brightness Adjustment

---

## Exploratory Data Analysis

The dataset was analyzed to understand:

- Class Distribution
- Dataset Balance
- Sample Images from Each Category
- Image Quality Assessment

---

## Transfer Learning Models

The following pretrained models were evaluated:

| Model | Purpose |
|---------|---------|
| MobileNetV2 | Lightweight CNN for efficient classification |
| MobileNetV3 | Improved mobile-friendly architecture |
| ResNet50 | Deep residual network |
| EfficientNetB0 | Scalable and efficient CNN architecture |
| ShuffleNetV2 | Lightweight architecture optimized for speed |

### Training Strategy

- Pretrained ImageNet Weights
- Frozen Feature Extraction Layers
- Custom Classification Head
- Adam Optimizer
- CrossEntropy Loss Function
- GPU Training using PyTorch

---

## Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The best-performing model was selected based on overall classification performance and balanced evaluation metrics.

---

## Technologies Used

### Programming Language
- Python

### Deep Learning Framework
- PyTorch

### Computer Vision
- OpenCV

### Data Processing
- NumPy
- Pandas

### Data Visualization
- Matplotlib
- Seaborn

### Deployment
- Streamlit

---

## Project Structure

```text
Garbage Classification/
│
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
├── trash/
├── Noisy_Images/
│
├── Garbage_Image_Classification.ipynb
├── garbage.py
│
└── README.md
```

---

## Results

- Successfully classified garbage images into six categories.
- Compared five transfer learning architectures.
- Achieved high classification accuracy using pretrained CNN models.
- Developed a Streamlit application for real-time image classification.

---

## Streamlit Application Features

- Upload Garbage Images
- Predict Waste Category
- Display Classification Results
- Interactive Project Dashboard
- Creator Information Section

---

## Applications

### Smart Recycling Systems
Automated waste sorting for recycling facilities.

### Municipal Waste Management
Reduction of manual waste segregation efforts.

### Environmental Sustainability
Improved recycling efficiency and waste monitoring.

### Educational Platforms
Awareness and learning tools for proper waste segregation.

---

## Future Enhancements

- Real-time webcam-based waste detection
- Multi-object waste classification
- Deployment on edge devices
- Integration with smart recycling bins
- Support for additional waste categories

---

## Author

### Vasantha Leela M

Bachelor of Engineering (Computer Science and Engineering)

Karpagam Academy of Higher Education

---

## License

This project is developed for academic, educational, and research purposes.
