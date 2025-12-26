# Real-vs-AI Image Detector

## Project Overview

This project implements a **Deep Learning-based Real-vs-AI Image Detector** capable of classifying whether an image is **real (photographed)** or **AI-generated**. The detector also outputs a **confidence score** for its predictions. This can be useful for verifying visual content authenticity in research, social media, and digital media verification.

**Demo:**  
The application includes a simple web interface where users can **upload images**, see the prediction label (Real / AI), confidence score, and a preview of the uploaded image.

---

## Dataset

**Dataset Used:** [200K Real vs AI Visuals by Muhammad Bilal](https://www.kaggle.com/datasets/muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal)  

- **Size:** 200,000 images  
- **Classes:**  
  - `Real` – genuine photographs  
  - `AI` – AI-generated visuals  
- **Format:** Images in JPEG/PNG format  
- **Usage:** The dataset was split into training, validation, and test sets to train a CNN-based classifier.

---

## Architecture & Approach

The model uses **EfficientNet-B0** pre-trained on ImageNet as the backbone for feature extraction. Key steps include:

1. **Data Preprocessing**  
   - Resize to `224x224` pixels  
   - Normalize with ImageNet statistics  
   - Data augmentation (optional): rotations, flips, color jitter  

2. **Model Architecture**  
   - Backbone: EfficientNet-B0 (pre-trained)  
   - Fully-connected layers for binary classification  
   - Output: probability scores for `Real` and `AI`  

3. **Training**  
   - Loss: Cross-Entropy Loss  
   - Optimizer: Adam  
   - Device: GPU if available  
   - Metrics: Accuracy, confidence scores
   - Optuna : for optimized batch_size, dropout rate, learning_rate

4. **Inference & API**  
   - FastAPI serves a `/predict` endpoint  
   - Users can upload images via the web interface  
   - Model returns **prediction label** and **confidence**  

---

## Tech Stack

- **Backend & API:** FastAPI  
- **Frontend UI:** Jinja2 templates (HTML + CSS)  
- **Deep Learning:** PyTorch, torchvision  
- **Model:** EfficientNet-B0 (pre-trained)  
- **Deployment:** Localhost (development)
- **Environment:** Conda / virtualenv, Python 3.12  

---

### Test Metrics

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 0.9564  |
| Precision | 0.9511  |
| Recall    | 0.9623  |
| F1-score  | 0.9567  |

### Classification Report

| Class   | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| AI (0)  | 0.96      | 0.95   | 0.96     | 10000   |
| REAL(1) | 0.95      | 0.96   | 0.96     | 10000   |
| **Accuracy** | -     | -      | 0.96     | 20000   |
| **Macro Avg** | 0.96  | 0.96   | 0.96     | 20000   |
| **Weighted Avg** | 0.96 | 0.96 | 0.96     | 20000   |

### Confusion Matrix
[[9505 , 495]
 [ 377 , 9623]]

---

## Limitations

- The model may **misclassify images with heavy post-processing** or low resolution.  
- AI-generated images from models outside the dataset’s training distribution may reduce accuracy.  
- Real-world deployment requires **robust security** and **scalable infrastructure** for high-volume image uploads.  
- Current UI is basic; a **modern frontend** (React/Vue) can enhance user experience.  

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/real-vs-ai-image-detector.git
cd real-vs-ai-image-detector

# Create environment
conda create -n imageaidetector python=3.11
conda activate imageaidetector

# Install dependencies
pip install -r requirements.txt
