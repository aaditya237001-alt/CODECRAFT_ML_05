# 🍽️ Food Recognition & Calorie Estimator

This project uses a trained deep learning model to **identify food items from images** and **estimate their calories** (per 100g).  
It is based on a ResNet-18 architecture fine-tuned on the **Food-101** dataset.

---

## 📂 Project Files
- `food_predictor.py` – Main script to load the model, process an image, and display the prediction.
- `food101_model.pth` – Trained PyTorch model weights (tracked via Git LFS).
- `label_encoder.pkl` – Label encoder to map class indices to food names.
- `requirements.txt` – List of dependencies.
- `README.md` – Project documentation.

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
📦 Requirements

torch

torchvision

pillow

scikit-learn

🧠 Model Details

Architecture: ResNet-18 (modified final layer for 101 classes)

Dataset: Food-101

Training: Fine-tuned on Food-101 images

Calorie Mapping: Based on manually collected nutrition data

📌 Notes

Place food101_model.pth and label_encoder.pkl in the same folder as food_predictor.py.

If the model file is larger than 100MB, use Git LFS to store it.

Calorie estimates are approximate and based on 100g serving sizes.
