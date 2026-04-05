# eye-disease-detection-odir
Eye Disease Detection using Deep Learning on ODIR-5K Dataset
# 🧠 Eye Disease Detection using Deep Learning

## 📌 Dataset
This project uses the ODIR-5K retinal fundus dataset from Kaggle.
- ~10,000 images
- ~5,000 patients
- Classes: Normal, Diabetic Retinopathy, Glaucoma, Cataract

⚠️ Due to size constraints, the dataset is not included in this repository.

👉 Download from Kaggle: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

After downloading, place the dataset inside the `data/` folder.

## ⚙️ Preprocessing
- Resized images to 224×224
- Normalization
- Label encoding

## 🤖 Model
- Architecture: ResNet (Transfer Learning)
- Explainability: Grad-CAM

## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/train.py
```

## 📊 Challenges
- Class imbalance
- Label noise
- Real-world variability

## 📁 Folder Structure
```
eye-disease-detection-odir/
├── data/
│   └── sample_images/
├── notebooks/
│   └── training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── train.py
├── requirements.txt
└── README.md
```
