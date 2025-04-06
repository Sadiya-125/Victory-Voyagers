# 🌾 Agro-Sense: Smart Agriculture Assistant

An Integrated AI-powered Web Application for Farmers and Agri-experts that offers:

- 🌱 **Crop Prediction** based on soil parameters and location
- 💊 **Fertilizer Recommendation** based on soil nutrient imbalances
- 🦠 **Plant Disease Detection** using leaf images and deep learning

## 🚀 Features

- Predict the best crop to grow based on NPK, pH, and climate
- Recommend fertilizers or amendments depending on nutrient excess or deficiency
- Detect plant diseases using a pre-trained CNN model on leaf images
- Powered by LLMs and ML models for smart agriculture

## 🏗️ Tech Stack

- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** Python, Flask
- **ML/AI Models:**
  - Random Forest / XGBoost for crop prediction
  - Rule-based + LLM (e.g. GPT) for fertilizer recommendation
  - CNN (e.g., ResNet/MobileNet) for disease classification
- **Libraries:** scikit-learn, TensorFlow/Keras, OpenCV, Pillow, NumPy, Pandas

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sadiya-125/Agro-Sense.git
cd Agro-Sense
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Open your Browser and Navigate to:
🔗 http://127.0.0.1:5000/

## 🚀 Deployed Website

Agro-Sense is fully deployed at https://agro-sense-9n1e.onrender.com
