# 🎙️ Speech Emotion Recognition System

A Python-based deep learning system that detects human emotions from short speech recordings using **Wav2Vec2** and a custom-trained classifier.

This project loads a pretrained model and provides an easy **Gradio web interface** for inference using uploaded audio (.wav).

---

## 📌 Overview

The goal of this project is to classify human emotions from speech into categories such as:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

The system uses:

* A **Wav2Vec2 pretrained backbone**
* Feature extraction from raw audio
* Torch + torchaudio + HuggingFace transformers
* A simple Gradio UI for real-time prediction

---

## 📂 Dataset

We used the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** speech dataset from Kaggle.

👉 Dataset link:
[https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

This dataset contains emotional speech audio clips. Each audio file is labeled with one emotion.

⚠️ **NOTE:** Dataset audio files are not included in this repository.
Please download the dataset manually from Kaggle and preprocess before training.

---

## 🚀 Features & Workflow

1. **Feature Extraction**
   Audio is resampled to 16 kHz and features extracted using `AutoFeatureExtractor` from HuggingFace.

2. **Model Architecture**
   Uses `Wav2Vec2ForSequenceClassification` for speech emotion classification.

3. **Inference**
   Speech uploaded by the user is passed through the model to predict the emotion with confidence score.

4. **UI**
   A Gradio interface accepts .wav audio upload and displays predictions.

---

## 🧠 Installation

Clone repository:

```bash
git clone https://github.com/<yourusername>/emotion-speech-recognition-system.git
cd emotion-speech-recognition-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📋 Requirements

Below is a sample `requirements.txt` file:

```
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.23.0
gradio>=4.0.0
transformers>=4.35.0
```

---

## 📦 Folder Structure

```
emotion-speech-recognition-system/
├── models/
│   └── emotion_wav2vec2_model/
├── app.py
├── requirements.txt
└── README.md
```

* **models/** — Pretrained model (downloaded or trained)
* **app.py** — Gradio inference application
* **requirements.txt** — Python dependencies

---

## 🛠 How to Run

Run the Gradio app:

```bash
python app.py
```

It will launch a web interface where you can upload a .wav audio file to detect emotion.

---

## 🧩 Training (optional)

If you want to train your own model using RAVDESS:

1. Download and extract the dataset from Kaggle.
2. Preprocess audio (resample to 16kHz, label encode).
3. Train with HuggingFace Trainer or custom PyTorch loop.
4. Save model to:

```
models/emotion_wav2vec2_model/
```

---

## 📊 Results

Emotion prediction includes:

* **Label**: e.g., *Happy*
* **Confidence score**
