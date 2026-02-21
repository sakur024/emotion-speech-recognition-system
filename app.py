import torch
import torchaudio
import numpy as np
import gradio as gr
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

# =============================
# LOAD MODEL
# =============================
MODEL_PATH = "./emotion_wav2vec2_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("Model loaded successfully.")


# =============================
# PREDICTION FUNCTION
# =============================
def predict_emotion(audio):
    if audio is None:
        return "No audio provided"

    sr, speech = audio  # (sample_rate, numpy array)

    # Convert stereo to mono if needed
    if len(speech.shape) > 1:
        speech = np.mean(speech, axis=1)

    speech = speech.astype(np.float32)

    # Resample to 16kHz if needed
    if sr != 16000:
        speech_tensor = torch.tensor(speech)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech = resampler(speech_tensor).numpy()

    # Limit to 3 seconds
    max_length = 16000 * 3
    speech = speech[:max_length]

    # Feature extraction
    inputs = feature_extractor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    pred_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_id].item()

    emotion = model.config.id2label[pred_id]

    return f"Emotion: {emotion}\nConfidence: {confidence:.4f}"


# =============================
# GRADIO UI (UPLOAD ONLY)
# =============================
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="numpy", sources=["upload"], label="Upload WAV File"),
    outputs="text",
    title="Speech Emotion Recognition",
    description="Upload a 2–3 second WAV file."
)

if __name__ == "__main__":
    interface.queue().launch()