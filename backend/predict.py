import joblib
import librosa
import numpy as np
import os

# Load model
MODEL_PATH = "../models/final_model.pkl"
model = joblib.load(MODEL_PATH)


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    core = np.hstack([mfcc, chroma, contrast, zcr, rms, flatness])

    # binaural simulated features
    delay = int(0.0003 * sr)
    left = y.copy()
    right = np.roll(y, delay) * 0.92

    itd = delay / sr
    ild = np.mean(np.abs(left)) - np.mean(np.abs(right))
    corr = np.corrcoef(left[:len(right)], right)[0, 1]
    width = np.mean(np.abs(left - right))
    angle = np.arctan2(itd, abs(ild) + 1e-6)

    binaural = np.array([itd, ild, corr, width, angle])

    return np.hstack([core, binaural])


def predict_voice(file_path):
    feat = extract_features(file_path).reshape(1, -1)

    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0]

    label = "REAL" if pred == 0 else "FAKE"

    confidence = float(np.max(prob))

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }