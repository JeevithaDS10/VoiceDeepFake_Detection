from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from predict import predict_voice

app = FastAPI(title="Voice Deepfake Detection API")

# Allow frontend access later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary upload folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Voice Deepfake Detection Backend Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed = [".wav", ".mp3"]

    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        return {"error": "Only .wav and .mp3 files allowed"}

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_voice(file_path)

    return {
        "filename": file.filename,
        "result": result
    }