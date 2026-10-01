from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'outputs' / 'digits_random_forest.joblib'
classifier = joblib.load(MODEL_PATH)

app = FastAPI(
    title='Handwritten Digit Classifier API',
    version='1.0.0',
    description='Predict a digit from an 8-by-8 grayscale image.',
)

class DigitRequest(BaseModel):
    image: list[list[int]]

    @field_validator('image')
    @classmethod
    def validate_image(cls, image):
        if len(image) != 8 or any(len(row) != 8 for row in image):
            raise ValueError('image must contain exactly 8 rows of 8 pixels')
        if any(pixel < 0 or pixel > 16 for row in image for pixel in row):
            raise ValueError('pixel values must be between 0 and 16')
        return image

class DigitResponse(BaseModel):
    digit: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: list[float]

class BatchRequest(BaseModel):
    images: list[DigitRequest]

    @field_validator('images')
    @classmethod
    def validate_batch_size(cls, images):
        if not 1 <= len(images) <= 32:
            raise ValueError('batch must contain between 1 and 32 images')
        return images

class BatchResponse(BaseModel):
    predictions: list[DigitResponse]

def predict_image(image):
    row = np.asarray(image, dtype=np.float32).reshape(1, -1)
    probabilities = classifier.predict_proba(row)[0]
    digit = int(np.argmax(probabilities))
    return {
        'digit': digit,
        'confidence': float(probabilities[digit]),
        'probabilities': [float(value) for value in probabilities],
    }

@app.get('/health')
def health_check():
    return {'status': 'ok', 'model_loaded': classifier is not None}

@app.get('/model')
def model_info():
    return {
        'model_type': classifier.__class__.__name__,
        'input_shape': [8, 8],
        'pixel_range': [0, 16],
        'classes': [int(value) for value in classifier.classes_],
    }

@app.post('/predict', response_model=DigitResponse)
def predict(request: DigitRequest):
    return predict_image(request.image)

@app.post('/predict/batch', response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    return {'predictions': [predict_image(item.image) for item in request.images]}
