from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load your model once when the server starts
model = tf.keras.models.load_model("path/to/your_model.h5")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # Match your model input shape
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    input_data = preprocess_image(image_bytes)
    prediction = model.predict(input_data)[0]

    blood_group_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # You can customize these labels
    labels = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    return JSONResponse(content={
        "blood_group": labels[blood_group_index],
        "confidence": round(confidence, 4),
        "status": "Prediction successful"
    })
