from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

# ==========================
# FASTAPI APP
# ==========================
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model("agrovision_model.h5")

print("✅ Model Loaded")
print("📊 Output Shape:", model.output_shape)

# ==========================
# LOAD DISEASE INFO
# ==========================
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# ==========================
# LOAD CLASSES
# ==========================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Sort according to model output order
class_names = [
    class_name
    for class_name, index in sorted(
        class_indices.items(),
        key=lambda x: x[1]
    )
]

print("🔥 Classes Loaded:", class_names)
print("📦 Total Classes:", len(class_names))

IMG_SIZE = 224

# ==========================
# SEVERITY LOGIC
# ==========================
def get_severity(confidence):
    if confidence < 0.60:
        return "Mild"
    elif confidence < 0.85:
        return "Moderate"
    else:
        return "Severe"

# ==========================
# FRONTEND ROUTE
# ==========================
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# ==========================
# PREDICTION API
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Process image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction = model.predict(image_array, verbose=0)

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if predicted_index >= len(class_names):
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Model classes do not match class_indices.json"
                }
            )

        predicted_class = class_names[predicted_index]
        severity = get_severity(confidence)

        info = disease_info.get(predicted_class, {})

        return JSONResponse({
            "status": "success",
            "prediction": {
                "disease": info.get(
                    "display_name",
                    predicted_class
                ),
                "confidence": round(confidence, 4),
                "severity": severity
            },
            "details": {
                "cause": info.get("cause", "N/A"),
                "treatment": info.get("treatment", "N/A"),
                "prevention": info.get("prevention", "N/A")
            }
        })

    except Exception as e:
        print("❌ Prediction Error:", str(e))

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# ==========================
# HEALTH CHECK
# ==========================
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }