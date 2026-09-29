# ============================================================
# SMART PLANT MONITORING SYSTEM
# Backend using FastAPI + Deep Learning + Gemini GenAI
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import os
import pickle
import numpy as np

from dotenv import load_dotenv


# ============================================================
# GEMINI AI
# ============================================================

try:
    from google import genai
except ImportError:
    genai = None


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Smart Plant Monitoring API",
    description="IoT + Deep Learning + Generative AI backend",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DEEP LEARNING MODEL
# ============================================================

MODEL_PATH = "plant_model.keras"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None


try:

    import tensorflow as tf

    if os.path.exists(MODEL_PATH):

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

        print("✅ Deep Learning model loaded.")

    else:

        print(
            "⚠️ plant_model.keras not found."
        )

except Exception as e:

    print(
        "⚠️ TensorFlow model could not be loaded:",
        e
    )


# ============================================================
# LOAD SCALER
# ============================================================

try:

    if os.path.exists(SCALER_PATH):

        with open(
            SCALER_PATH,
            "rb"
        ) as file:

            scaler = pickle.load(file)

        print("✅ Scaler loaded.")

    else:

        print(
            "⚠️ scaler.pkl not found."
        )

except Exception as e:

    print(
        "⚠️ Scaler could not be loaded:",
        e
    )


# ============================================================
# SENSOR DATA MODEL
# ============================================================

class SensorData(BaseModel):

    soil_moisture: float = Field(
        ...,
        ge=0,
        le=100,
        description="Soil moisture percentage"
    )

    temperature: float = Field(
        ...,
        ge=-20,
        le=80,
        description="Temperature in Celsius"
    )

    humidity: float = Field(
        ...,
        ge=0,
        le=100,
        description="Humidity percentage"
    )


# ============================================================
# LATEST SENSOR DATA
# ============================================================

latest_data = {

    "soil_moisture": 62.0,

    "temperature": 24.0,

    "humidity": 68.0,

    "status": "Healthy"

}


# ============================================================
# PLANT STATUS PREDICTION
# ============================================================

def predict_plant_status(
    soil_moisture,
    temperature,
    humidity
):

    # --------------------------------------------------------
    # If Deep Learning model is available
    # --------------------------------------------------------

    if model is not None:

        try:

            input_data = np.array(
                [[
                    soil_moisture,
                    temperature,
                    humidity
                ]]
            )

            # Apply scaler if available

            if scaler is not None:

                input_data = scaler.transform(
                    input_data
                )

            prediction = model.predict(
                input_data,
                verbose=0
            )

            class_index = int(
                np.argmax(prediction)
            )

            labels = {

                0: "Healthy",

                1: "Needs Water",

                2: "Dry / Risk"

            }

            return labels.get(
                class_index,
                "Healthy"
            )

        except Exception as e:

            print(
                "Model prediction error:",
                e
            )


    # --------------------------------------------------------
    # Fallback rule-based prediction
    # --------------------------------------------------------

    if soil_moisture < 25:

        return "Dry / Risk"

    elif soil_moisture < 40:

        return "Needs Water"

    else:

        return "Healthy"


# ============================================================
# HOME API
# ============================================================

@app.get("/")
def home():

    return {

        "message":
        "🌱 Smart Plant Monitoring API is running!",

        "status":
        "success"

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():

    return {

        "status": "healthy",

        "deep_learning_model":
            model is not None,

        "gemini_configured":
            bool(GEMINI_API_KEY)

    }


# ============================================================
# GET LATEST SENSOR DATA
# ============================================================

@app.get("/api/latest")
def get_latest_data():

    return latest_data


# ============================================================
# RECEIVE SENSOR DATA FROM ESP32
# ============================================================

@app.post("/api/sensor-data")
def receive_sensor_data(
    data: SensorData
):

    global latest_data


    # --------------------------------------------------------
    # Deep Learning Prediction
    # --------------------------------------------------------

    status = predict_plant_status(

        data.soil_moisture,

        data.temperature,

        data.humidity

    )


    # --------------------------------------------------------
    # Store Latest Data
    # --------------------------------------------------------

    latest_data = {

        "soil_moisture":
            round(
                data.soil_moisture,
                2
            ),

        "temperature":
            round(
                data.temperature,
                2
            ),

        "humidity":
            round(
                data.humidity,
                2
            ),

        "status":
            status

    }


    print(
        "📡 Sensor Data:",
        latest_data
    )


    return {

        "message":
            "Sensor data received successfully",

        "data":
            latest_data

    }


# ============================================================
# GEMINI AI RECOMMENDATION
# ============================================================

def generate_ai_recommendation(
    soil_moisture,
    temperature,
    humidity,
    status
):

    # --------------------------------------------------------
    # If Gemini API key is not available
    # --------------------------------------------------------

    if not GEMINI_API_KEY:

        if status == "Dry / Risk":

            return (
                "The soil moisture is very low. "
                "Please water the plant and continue "
                "monitoring the soil."
            )

        elif status == "Needs Water":

            return (
                "The soil is getting dry. "
                "Consider watering the plant soon."
            )

        else:

            return (
                "The plant conditions look good. "
                "Continue regular monitoring."
            )


    # --------------------------------------------------------
    # Gemini SDK not installed
    # --------------------------------------------------------

    if genai is None:

        return (
            "Gemini AI package is not installed. "
            "Please install the google-genai package."
        )


    try:

        # ----------------------------------------------------
        # Create Gemini Client
        # ----------------------------------------------------

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )


        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = f"""

You are an AI plant-care assistant.

Analyze these smart plant sensor readings:

Soil Moisture: {soil_moisture}%
Temperature: {temperature}°C
Humidity: {humidity}%
Deep Learning Status: {status}

Give a simple plant-care recommendation.

Rules:
- Give only 2 to 4 short sentences.
- Use simple English.
- Mention watering only if necessary.
- Do not claim to diagnose plant diseases.
- Do not invent sensor values.
- Give practical advice.

"""


        # ----------------------------------------------------
        # Generate Response
        # ----------------------------------------------------

        response = client.models.generate_content(

            model="gemini-3.8-flash",

            contents=prompt

        )


        # ----------------------------------------------------
        # Return AI Text
        # ----------------------------------------------------

        if response.text:

            return response.text.strip()


        return (
            "The plant is being monitored. "
            "Continue checking the sensor readings."
        )


    except Exception as e:

        print(
            "Gemini API error:",
            e
        )


        # ----------------------------------------------------
        # Fallback Recommendation
        # ----------------------------------------------------

        if status == "Dry / Risk":

            return (
                "The soil moisture is very low. "
                "Please water the plant and monitor it."
            )

        elif status == "Needs Water":

            return (
                "The soil is becoming dry. "
                "Consider watering the plant soon."
            )

        else:

            return (
                "The current plant conditions look healthy. "
                "Continue regular monitoring."
            )


# ============================================================
# RECOMMENDATION API
# ============================================================

@app.post("/api/recommendation")
def get_recommendation(
    data: SensorData
):

    # --------------------------------------------------------
    # Predict Status
    # --------------------------------------------------------

    status = predict_plant_status(

        data.soil_moisture,

        data.temperature,

        data.humidity

    )


    # --------------------------------------------------------
    # Generate AI Recommendation
    # --------------------------------------------------------

    recommendation = generate_ai_recommendation(

        data.soil_moisture,

        data.temperature,

        data.humidity,

        status

    )


    return {

        "status":
            status,

        "recommendation":
            recommendation

    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )