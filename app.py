import os
import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from PIL import Image


app = Flask(__name__)

model = tf.keras.models.load_model("car_recog_incepv3.h5")
with open("labels.txt", "r") as file:
    labels = file.read().splitlines()

co2_model = tf.keras.models.load_model("co2_emission_predictor.h5")

def car_recog_incepv3(image):
    img = Image.open(image).convert("RGB")
    resize_img = tf.image.resize(img, (224, 224))
    img_array = np.expand_dims(np.array(resize_img) / 255, axis=0)
    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    class_name = labels[index]
    confidence_score = predictions[0][index]
    return class_name[2:], confidence_score

def forecast_co2(cylinder: str, engine_size: str, fuel_consumption: str):
    # Preprocess input and make predictions
    input_data = np.array([[float(cylinder), float(engine_size), float(fuel_consumption)]])
    co2_prediction = co2_model.predict(input_data)

    # Convert the float32 prediction to a regular Python float
    co2_prediction_float = co2_prediction.item()

    return co2_prediction_float

@app.route("/")
def index():
    return jsonify ({
        "status" : {
            "code" : 200,
            "message" : "Success fetching the API",
        },
        "data": None
    }), 200

@app.route("/prediction/image", methods=["POST"])
def prediction_image():
    if request.method == "POST":
        image = request.files["image"]
        if image:
            try:
                class_name, confidence_score = car_recog_incepv3(image)
                percentage_confidence = confidence_score * 100
                return jsonify({
                    "status": {
                        "code": 200,
                        "message": "Success",
                    },
                    "data": {
                        "class_name": class_name,
                        "confidence_score": percentage_confidence,
                    }
                }), 200
            except Exception as e:
                return jsonify({
                    "status": {
                        "code": 400,
                        "message": f"Error: {str(e)}",
                    },
                    "data": None,
                }), 400
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "image key not found in request",
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed",
            },
            "data": None,
        }), 405

@app.route("/prediction/forecast", methods=["POST"])
def prediction_forecast():
    if request.method == "POST":
        try:
            # Get input data from the request JSON body
            data = request.get_json()

            # Extract values for cylinder, engine_size, and fuel_consumption
            cylinder = data.get("cylinder")
            engine_size = data.get("engine_size")
            fuel_consumption = data.get("fuel_consumption")

            # Perform CO2 emission forecasting
            co2_forecast = forecast_co2(cylinder, engine_size, fuel_consumption)

            # Return the result as JSON
            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success",
                },
                "data": {
                    "co2_forecast": co2_forecast,
                }
            }), 200
        except Exception as e:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": f"Error: {str(e)}",
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed",
            },
            "data": None,
        }), 405

if __name__ == "__main__":
    app.run(debug=True,
             host="0.0.0.0", 
             port=int(os.environ.get("PORT", 8080)))