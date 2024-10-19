import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib

# Create Flask app
app = Flask(__name__)

# Load the model
model = load_model("model.h5", compile=False)

# Recompile the model after loading with the correct loss and metric
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

# Load the scaler
scaler = joblib.load('scaler.joblib')

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Convert the form inputs to a list of floats
    float_features = [
        float(request.form['Square_Feet']),
        float(request.form['Bedrooms']),
        float(request.form['Age']),
        float(request.form['Location_Rating'])
    ]

    # Scale the first (Square_Feet) and third (Age) features
    features_to_scale = np.array([[float_features[0], float_features[2]]])  # Square_Feet and Age
    scaled_features = scaler.transform(features_to_scale)

    # Combine scaled features with the unscaled Bedrooms
    final_features = np.array([
        scaled_features[0][0],  # Scaled Square_Feet
        float_features[1],      # Bedrooms (unchanged)
        scaled_features[0][1]   # Scaled Age
    ]).reshape(1, -1)  # Reshape to (1, 3)

    # Predict using the loaded model
    prediction = model.predict(final_features)

    # Print the prediction to the console for debugging
    print(f"Predicted price: {prediction[0][0]}")

    # Display the prediction on the HTML page
    return render_template("index.html", prediction_text="The predicted price is ${:.2f}".format(prediction[0][0]))

if __name__ == "__main__":
    app.run(debug=True)
