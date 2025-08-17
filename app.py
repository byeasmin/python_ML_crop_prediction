import numpy as np
from flask import Flask, request, render_template
import pickle

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Form page
@app.route("/form")
def form():
    return render_template("form.html", prediction_text="")

# Prediction logic
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form and convert to float
        float_features = [float(x) for x in request.form.values()]
        features = np.array([float_features])

        # Predict crop
        prediction = model.predict(features)[0]

        # Render form.html with prediction
        return render_template("form.html", prediction_text=f"The Predicted Crop is: {prediction}")

    except Exception as e:
        # Handle errors gracefully
        return render_template("form.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
