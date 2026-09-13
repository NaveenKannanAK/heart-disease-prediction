from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load final model and scaler
model = joblib.load("heartcare_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    disease_probability = None
    no_disease_probability = None

    if request.method == "POST":

        try:
            # Get input values
            age = float(request.form["age"])
            sex = float(request.form["sex"])
            cp = float(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = float(request.form["fbs"])
            restecg = float(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = float(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            slope = float(request.form["slope"])
            ca = float(request.form["ca"])
            thal = float(request.form["thal"])

            # Same feature order as training
            input_data = np.array([[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]])

            # Scale input
            scaled_data = scaler.transform(input_data)

            # Prediction
            result = model.predict(scaled_data)[0]

            # Probability
            probabilities = model.predict_proba(scaled_data)[0]

            no_disease_probability = round(
                probabilities[0] * 100, 2
            )

            disease_probability = round(
                probabilities[1] * 100, 2
            )

            if result == 1:
                prediction = "Heart Disease Detected"
            else:
                prediction = "No Heart Disease Detected"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        disease_probability=disease_probability,
        no_disease_probability=no_disease_probability
    )


if __name__ == "__main__":
    app.run(debug=True)