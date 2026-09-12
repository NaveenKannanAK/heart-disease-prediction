from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    predicted_class = None

    disease_probability = None
    no_disease_probability = None

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # GET VALUES FROM HTML FORM
            # ------------------------------------------------

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


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if age <= 0:
                raise ValueError("Age must be greater than 0.")

            if trestbps <= 0:
                raise ValueError("Resting blood pressure must be greater than 0.")

            if chol <= 0:
                raise ValueError("Cholesterol must be greater than 0.")

            if thalach <= 0:
                raise ValueError("Maximum heart rate must be greater than 0.")


            # ------------------------------------------------
            # ARRANGE FEATURES
            #
            # MUST BE THE SAME ORDER USED DURING TRAINING
            # ------------------------------------------------

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


            # ------------------------------------------------
            # SCALE INPUT
            # ------------------------------------------------

            scaled_data = scaler.transform(input_data)


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            result = model.predict(scaled_data)[0]

            predicted_class = int(result)


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            probabilities = model.predict_proba(scaled_data)[0]

            # Match probabilities with their actual classes
            class_probabilities = dict(
                zip(model.classes_, probabilities)
            )

            no_disease_probability = round(
                class_probabilities.get(0, 0) * 100,
                2
            )

            disease_probability = round(
                class_probabilities.get(1, 0) * 100,
                2
            )


            # ------------------------------------------------
            # FINAL PREDICTION
            # ------------------------------------------------

            if result == 1:

                prediction = "Heart Disease Detected"

            else:

                prediction = "No Heart Disease Detected"


        except Exception as e:

            prediction = f"Error: {str(e)}"


    # ========================================================
    # SEND DATA TO HTML
    # ========================================================

    return render_template(
        "index.html",

        prediction=prediction,

        predicted_class=predicted_class,

        disease_probability=disease_probability,

        no_disease_probability=no_disease_probability
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)