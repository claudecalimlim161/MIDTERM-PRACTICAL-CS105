import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained SVM model
import joblib  # You may need to install joblib (pip install joblib)
model = joblib.load("svm_model.pkl")
sc = joblib.load('scaler.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect user input from the HTML form
        age = float(request.form.get("age"))
        hypertension = float(request.form.get("hypertension"))
        heart_disease = float(request.form.get("heart_disease"))
        bmi = float(request.form.get("bmi"))
        HbA1c_level = float(request.form.get("HbA1c_level"))
        blood_glucose_level = float(request.form.get("blood_glucose_level"))


        # Organize user input into a numpy array
        user_input = np.array([age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level,
                               ]).reshape(1, -1)

        user_input_scaled = sc.transform(user_input)

        prediction = model.predict(user_input_scaled)


        # Determine the prediction message
        if prediction == 1:
            prediction_message = "The user has Diabetes."
        else:
            prediction_message = "The user does not have Diabetes."

        return render_template("index.html", prediction_message=prediction_message)

    return render_template("index.html", prediction_message="")

if __name__ == "__main__":
    app.run(debug=True)
