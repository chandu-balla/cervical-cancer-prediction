from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ==============================
# Load model & scaler
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["age"]),
                float(request.form["partners"]),
                float(request.form["pregnancies"]),
                float(request.form["sexual_years"]),
                float(request.form["smoker"]),
                float(request.form["hormonal_flag"]),
                float(request.form["screening_count"]),
                float(request.form["hpv"]),
            ]

            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)

            pred = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0][1]

            prediction = (
                "⚠️ Cervical Cancer Detected"
                if pred == 1
                else "✅ No Cervical Cancer Detected"
            )
            probability = round(prob, 4)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )

# ==============================
# Run server
# ==============================
if __name__ == "__main__":
    app.run(debug=True)