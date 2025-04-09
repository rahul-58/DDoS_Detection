from flask import Flask, render_template, request
import joblib
import numpy as np
from collections import Counter
from preprocess import preprocess_input

models = {
    "LogisticRegression": joblib.load("models/LogisticRegression.pkl"),
    "SVM": joblib.load("models/SVM.pkl"),
    "RandomForest": joblib.load("models/RandomForest.pkl"),
    "XGBoost": joblib.load("models/XGBoost.pkl"),
}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        user_input = {
            "Highest Layer": request.form["highest_layer"],
            "Transport Layer": request.form["transport_layer"],
            "Source Port": int(request.form["source_port"]),
            "Dest Port": int(request.form["dest_port"]),
            "Packet Length": int(request.form["packet_length"]),
            "Packets/Time": float(request.form["packets_per_time"])
        }

        X = preprocess_input(user_input)

        predictions = {}
        confidences = {}

        for name, model in models.items():
            pred = model.predict(X)[0]
            predictions[name] = int(pred)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidences[name] = round(max(proba), 3)
            else:
                confidences[name] = 0 

        majority = Counter(predictions.values()).most_common(1)[0][0]

        confidences = {k: float(v) for k, v in confidences.items()}

        return render_template("result.html", predictions=predictions, confidences=confidences, majority=majority)

    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
