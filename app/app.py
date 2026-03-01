from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Chargement des modèles (chemins relatifs au dossier app/)
BASE = os.path.join(os.path.dirname(__file__), "..")
imputer = joblib.load(os.path.join(BASE, "models/imputer.joblib"))
scaler  = joblib.load(os.path.join(BASE, "models/scaler.joblib"))
modele  = joblib.load(os.path.join(BASE, "models/random_forest.joblib"))

PAGE_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Prédiction Churn</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; }
        h1   { color: #333; }
        pre  { background: #f4f4f4; padding: 15px; border-radius: 5px; }
        code { color: #c7254e; }
    </style>
</head>
<body>
    <h1>🛒 Prédiction Churn Client</h1>
    <p>API REST pour prédire si un client va partir (churn = 1) ou rester (churn = 0).</p>

    <h2>Endpoint disponible</h2>
    <p><strong>POST</strong> <code>/predict</code></p>
    <p>Envoyer un JSON avec les features du client.</p>

    <h2>Exemple</h2>
    <pre>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"Recency": 30, "Frequency": 5, "MonetaryTotal": 500}'</pre>

    <h2>Réponse</h2>
    <pre>{
  "churn": 0,
  "probabilite_churn": 0.12,
  "interpretation": "Client fidèle"
}</pre>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"erreur": "Aucune donnée reçue"}), 400

        df = pd.DataFrame([data])

        # Aligner les colonnes avec celles attendues par le modèle
        feature_names = scaler.feature_names_in_
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        # Pipeline : imputation -> normalisation -> prédiction
        df_imputed = pd.DataFrame(imputer.transform(df),       columns=feature_names)
        df_scaled  = pd.DataFrame(scaler.transform(df_imputed), columns=feature_names)

        prediction  = modele.predict(df_scaled)[0]
        probabilite = modele.predict_proba(df_scaled)[0][1]

        return jsonify({
            "churn": int(prediction),
            "probabilite_churn": round(float(probabilite), 3),
            "interpretation": "Client à risque" if prediction == 1 else "Client fidèle"
        })

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
