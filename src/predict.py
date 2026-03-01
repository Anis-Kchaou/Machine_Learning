import pandas as pd
import joblib


def predire(data_dict):
    """
    Prédit le churn pour un client.

    Paramètre :
        data_dict : dictionnaire contenant les features du client

    Retourne :
        dict avec 'churn' (0 ou 1) et 'probabilite_churn'
    """
    imputer = joblib.load("models/imputer.joblib")
    scaler  = joblib.load("models/scaler.joblib")
    modele  = joblib.load("models/random_forest.joblib")

    # Création du dataframe
    df = pd.DataFrame([data_dict])

    # Aligner les colonnes avec celles du scaler
    feature_names = scaler.feature_names_in_
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Imputation puis normalisation
    df_imputed = pd.DataFrame(imputer.transform(df), columns=feature_names)
    df_scaled  = pd.DataFrame(scaler.transform(df_imputed), columns=feature_names)

    # Prédiction
    prediction  = modele.predict(df_scaled)[0]
    probabilite = modele.predict_proba(df_scaled)[0][1]

    return {
        "churn": int(prediction),
        "probabilite_churn": round(float(probabilite), 3)
    }


if __name__ == "__main__":
    # Exemple : client avec valeurs par défaut
    colonnes = pd.read_csv("data/train_test/X_train.csv").columns
    exemple  = {col: 0 for col in colonnes}

    resultat = predire(exemple)
    print("Prédiction :", resultat)
    if resultat["churn"] == 1:
        print("→ Client à risque de churn")
    else:
        print("→ Client fidèle")
