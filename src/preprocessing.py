import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

sys.path.append(os.path.dirname(__file__))
from utils import supprimer_features_inutiles, parser_dates, feature_engineering, corriger_aberrants

# -------------------------------------------------------
# 1. Chargement des données
# -------------------------------------------------------
df = pd.read_csv("data/raw/clients.csv")
print("Données chargées :", df.shape)

# -------------------------------------------------------
# 2. Suppression des colonnes inutiles
# -------------------------------------------------------
# Newsletter : constante (toujours "Yes") -> inutile
if 'Newsletter' in df.columns:
    df.drop(columns=['Newsletter'], inplace=True)

# LastLoginIP : extraire si c'est une IP privée ou publique
if 'LastLoginIP' in df.columns:
    df['IP_Prive'] = df['LastLoginIP'].str.startswith(('192.168', '10.', '172.')).astype(int)
    df.drop(columns=['LastLoginIP'], inplace=True)

# -------------------------------------------------------
# 3. Parsing des dates
# -------------------------------------------------------
if 'RegistrationDate' in df.columns:
    df = parser_dates(df, 'RegistrationDate')

# -------------------------------------------------------
# 4. Feature engineering
# -------------------------------------------------------
df = feature_engineering(df)

# -------------------------------------------------------
# 5. Suppression variance nulle
# -------------------------------------------------------
df = supprimer_features_inutiles(df)

# -------------------------------------------------------
# 6. Correction des valeurs aberrantes
# -------------------------------------------------------
df = corriger_aberrants(df)

# -------------------------------------------------------
# 7. Encodage des variables catégorielles
# -------------------------------------------------------

# Encodage ordinal (ordre logique)
ordinal_map = {
    'AgeCategory':   ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
    'SpendingCat':   ['Low', 'Medium', 'High', 'VIP'],
    'LoyaltyLevel':  ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
    'ChurnRisk':     ['Faible', 'Moyen', 'Élevé', 'Critique'],
    'BasketSize':    ['Petit', 'Moyen', 'Grand', 'Inconnu'],
    'PreferredTime': ['Nuit', 'Matin', 'Midi', 'Après-midi', 'Soir'],
}

for col, categories in ordinal_map.items():
    if col in df.columns:
        enc = OrdinalEncoder(
            categories=[categories],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        df[col] = enc.fit_transform(df[[col]])

# Encodage One-Hot (pas d'ordre)
one_hot_cols = [
    'CustomerType', 'FavoriteSeason', 'Region', 'WeekendPref',
    'ProdDiversity', 'Gender', 'AccountStatus', 'RFMSegment'
]
one_hot_cols = [c for c in one_hot_cols if c in df.columns]
df = pd.get_dummies(df, columns=one_hot_cols)

# Target encoding pour Country (trop de modalités)
if 'Country' in df.columns and 'Churn' in df.columns:
    country_enc = df.groupby('Country')['Churn'].mean()
    df['Country'] = df['Country'].map(country_enc)

# -------------------------------------------------------
# 8. Séparation X / y
# -------------------------------------------------------
target = 'Churn'
X = df.drop(columns=[target, 'CustomerID'], errors='ignore')
y = df[target]

# -------------------------------------------------------
# 9. Imputation KNN (sur X entier avant split)
# -------------------------------------------------------
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
joblib.dump(imputer, "models/imputer.joblib")
print("Imputer sauvegardé.")

# -------------------------------------------------------
# 10. Split train / test
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------------
# 11. Normalisation (fit sur train uniquement)
# -------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
joblib.dump(scaler, "models/scaler.joblib")
print("Scaler sauvegardé.")

# -------------------------------------------------------
# 12. Sauvegarde
# -------------------------------------------------------
df.to_csv("data/processed/clients_clean.csv", index=False)
X_train_scaled.to_csv("data/train_test/X_train.csv", index=False)
X_test_scaled.to_csv("data/train_test/X_test.csv",   index=False)
y_train.to_csv("data/train_test/y_train.csv", index=False)
y_test.to_csv("data/train_test/y_test.csv",   index=False)

print("Prétraitement terminé.")
print(f"Train : {X_train_scaled.shape}  |  Test : {X_test_scaled.shape}")
