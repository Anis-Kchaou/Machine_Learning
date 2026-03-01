import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import os

# Chemin racine du projet (dossier parent de src/)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# -------------------------------------------------------
# 1. Chargement des données prétraitées
# -------------------------------------------------------
X_train = pd.read_csv(os.path.join(BASE, "data/train_test/X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE, "data/train_test/X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE, "data/train_test/y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(BASE, "data/train_test/y_test.csv")).squeeze()

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)

# -------------------------------------------------------
# 2. ACP - Réduction de dimension
# -------------------------------------------------------
n_components = 10
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
joblib.dump(pca, os.path.join(BASE, "models/pca.joblib"))

variance_cumulee = pca.explained_variance_ratio_.cumsum()
print(f"\nACP - Variance expliquée ({n_components} composantes) : {variance_cumulee[-1]*100:.2f}%")

# Graphique variance expliquée
plt.figure()
plt.plot(range(1, n_components + 1), variance_cumulee, marker='o')
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance cumulée")
plt.title("ACP - Variance expliquée cumulée")
plt.grid(True)
plt.savefig(os.path.join(BASE, "reports/acp_variance.png"))
plt.close()
print("Graphique ACP sauvegardé dans reports/")

# -------------------------------------------------------
# 3. Clustering KMeans (non supervisé)
# -------------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_train_pca)
joblib.dump(kmeans, os.path.join(BASE, "models/kmeans.joblib"))
print("\nKMeans entraîné - 4 clusters créés")

# -------------------------------------------------------
# 4. Classification - Random Forest
# -------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(BASE, "models/random_forest.joblib"))

y_pred_rf = rf.predict(X_test)
print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# -------------------------------------------------------
# 5. Classification - Régression Logistique (sur ACP)
# -------------------------------------------------------
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_pca, y_train)
joblib.dump(lr, os.path.join(BASE, "models/logistic_regression.joblib"))

y_pred_lr = lr.predict(X_test_pca)
print("\n=== Régression Logistique (sur ACP) ===")
print(classification_report(y_test, y_pred_lr))

print("\nTous les modèles ont été sauvegardés dans models/")