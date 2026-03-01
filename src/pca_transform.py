from sklearn.decomposition import PCA
import joblib
import os

def apply_pca(X_train, X_test, n_components=10):

    pca = PCA(n_components=n_components, random_state=42)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Variance expliquée totale :",
          sum(pca.explained_variance_ratio_))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pca, "models/pca.pkl")

    return X_train_pca, X_test_pca