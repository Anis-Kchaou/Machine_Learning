from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_clustering(X):

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X)

    score = silhouette_score(X, clusters)
    print("Silhouette Score:", score)

    return kmeans, clusters