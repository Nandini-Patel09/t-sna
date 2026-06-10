import pandas as pd
import pickle
import os


from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


os.makedirs(
    "models",
    exist_ok=True
)



# ======================
# Load dataset
# ======================

df = pd.read_csv(
    "dataset/iris.csv"
)


print(df.head())



# ======================
# Remove unnecessary cols
# ======================


if "Id" in df.columns:

    df.drop(
        "Id",
        axis=1,
        inplace=True
    )



if "Species" in df.columns:

    df.drop(
        "Species",
        axis=1,
        inplace=True
    )



# ======================
# Feature Engineering
# ======================


df["Sepal_Ratio"] = (
    df["SepalLengthCm"]
    /
    df["SepalWidthCm"]
)


df["Petal_Ratio"] = (
    df["PetalLengthCm"]
    /
    df["PetalWidthCm"]
)



# ======================
# Scaling
# ======================


scaler = StandardScaler()


scaled = scaler.fit_transform(
    df
)



# ======================
# t-SNE
# ======================


tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)


tsne_data = tsne.fit_transform(
    scaled
)


print(
    tsne_data.shape
)



# ======================
# Create clusters
# ======================


kmeans = KMeans(
    n_clusters=3,
    random_state=42
)


clusters = kmeans.fit_predict(
    tsne_data
)



# ======================
# Predictor for deployment
# ======================


predictor = KNeighborsClassifier(
    n_neighbors=5
)


predictor.fit(
    scaled,
    clusters
)



# ======================
# Save files
# ======================


with open(
    "models/scaler.pkl",
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )



with open(
    "models/predictor.pkl",
    "wb"
) as f:

    pickle.dump(
        predictor,
        f
    )



print(
    "Training completed successfully"
)