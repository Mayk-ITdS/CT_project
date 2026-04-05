from pathlib import Path
import napari
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import label
from sklearn.cluster import KMeans

from src.Factory.dataset.build_dataset import build_records, load_pair_from_ct

HERE = Path(__file__).parent
principal_path = HERE / "../principal_sharp_only.csv"

df = pd.read_csv(principal_path)
records = build_records(df)
test_records = records[records["ok"]].reset_index(drop=True)

#     offset test
#     for row in test_records.itertuples():
#     _, mk = load_pair_from_ct(row.ct, row.mk)
#     print(np.min(mk[mk>0]), np.max(mk))   # total class incnsistency
#     centroid = np.mean(mk[mk>0], axis=0)

def extract_features(mk,case_id):
    binary = (mk > 0)
    components, num = label(binary)

    features = []

    Z, Y, X_dim = mk.shape

    for comp_id in range(1, num + 1):
        coords = np.argwhere(components == comp_id)

        centroid = coords.mean(axis=0)
        size = len(coords)
        z_min,y_min,x_min = coords.min(axis=0)
        z_max,y_max,x_max = coords.max(axis=0)

        bbox = [z_max - z_min, y_max - y_min, x_max - x_min]
        bbox_volume = bbox[0] * bbox[1] * bbox[2]
        compactnes = bbox_volume / size
        labels = mk[components == comp_id]
        label_val = np.bincount(labels).argmax()
        centroid_norm = [
            centroid[0] / Z,
            centroid[1] / Y,
            centroid[2] / X_dim
        ]

        features.append({
            "centroid_norm": centroid_norm,
            "size": size,
            "bbox":bbox,
            # "mask": (components == comp_id),
            "case_id": case_id,
            "indices": np.where(components == comp_id),
            "label": label_val,
        })
    # print("Component:", comp_id, "Centroid:", centroid, "Size:", size)
    return features
    # normalize

def global_feature_collection(test_records):
    all_features = []

    for i, row in enumerate(test_records.itertuples()):
        _, mk = load_pair_from_ct(row.ct, row.mk)
        i = df['sample_nr'][i]
        feats = extract_features(mk,case_id=i)

        all_features.extend(feats)
    return all_features

def compute_components_distribution(test_records):
    from scipy.ndimage import label

    counts = []

    for row in test_records.itertuples():
        _, mk = load_pair_from_ct(row.ct, row.mk)

        binary = (mk > 0)
        _, num = label(binary)

        counts.append(num)

        # print(row.ct.name, "→ components:", num)

    return counts

component_counts = compute_components_distribution(test_records)

def global_clustering(all_features,component_counts):

    X = np.array([
        list(f["centroid_norm"]) + [f["size"]]
        for f in all_features
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    K = int(np.median(component_counts))
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    for f, label in zip(all_features, labels):
        f["cluster"] = label
    return all_features

def build_new_mask(mk,case_id,all_features):

    new_mask = np.zeros_like(mk)
    features_for_this_ct = [
        f for f in all_features if f["case_id"] == case_id
    ]
    for f in features_for_this_ct:

        z_idx, y_idx, x_idx = f["indices"]
        cluster = f["cluster"]

        new_mask[z_idx, y_idx, x_idx] = cluster + 1

    return new_mask

def get_new_masks(all_features, test_records):
    new_masks = []
    for i, row in enumerate(test_records.itertuples()):
        ct, mk = load_pair_from_ct(row.ct, row.mk)
        i = df['sample_nr'][i]
        new_mask = build_new_mask(mk, i, all_features)
        new_masks.append((i,ct, mk, new_mask))
    return new_masks

def main():
    all_features = global_feature_collection(test_records)
    all_features = global_clustering(all_features,component_counts)

    for id, ct, mk, new_mask in get_new_masks(all_features, test_records)[:10]:

        viewer = napari.Viewer()

        viewer.add_image(ct, name=f"CT_{id}", colormap="gray")

        viewer.add_labels(
            mk,
            name="Original Mask",
            opacity=0.4
        )
        viewer.add_labels(
            new_mask,
            name="Clustered Mask",
            opacity=0.4
        )
        viewer.layers["Original Mask"].visible = False

        napari.run()
        print("Clusters in this CT:", np.unique(new_mask))

if __name__ == "__main__":
    main()

#print("MIN:", np.min(component_counts))
#print("MAX:", np.max(component_counts))
#print("MEAN:", np.mean(component_counts))
#print("MEDIAN:", np.median(component_counts))
#plt.figure(figsize=(8,5))
#plt.hist(component_counts, bins=15)
#plt.xlabel("Number of components per CT")
#plt.ylabel("Frequency")
#plt.title("Distribution of anatomical components")
#plt.show()