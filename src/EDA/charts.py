import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
import plotly.express as px

from src.Factory.dataset.build_dataset import build_records, load_pair_from_ct

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from src.Factory.MaskRebuilder.Rebuilder import global_feature_collection, global_clustering, \
    compute_components_distribution, get_new_masks


HERE = Path(__file__).parent
df = pd.read_csv("../Extract/principal_sharp_only.csv")

test_records = build_records(df)

all_features = global_feature_collection(test_records)

all_features = global_clustering(all_features,compute_components_distribution(test_records))

get_new_masks(all_features, test_records)
# mean_intensity = ct_values.mean()
# std_intensity = ct_values.std()
# ct_values = ct[components == comp_id]

# contains: 'centroid_norm', 'size', 'bbox', 'mask', 'case_id', 'indices','cluster'

rows = []

for row in test_records.itertuples():
    ct, mk = load_pair_from_ct(row.ct, row.mk)

    # binarna maska
    binary = (mk > 0)

    # labelowanie komponentów
    components, num = label(binary)
    Z, Y, X_dim = mk.shape
    for comp_id in range(1, num + 1):
        coords = np.argwhere(components == comp_id)

        if len(coords) == 0:
            continue
        # centroid (Z,Y,X)
        centroid = coords.mean(axis=0)
        labels = mk[components == comp_id]
        label_val = np.bincount(labels).argmax()
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        bbox = [z_max - z_min, y_max - y_min, x_max - x_min]
        bbox_volume = bbox[0] * bbox[1] * bbox[2]
        centroid_norm = [
            centroid[0] / Z,
            centroid[1] / Y,
            centroid[2] / X_dim
        ]
        # bounding box
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        bbox = (
            z_max - z_min,
            y_max - y_min,
            x_max - x_min
        )
        # wartości HU dla komponentu
        ct_values = ct[components == comp_id]
        rows.append({
            "sample_nr":df['sample_nr'].values[comp_id],
            "ct_path": row.ct,
            "mask_path": row.mk,
            "case_id": row.sample_nr,
            "z": centroid[0],
            "y": centroid[1],
            "x": centroid[2],
            "size": len(coords),
            "bbox_z": bbox[0],
            "bbox_y": bbox[1],
            "bbox_x": bbox[2],
            "hu_mean": float(ct_values.mean()),
            "hu_std": float(ct_values.std()),
            "hu_min": float(ct_values.min()),
            "hu_max": float(ct_values.max()),
            "ct_shape": ct.shape,
            "mask_shape": mk.shape,
            "ct_voxels": ct.size,
            "mk_voxels": np.sum(mk>0),
            "components": num,
            "components_voxels": np.sum(components == comp_id),
            'label': label_val,
            "centroid_norm": centroid_norm,
        })

df_features = pd.DataFrame(rows)

df_features['indices'] = [f['indices'] for f in all_features]
df_features['cluster'] = [f['cluster'] for f in all_features]
df_features["log_size"] = np.log1p(df_features["size"])
df_features.to_csv('../../report_streamlit/data/features.csv',index=False,header=True)
plt.figure(figsize=(8,5))
sns.histplot(data=df_features,x='size', bins=50)
plt.title("Log(size)")
plt.title("Size distribution")
plt.savefig('../../report_streamlit/data/log_size.png')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x="cluster", y='log_size', data=df_features)
plt.title("Cluster vs size (log)")
plt.savefig('../../report_streamlit/data/cluster_vs_size.png')
plt.show()

fig = px.scatter_3d(df_features,
                    x="z",
                    y="y",
                    z='x',
                    color="cluster",
                    size="log_size",
                    size_max=12,
                    opacity=0.7,
                    title="3D Clustering Of Fetal Ossification Centers"
                    )
fig.show()


cross = pd.crosstab(df_features['label'], df_features['cluster'])
cross_norm = cross.div(cross.sum(axis=1), axis=0)
cross_norm.to_csv('../../report_streamlit/data/cross_norm.csv', index=False,header=True)
# --- heatmap ---
plt.figure(figsize=(12,8))
sns.heatmap(cross_norm, cmap="viridis")
plt.title("Normalized label vs cluster")
plt.xlabel("Cluster")
plt.ylabel("Label")
plt.savefig('../../report_streamlit/data/cross_norm.png', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    cross,
    cmap="mako",
    linewidths=0.2,
    linecolor="white",
    cbar_kws={"label": "Proportion"},
    ax=ax
)

ax.set_title("Normalized Label vs Cluster Distribution", fontsize=14)
ax.set_xlabel("Cluster")
ax.set_ylabel("Label")
plt.savefig('../../report_streamlit/data/cross_norm.png', dpi=300)
plt.show()


# --- scatter comparison ---
plt.figure(figsize=(12,5))

# cluster
plt.subplot(1,2,1)
plt.scatter(
    df_features["z"],
    df_features["y"],
    c=df_features["cluster"],
    cmap='tab20',
    s=10
)
plt.title("Clusters")

# label
plt.subplot(1,2,2)
plt.scatter(
    df_features["z"],
    df_features["y"],
    c=df_features["label"],
    cmap='tab20',
    s=10
)
plt.title("True labels")
plt.savefig('../../report_streamlit/data/cross_norm.png', dpi=300)
plt.show()