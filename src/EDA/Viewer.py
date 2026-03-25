from pathlib import Path
import random
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from src.Factory.load import load_rdata, load_pair_from_ct, build_records

def voxels_per_class(test_records):
    sizes = []
    for row in test_records.itertuples():
        _, mk = load_pair_from_ct(row.ct, row.mk)

        for cls in np.unique(mk):
            if cls == 0:
                continue

            size = (mk == cls).sum()
            sizes.append(size)
            print(row.ct.name, cls, size)
    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=30)
    plt.title("Distribution of segment sizes (voxel count)")
    plt.xlabel("Voxel count")
    plt.ylabel("Frequency")
    plt.show()

HERE = Path(__file__).parent
principal_path = HERE / "principal.csv"
df = pd.read_csv(principal_path)

records = build_records(df)

test_records = records[records["ok"]].reset_index(drop=True)
# voxels_per_class(test_records)
print(test_records.keys())

viewer = napari.Viewer()
@viewer.mouse_move_callbacks.append
def inspect_label(viewer, event):
    layer = viewer.layers.selection.active

    if layer is None:
        return

    val = layer.get_value(event.position)

    if val is not None:
        print(f"label: {val}")

index = 0
def load_index(i):
    row = test_records.iloc[i]
    ct, mk = load_pair_from_ct(row["ct"], row["mk"])

    viewer.layers.clear()
    viewer.add_image(ct, name=f"CT_{i}")
    viewer.add_labels(mk, name=f"MK_{i}", opacity=0.4)

@viewer.bind_key("n",overwrite=True)
def next_case(viewer):
    global index
    print("N pressed")
    index = (index + 1) % len(test_records)
    load_index(index)

@viewer.bind_key("b",overwrite=True)
def prev_case(viewer):
    global index
    print("B pressed")
    index = (index - 1) % len(test_records)
    load_index(index)

def main():

    load_index(0)

    napari.run()

if __name__ == "__main__":
    main()

