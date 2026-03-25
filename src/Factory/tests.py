import json
import matplotlib.pyplot as plt
import napari
import numpy as np
import segmentation_models_pytorch as smp
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from src.Factory.load import CTDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
meta = pd.read_csv("meta_index_full.csv")

case_ids = meta['case_id'].unique()

train_ids, test_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

val_files_2_5D = meta[meta['case_id'].isin(val_ids)]['file'].tolist()

case_id = val_ids[0]
valid_case_ids = meta[(meta['case_id'] == case_id) & (meta['plane'] == 'axial') ].sort_values('slice_idx')
files = valid_case_ids['file'].tolist()

with open("val_files.json","w") as f:
    json.dump(val_files_2_5D, f)

with open("val_files.json","r") as f:
    val_files_2_5D = json.load(f)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=5,
    classes=55
).to(device)

model.load_state_dict(torch.load("model_last_v.pth", map_location=device))

val_dataset = CTDataset(files, [])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model.eval()
ct_volume = []
gt_volume = []
pred_volume = []

with torch.no_grad():
    for ct, mask in val_loader:

        ct = ct.to(device)

        pred = model(ct)
        pred = torch.argmax(pred, dim=1)

        #weź jeden batch
        ct_np = ct[0, 0].cpu().numpy()       # (B, H, W)
        gt_np = mask[0].cpu().numpy()           # (B, H, W)
        pred_np = pred[0].cpu().numpy()         # (B, H, W)
        ct_volume.append(ct_np)
        gt_volume.append(gt_np)
        pred_volume.append(pred_np)
        #traktujemy batch jako pseudo-3D
ct = np.stack(ct_volume)
gt = np.stack(gt_volume)
mk = np.stack(pred_volume)
error = (gt != mk).astype(int)

viewer = napari.Viewer()

viewer.add_image(ct, name="CT", colormap="gray")
viewer.add_labels(gt, name="Truth",opacity=0.4)
viewer.add_labels(mk, name="PRED",opacity=0.4)
viewer.add_labels(error, name="ERROR",opacity=0.6)

napari.run()

