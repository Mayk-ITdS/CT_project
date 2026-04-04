from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]

class CTDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def remap_mask(self,mask):
        new_mask = torch.zeros_like(mask)

        new_mask[mask == 0] = 0

        new_mask[mask == 5] = 1
        new_mask[mask == 6] = 2

        new_mask[(mask == 7) | (mask == 8)] = 3
        new_mask[(mask == 9) | (mask == 10)] = 4
        new_mask[(mask == 11) | (mask == 12)] = 5
        new_mask[(mask == 13) | (mask == 14)] = 6

        return new_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]
        data = torch.load(path, map_location="cpu")

        ct = data['ct']
        mk = data['mk']

        ct = ct.float()
        mk = mk.long()
        mk = self.remap_mask(mk)
        assert ct.shape[0] == 5, f"CT shape wrong: {ct.shape}"
        case_id = path
        ct = (ct - ct.mean()) / (ct.std() + 1e-6)

        return ct, mk,case_id

def build_dataloaders(sampler,train_files,val_files,test_files):

    train_dataset = CTDataset(train_files)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_dataset = CTDataset(val_files)
    val_loader = DataLoader(val_dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True)
    test_dataset = CTDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_loader, val_loader, test_loader

def load_my_model(path,device):

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=5,
        classes=7
    ).to(device, non_blocking=True)

    state = torch.load(ROOT / path, map_location=device)

    model.load_state_dict(state)

    model.eval()
    return model

# def train_one_epoch(...):
#     return train_loss
#
# def validate(...):
#     return val_loss, val_dice

def dice_score(pred, target):

    pred = torch.argmax(pred, dim=1)

    dice_scores = []
    classes = torch.unique(target)
    for cls in classes:
        if cls <= 1:
            continue
        pred_c = pred == cls
        target_c = target == cls

        intersection = (pred_c & target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union == 0:
            continue

        dice = (2 * intersection) / (union + 1e-6)
        dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0

    return torch.mean(torch.stack(dice_scores)).item()

def custom_lossfn(y_pred,y_true):
    ce_loss = nn.CrossEntropyLoss(ignore_index=1)
    dice_loss = smp.losses.TverskyLoss(mode='multiclass', ignore_index=1, alpha=0.3, beta=0.7)
    return 0.5 * ce_loss(y_pred,y_true) + 0.5 * dice_loss(y_pred,y_true)

def evaluate_model(model, test_loader, device):
    test_loss = 0
    dice_total = 0

    per_class = {}
    per_case = {}
    with torch.no_grad():
        for ct,mk,case_id in tqdm(test_loader):
            ct = ct.to(device)
            mk = mk.to(device)

            pred = model(ct)
            pred_mask = torch.argmax(pred, dim=1)

            loss = custom_lossfn(pred, mk)
            test_loss += loss.item()

            # GLOBAL DICE
            d = dice_score(pred, mk)
            dice_total += d

            # PER CASE
            for i in range(ct.size(0)):
                cid = case_id[i]
                per_case.setdefault(cid, []).append(d)
            classes = torch.unique(mk)
            for cls in classes:
                if cls <= 1:
                    continue

                pred_c = pred_mask == cls
                target_c = mk == cls

                intersection = (pred_c & target_c).sum().item()
                union = pred_c.sum().item() + target_c.sum().item()

                if union == 0:
                    continue

                dice = (2 * intersection) / (union + 1e-6)

                per_class.setdefault(int(cls), []).append(dice)

                test_loss /= len(test_loader)
                dice_avg = dice_total / len(test_loader)

        pd.DataFrame([
            {"class": cls, "dice": sum(vals) / len(vals)}
            for cls, vals in per_class.items()
        ]).to_csv("test_per_class.csv", index=False)

        pd.DataFrame([
            {"case_id": cid, "dice": sum(vals) / len(vals)}
            for cid, vals in per_case.items()
        ]).to_csv("test_per_case.csv", index=False)

        return {
            "loss": test_loss,
            "dice": dice_avg
        }

import napari

def visualize_napari(model, dataset, device, idx=0):

    model.eval()

    ct,mk ,case_id = dataset[idx]

    ct_input = ct.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(ct_input)
        pred_mask = torch.argmax(pred, dim=1).cpu().squeeze(0)

    ct_np = ct.cpu().numpy()          # (5,128,128)
    mk_np = mk.cpu().numpy()          # (128,128)
    pred_np = pred_mask.numpy()       # (128,128)

    print("PRED unique:", np.unique(pred_np))
    print("CASE:", case_id)
    print("CT mean/std:", ct.mean().item(), ct.std().item())
    print("GT unique:", torch.unique(mk))
    print("GT classes:", torch.unique(mk))
    print("PRED classes:", torch.unique(pred_mask))
    print((pred_np == 6).sum(), pred_np.size)
    vmin, vmax = np.percentile(ct, [5, 95])
    ct_np = np.clip(ct_np, vmin, vmax)
    viewer = napari.Viewer()

    viewer.add_image(ct_np[2], name="CT", colormap="gray")
    viewer.add_labels(mk_np, name="GT")
    viewer.add_labels(pred_np, name="PRED")

    napari.run()

import matplotlib.pyplot as plt
from pathlib import Path
import os

def visualize_test_sample(model, dataset, device, idx=0, save=False):

    model.eval()

    data = dataset[idx]

    ct, mk, case_id = data

    ct = ct.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(ct)
        pred_mask = torch.argmax(pred, dim=1).cpu().squeeze(0)

    ct_img = ct.cpu().squeeze(0)[2]  # central slice

    fig, axs = plt.subplots(1,3, figsize=(15,5))

    axs[0].imshow(ct_img, cmap="gray")
    axs[0].set_title("CT")

    axs[1].imshow(ct_img, cmap="gray")
    axs[1].imshow(mk, alpha=0.5)
    axs[1].set_title("GT")

    axs[2].imshow(ct_img, cmap="gray")
    axs[2].imshow(pred_mask, alpha=0.5)
    axs[2].set_title("PRED")

    for ax in axs:
        ax.axis("off")

    if save:
        os.makedirs("results/visuals", exist_ok=True)
        path = f"results/visuals/{case_id}_{idx}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path
    else:
        plt.show()
