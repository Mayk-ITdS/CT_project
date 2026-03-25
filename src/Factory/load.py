import hashlib
import logging
import shutil
import time
from pathlib import Path
import random
import segmentation_models_pytorch as smp
import torch.nn as nn
from tqdm import tqdm
import napari
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import faulthandler
import psutil, os
import cv2
from src.config.logger import setup_logging

# dataset_path = Path("Dataset")
#
# if dataset_path.exists():
#     shutil.rmtree(dataset_path)
#
# dataset_path.mkdir(parents=True, exist_ok=True)
#
# for f in ["meta_index_raw.csv", "meta_index_full.csv"]:
#     if Path(f).exists():
#         Path(f).unlink()

DATA_MAIN_DIR = Path(r"D:\Kozmin\jadrakostnienia2024\jadrakostnienia2024")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

setup_logging()

logger = logging.getLogger(__name__)
logger.debug(torch.__version__)
logger.debug(torch.cuda.is_available())
logger.debug(torch.cuda.get_device_name(0))

def parse_sizes_from_header(header_path: str | Path) -> tuple[int, int, int]:
    """gets x y z from header."""
    header_path = Path(header_path)

    lines = header_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if len(lines) < 2:
        raise ValueError(f"Header {header_path} does not contain at least two lines")

    for i, line in enumerate(lines):

        if 'SIZES:' in line:
            sizes_line = lines[i+1].strip()
            (x,y,z) = map(int, sizes_line.split())

            return x, y, z

    raise ValueError(f"SIZES not found in header {header_path}")

def parse_maskset_segments(maskset_header_path: str | Path):
    """Reads maskSet.header list of: (seg_size_ints, color_index)."""
    p = Path(maskset_header_path)
    lines = [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    seg_sizes = []
    colors = []
    sample = p.name
    i = 0
    while i < len(lines):
        if lines[i].lower() == "size:":
            seg_sizes.append(int(lines[i + 1]))
            i += 2
            continue
        if lines[i].upper() == "COLOR_INDEX:":
            colors.append(int(lines[i + 1]))
            i += 2
            continue
        i += 1
    if len(seg_sizes) != len(colors):
        raise ValueError(f"voxelizedSurfaces.header: sizes={len(seg_sizes)} colors={len(colors)} (powinno być równo)")
    frame = pd.DataFrame({"size": seg_sizes, "color": colors, "sample": sample})
    # frame.to_csv(f'mask_segs_colors_{sample}.csv', index=False)
    return seg_sizes, colors, frame

def load_rdata(ct_path: str | Path, rdata_header_path: str | Path, dtype=np.int16) -> np.ndarray:
    """Reads .rdata as (Z,Y,X) based on sizes from header CT headers"""
    x, y, z = parse_sizes_from_header(rdata_header_path)
    vox = x * y * z
    ct_path = Path(ct_path)

    arr = np.fromfile(ct_path, dtype=dtype)

    if arr.size != vox:
        raise ValueError(f"CT size mismatch: {ct_path.name} ma {arr.size} elem, oczekiwane {vox}")

    return arr.reshape((z, y, x))  # (Z,Y,X)

def load_mask_from_maskset(maskset_path: str | Path, maskset_header_path: str | Path) -> np.ndarray:
    """
    Decoding *.maskSet into dense labelmap (Z,Y,X).
    Segments are a RLE`s per volume:
      seg = [header2] + runs + [footer2]
      runs == vox  sum must be equal
      start_val = 1 (foreground)
    """

    maskset_path = Path(maskset_path)

    x, y, z = parse_sizes_from_header(maskset_header_path)
    vox = x * y * z

    seg_sizes, colors,_ = parse_maskset_segments(maskset_header_path) #dtype=np.uint32

    data = np.fromfile(maskset_path, dtype=np.uint32)

    if data.size != sum(seg_sizes):
        raise ValueError(f"maskSet size mismatch: {maskset_path.name} ma {data.size} intów, oczekiwane {sum(seg_sizes)}")

    label_vec = np.zeros(vox, dtype=np.uint16)

    pos = 0
    for i,(sz, label) in enumerate(zip(seg_sizes, colors)):
        seg = data[pos:pos + sz]
        pos += sz

        runs = seg[2:-2].astype(np.int64)

        if runs.sum() != vox:
            raise ValueError(f"RLE runs nie sumują się do vox dla segmentu label={label} (sum={runs.sum()} vox={vox})")

        idx = 0
        val = 1  # start od foreground
        for length in runs:
            if length and val == 1:
                label_vec[idx:idx + length] = label
            idx += length
            val ^= 1

    return label_vec.reshape((z, y, x))

def load_pair_from_ct(ct_rdata_path: str | Path,voxpath:str | Path) -> tuple[np.ndarray, np.ndarray]:

    ct_hdr = Path(str(ct_rdata_path) + ".header")
    ct_rdata_path = Path(ct_rdata_path)

    maskset_hdr = Path(str(voxpath) + ".header")

    maskset = Path(voxpath)

    if not ct_hdr.exists():
        raise FileNotFoundError(f"Brak CT header: {ct_hdr}")
    if not maskset.exists() or not maskset_hdr.exists():
        raise FileNotFoundError(f"Brak maskSet lub header: {maskset} / {maskset_hdr}")


    ct = load_rdata(ct_rdata_path, ct_hdr, dtype=np.int16)

    mk = load_mask_from_maskset(maskset, maskset_hdr)

    if ct.shape != mk.shape:
        raise AssertionError(f"CT shape mismatch: {ct.shape} != {mk.shape}")
    return ct, mk


def build_records(principal: pd.DataFrame) -> pd.DataFrame:
    """
    Building records for training.
    CT volume + its maskSet.
    """

    rows = []

    for _, row in principal.iterrows():

        sample_nr = row["sample_nr"]

        pairs = [
            ("DATA_0", "MASKSET_0"),
            ("DATA_1", "MASKSET_1")
        ]

        print("Sample:", sample_nr, "row:", row)

        for data_key, mask_key in pairs:

            rel_ct = row.get(data_key)
            rel_mask = row.get(mask_key)

            print("ct rel path:", rel_ct)
            print("mask rel path:", rel_mask)

            # skip missing
            if pd.isna(rel_ct) or pd.isna(rel_mask):
                print("skip missing path")
                continue

            ct_path = DATA_MAIN_DIR / sample_nr / rel_ct
            mask_path = DATA_MAIN_DIR / sample_nr / rel_mask

            ct_hdr = Path(str(ct_path) + ".header")
            mask_hdr = Path(str(mask_path) + ".header")

            ok = (
                ct_path.exists()
                and ct_hdr.exists()
                and mask_path.exists()
                and mask_hdr.exists()
            )

            rows.append({
                "sample_nr": sample_nr,
                "type": data_key,
                "ct": ct_path,
                "mk": mask_path,
                "ok": ok
            })

    print("Je retourne ton DF ;)")

    return pd.DataFrame(rows)

def show_3slices(ct, mk):

    z = ct.shape[0] // 2
    y = ct.shape[1] // 2
    x = ct.shape[2] // 2

    fig, ax = plt.subplots(2,3, figsize=(12,8))

    # axial
    ax[0,0].imshow(ct[z], cmap="gray")
    ax[0,0].set_title("CT axial")

    ax[0,1].imshow(mk[z], cmap="jet",interpolation="nearest")
    ax[0,1].set_title("Mask axial")

    ax[0,2].imshow(ct[z], cmap="gray")
    ax[0,2].imshow(mk[z], alpha=0.4, cmap="tab20",interpolation="nearest")
    ax[0,2].set_title("Overlay axial")

    # coronal
    ax[1,0].imshow(ct[:,y,:], cmap="gray")
    ax[1,0].imshow(mk[:,y,:], alpha=0.4, cmap="tab20",interpolation="nearest")
    ax[1,0].set_title("Overlay coronal")

    # sagittal
    ax[1,1].imshow(ct[:,:,x], cmap="gray")
    ax[1,1].imshow(mk[:,:,x], alpha=0.4, cmap="tab20",interpolation="nearest")
    ax[1,1].set_title("Overlay sagittal")

    # mask sagittal
    ax[1,2].imshow(mk[:,:,x], cmap="jet",interpolation="nearest")
    ax[1,2].set_title("Mask sagittal")

    plt.tight_layout()
    plt.show()

def build_class_map(records):
    print("Building class map...")

    unique_labels = set()

    for row in records:

        _,_,ct, mk = row
        print("Mam moje maskSet : ",np.unique(mk))
        labels = np.unique(mk).tolist()
        unique_labels.update(labels)

    unique_labels = sorted(unique_labels)

    class_map = {int(label): i for i, label in enumerate(unique_labels)}

    print("Class mapping done :", class_map)

    return class_map

def hash_roi(mk_crop):
    """
    must have cleaner, deduplicates
    :param mk_crop:
    :return:
    """

    return hashlib.md5(mk_crop.tobytes()).hexdigest()

def remap_labels(mask, class_map):
    max_label = int(max(class_map.keys()))

    lut = np.zeros(max_label + 1, dtype=np.uint16)

    for old, new in class_map.items():
        lut[int(old)] = new

    return lut[mask]

def arrange_difficulties(idx_list,p33,p66):
    easy, medium, hard = [], [], []

    for i in range(len(idx_list)):
        data = torch.load(idx_list[i], map_location="cpu")
        mk = data["mk"]

        fill = (mk > 0).sum().item() / mk.numel()

        if fill >= p66:
            easy.append(i)
        elif fill >= p33:
            medium.append(i)
        else:
            hard.append(i)

    return easy, medium, hard

def crop_clean(ct_stack, mk_slice, y0, y1, x0, x1 , target=128):
    ct_crop = ct_stack[:, y0:y1, x0:x1]
    mk_crop = mk_slice[y0:y1, x0:x1]
    # zabezpieczenie
    if ct_crop.shape[1] < 5 or ct_crop.shape[2] < 5:
        return None, None

    # resize CT (kanały osobno)
    ct_resized = np.stack([
        cv2.resize(ct_crop[i], (target, target), interpolation=cv2.INTER_LINEAR)
        for i in range(ct_crop.shape[0])
    ])

    # resize mask (NEAREST!)
    mk_resized = cv2.resize(
        mk_crop,
        (target, target),
        interpolation=cv2.INTER_NEAREST
    )

    return ct_resized, mk_resized

def pad_to_fixed(ct, mk, target=128):
    _, h, w = ct.shape

    pad_h = target - h
    pad_w = target - w

    if pad_h < 0 or pad_w < 0:
        return None, None

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    ct = np.pad(ct, ((0,0),
                     (pad_top, pad_bottom),
                     (pad_left, pad_right)),constant_values=-1000)

    mk = np.pad(mk, ((pad_top, pad_bottom),
                     (pad_left, pad_right)))

    return ct, mk

def overlap_too_big(a, b, threshold=0.5):
    y0 = max(a[0], b[0])
    y1 = min(a[1], b[1])
    x0 = max(a[2], b[2])
    x1 = min(a[3], b[3])

    if y1 <= y0 or x1 <= x0:
        return False # no overlap

    inter_area = (y1 - y0) * (x1 - x0)

    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    return inter_area / min(area_a, area_b) > threshold

def compute_roi(mask_slice,min_side=80 ,max_side=196):
    from scipy.ndimage import label
    rois = []
    for cls in np.unique(mask_slice):
        if cls == 0:
            continue
        label_map, number = label(mask_slice == cls)

        for component in range(1, number + 1):

            ys, xs = np.where(label_map == component)

            if not ys.size:
                raise ValueError("No pixels in component")

            if len(ys) < 20:
                continue

            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2

            bbox_h = y_max - y_min +1
            bbox_w = x_max - x_min +1

            roi_side = max(bbox_h, bbox_w)
            roi_side = int(roi_side * 1.5)
            roi_side = max(roi_side, bbox_h + 10, bbox_w + 10)
            roi_side = int(np.clip(roi_side, min_side, max_side))

            half = roi_side // 2

            y0 = cy - half
            y1 = cy + half
            x0 = cx - half
            x1 = cx + half

            rois.append((y0, y1, x0, x1))

    return rois


def compute_clamp(y0, y1, x0, x1, H, W):
    roi_h = y1 - y0
    roi_w = x1 - x0

    # center-based clamp
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2

    y0 = cy - roi_h // 2
    y1 = y0 + roi_h

    x0 = cx - roi_w // 2
    x1 = x0 + roi_w

    # teraz clamp FINALNY
    if y0 < 0:
        y0 = 0
        y1 = roi_h
    if x0 < 0:
        x0 = 0
        x1 = roi_w
    if y1 > H:
        y1 = H
        y0 = H - roi_h
    if x1 > W:
        x1 = W
        x0 = W - roi_w

    return y0, y1, x0, x1

def slice_records(df_paths, class_map=None,collect_only=False):
    records = []
    case_ids = set()
    eda_logs = []
    meta_rows= []
    seen = set()
    for row in df_paths.itertuples():
        ct_path = row.ct
        mk_path = row.mk

        ct, mk = load_pair_from_ct(Path(ct_path), Path(mk_path))
        case_ID = Path(ct_path).name
        case_ids.add(case_ID)

        # ================= AXIAL =================
        step = random.choice([2, 3, 4])
        for z in range(2, ct.shape[0] - 2, step):

            if np.sum(mk[z] > 0) == 0:
                continue

            mk_slice = mk[z]
            ct_stack = ct[z - 2:z + 3,:,:]

            rois = compute_roi(mk_slice)

            filtered = []

            for roi in rois:
                keep = True

                for f in filtered:
                    if overlap_too_big(roi, f):
                        keep = False
                        break
                if keep:
                    filtered.append(roi)

            for roi in filtered:

                y0, y1, x0, x1 = roi

                y0, y1, x0, x1 = compute_clamp(y0, y1, x0, x1, *mk_slice.shape)

                ct_crop, mk_crop = crop_clean(ct_stack, mk_slice, y0, y1, x0, x1)

                if ct_crop is None:
                    continue

                if random.random() < 0.01:
                    print("SHAPE:", ct_crop.shape, mk_crop.shape)

                assert ct_crop.shape == (5, 128, 128), f"{ct_crop.shape}"
                assert mk_crop.shape == (128, 128)
                h = hash_roi(mk_crop)

                if h in seen:
                    continue
                seen.add(h)

                mask_pixels = (mk_crop > 0).sum().item()
                roi_pixels = mk_crop.size

                coverage = (mk_crop > 0).sum() / roi_pixels

                print(f"coverage: {coverage:.3f}")

                if coverage < 0.03:
                    continue

                assert ct_crop.shape[1:] == mk_crop.shape, "SHAPE MISMATCH"
                print("CROP SHAPE:", ct_crop.shape)
                print("MASK PIXELS:", (mk_crop > 0).sum())

                mk_crop = mk_crop if collect_only==True else remap_labels(mk_crop, class_map)

                if (mk_crop > 0).sum() == 0:
                    continue

                records.append((case_ID, 'axial', ct_crop, mk_crop))

                print(ct_crop.shape)
                print(mk_crop.shape)
                print(np.unique(mk_crop))
                print( coverage)
                print("CT:", ct_crop.shape)
                print("MK:", mk_crop.shape)
                print("coverage:", coverage)
                print(f"""
                        
                        Real Fill/Coverage: {coverage}
                        """)

                eda_logs.append({
                    'case_id': case_ID,
                    "ct_path": str(ct_path),
                    "mk_path": str(mk_path),
                    "plane": "axial",
                    "index": z,
                    "roi": (y0, y1, x0, x1),
                    "real_fill": coverage,
                    'classes': sorted(np.unique(mk_crop).tolist()),
                    'mask_pixels': mask_pixels,
                    'roi_pixels': roi_pixels,
                    'ct_crop_shape':ct_crop.shape,
                    'mask_crop_shape':mk_crop.shape,
                })
                print("Axial before save collect value: ",collect_only)
                save_path = None if collect_only else Path(f'Dataset/{case_ID}_axial_{z}.pt')

                if not collect_only:
                    torch.save({
                        "ct": torch.from_numpy(ct_crop).float(),
                        "mk": torch.from_numpy(mk_crop).long()
                     }, save_path)

                meta_rows.append({
                    "file": str(save_path),
                    "ct_path": str(ct_path),
                    "mk_path": str(mk_path),
                    "case_id": case_ID,
                    "plane": 'axial',
                    "real_fill":coverage,
                    "roi":(y0, y1, x0, x1),
                    "slice_idx": z,
                    'classes': np.unique(mk_crop).tolist(),
                    'mask_pixels': mask_pixels,
                    'roi_pixels': roi_pixels,
                    'ct_crop_shape': ct_crop.shape,
                    'mask_crop_shape': mk_crop.shape,
                })

        print(f"Axial finished for {case_ID}")

        # ================= CORONAL =================
        print(f"Starting Coronal for {case_ID}")
        step = random.choice([2, 3, 4])
        for y in range(2, ct.shape[1] - 2, step):
            ct_stack = ct[:, y - 2:y + 3, :]
            ct_stack = ct_stack.transpose(1, 0, 2)
            mask_slice = mk[:, y, :]
            if np.sum(mask_slice > 0) == 0:
                continue

            filtered = []
            rois = compute_roi(mask_slice)
            for roi in rois:
                keep = True

                for f in filtered:
                    if overlap_too_big(roi, f):
                        keep = False
                        break
                if keep:
                    filtered.append(roi)

            for roi in filtered:
                y0, y1, x0, x1 = roi

                y0, y1, x0, x1 = compute_clamp(y0, y1, x0, x1, *mk_slice.shape)

                ct_crop, mk_crop = crop_clean(ct_stack, mk_slice, y0, y1, x0, x1)

                if ct_crop is None:
                    continue

                if random.random() < 0.01:
                    print("SHAPE:", ct_crop.shape, mk_crop.shape)

                assert ct_crop.shape == (5, 128, 128), f"{ct_crop.shape}"
                assert mk_crop.shape == (128, 128)

                h = hash_roi(mk_crop)
                if h in seen:
                    continue
                seen.add(h)

                assert ct_crop.shape[1:] == mk_crop.shape, "SHAPE MISMATCH"

                mask_pixels = (mk_crop > 0).sum().item()
                roi_pixels = mk_crop.size
                coverage = (mk_crop > 0).sum() / roi_pixels

                if coverage < 0.03:
                    continue

                mk_crop = mk_crop if collect_only==True else remap_labels(mk_crop, class_map)
                if (mk_crop > 0).sum() == 0:
                    continue

                records.append((case_ID, 'coronal', ct_crop, mk_crop))

                print("CROP SHAPE:", ct_crop.shape)
                print("COVERAGE:", coverage)

                save_path = None if collect_only else Path(f'Dataset/{case_ID}_coronal_{y}.pt')
                if not collect_only:
                    torch.save({
                        "ct": torch.from_numpy(ct_crop).float(),
                        "mk": torch.from_numpy(mk_crop).long()
                    }, save_path)

                meta_rows.append({
                    "file": str(save_path),
                    "ct_path": str(ct_path),
                    "mk_path": str(mk_path),
                    "case_id": case_ID,
                    "plane": 'coronal',
                    "real_fill": coverage,
                    "roi": (y0, y1, x0, x1),
                    "slice_idx": y,
                    "classes": np.unique(mk_crop).tolist(),
                    'mask_pixels': mask_pixels,
                    'roi_pixels': roi_pixels,
                    'ct_crop_shape': ct_crop.shape,
                    'mask_crop_shape': mk_crop.shape,
                })

        # ================= SAGITAL =================
        step = random.choice([2, 3, 4])
        for x in range(2, ct.shape[2] - 2, step):

            mask_slice = mk[:, :, x]
            if np.sum(mask_slice > 0) == 0:
                continue

            ct_stack = ct[:,:,x - 2:x + 3].transpose(2, 0, 1)

            filtered = []
            rois = compute_roi(mask_slice)
            for roi in rois:
                keep = True

                for f in filtered:
                    if overlap_too_big(roi, f):
                        keep = False
                        break
                if keep:
                    filtered.append(roi)

            for roi in filtered:
                y0, y1, x0, x1 = roi

                y0, y1, x0, x1 = compute_clamp(y0, y1, x0, x1, *mask_slice.shape)

                ct_crop, mk_crop = crop_clean(ct_stack, mask_slice, y0, y1, x0, x1)

                if ct_crop is None:
                    continue

                if random.random() < 0.01:
                    print("SHAPE:", ct_crop.shape, mk_crop.shape)

                assert ct_crop.shape == (5, 128, 128), f"{ct_crop.shape}"
                assert mk_crop.shape == (128, 128)

                h = hash_roi(mk_crop)
                if h in seen:
                    continue
                seen.add(h)

                assert ct_crop.shape[1:] == mk_crop.shape, "SHAPE MISMATCH"

                roi_pixels = mk_crop.size
                coverage = (mk_crop > 0).sum() / roi_pixels

                if coverage < 0.03:
                    continue

                mk_crop = mk_crop if collect_only==True else remap_labels(mk_crop, class_map)
                if (mk_crop > 0).sum() == 0:
                    continue

                records.append((case_ID, 'sagital', ct_crop, mk_crop))

                logger.debug(f"Crop shape: {ct_crop.shape}")
                logger.debug(f"Mask pixels: {(mk_crop > 0).sum()}")
                logger.debug(f"Coverage: {coverage:.3f}")
                logger.debug(f"CT shape: {ct_crop.shape}")
                logger.debug(f"Mask shape: {mk_crop.shape}")
                logger.debug(f"Mask labels uniques: {np.unique(mk_crop)}")

                logger.debug(
                    f"""
                    REAL FILL: {coverage:.3f}
                    """
                )

                save_path = None if collect_only else Path(f'Dataset/{case_ID}_sagital_{x}.pt')

                if not collect_only:
                    torch.save({
                        "ct": torch.from_numpy(ct_crop).float(),
                        "mk": torch.from_numpy(mk_crop).long()
                    }, save_path)

                meta_rows.append({
                    "file": str(save_path),
                    "ct_path": str(ct_path),
                    "mk_path": str(mk_path),
                    "case_id": case_ID,
                    "plane": 'sagital',
                    "slice_idx": x,
                    "real_fill": coverage,
                    "roi_side": (y0, y1, x0 ,x1),
                    'classes': np.unique(mk_crop).tolist(),
                    'mask_pixels': "unknown",
                    'roi_pixels': roi_pixels,
                    'ct_crop_shape': ct_crop.shape,
                    'mask_crop_shape': mk_crop.shape,
                })
                eda_logs.append({
                    "ct_path": str(ct_path),
                    "mk_path": str(mk_path),
                    "plane": "sagital",
                    "index": x,
                    "roi": (y0, y1, x0, x1),
                    'classes': sorted(np.unique(mk_crop).tolist()),
                })
    meta_path = Path("meta_index_raw.csv") if collect_only else Path("meta_index_full.csv")
    eda_path = Path("eda_analysis_before.csv") if collect_only else Path("eda_analysis_after.csv")

    df = pd.DataFrame(meta_rows)
    df.to_csv(meta_path, index=False)
    eda_df = pd.DataFrame(eda_logs)
    eda_df.to_csv(eda_path, index=False)

    logger.info("Slicing ended")
    logger.info(f"Cases no: {len(case_ids)}")
    logger.debug("Meta saved as ",meta_path, "")
    logger.debug(f"EDA saved as {eda_path}")

    return records

def debug_crop(ct_crop, mk_crop, center_idx=2):
    import matplotlib.pyplot as plt
    import numpy as np

    print("="*40)
    print("CT shape:", ct_crop.shape)
    print("Mask shape:", mk_crop.shape)
    print("Classes:", np.unique(mk_crop))
    print("Coverage:", (mk_crop > 0).sum() / mk_crop.size)

    fig, ax = plt.subplots(1, 4, figsize=(14,4))

    ax[0].imshow(ct_crop[center_idx - 2], cmap="gray")
    ax[0].set_title("z-2")

    ax[1].imshow(ct_crop[center_idx], cmap="gray")
    ax[1].set_title("center")

    ax[2].imshow(ct_crop[center_idx + 2], cmap="gray")
    ax[2].set_title("z+2")

    ax[3].imshow(ct_crop[center_idx], cmap="gray")
    ax[3].imshow(mk_crop, alpha=0.4, cmap="tab20")
    ax[3].set_title("overlay")

    for a in ax:
        a.axis("off")

    plt.suptitle("CT sliced & preprocessed")
    plt.tight_layout()
    plt.show()

class DifficultyBatchSampler(torch.utils.data.BatchSampler):

    def __init__(self, easy, medium, hard, batch_size=4):
        self.easy = easy
        self.medium = medium
        self.hard = hard
        self.batch_size = batch_size
        self.length = len(easy) + len(medium) + len(hard)

    def __iter__(self):

        pool = []
        if self.easy:
            pool += self.easy
        if self.medium:
            pool += self.medium
        if self.hard:
            pool += self.hard

        # bias easy
        if self.easy:
            pool += self.easy

        for _ in range(self.length // self.batch_size):

            batch = []

            while len(batch) < self.batch_size:
                batch.append(random.choice(pool))

            yield batch

    def __len__(self):
        return self.length // self.batch_size

class CTDataset(torch.utils.data.Dataset):
    def __init__(self, files, class_map=None):
        self.files = files
        self.class_map = class_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx] if isinstance(idx, int) else idx
        data = torch.load(path, map_location="cpu")

        ct = data["ct"]
        mk = data["mk"]

        ct = ct.float()
        mk = mk.long()

        assert ct.shape[0] == 5, f"CT shape wrong: {ct.shape}"

        ct = (ct - ct.mean()) / (ct.std() + 1e-6)

        return ct, mk


def pipeline():

    time.sleep(2.5)
    logger.info(
        "=================================== Pipeline Start =================================================")

    torch.backends.cudnn.benchmark = True
    process = psutil.Process(os.getpid())
    logger.debug(f"RAM {process.memory_info().rss / 1024 ** 2:.1f} MB")
    faulthandler.enable()
    HERE = Path(__file__).parent
    principal_path = HERE / "principal.csv"
    try:
        records = pd.read_csv(principal_path)
        records = build_records(records)
        records_ok = records[records["ok"]].reset_index(drop=True)
        logger.info(f"Built records columns: {records_ok.columns}")
        logger.info(f"Main dataset length: {len(records_ok)}")
    except Exception:
        logger.error("Pipeline crashed at build records...",exc_info=True)
        raise
    try:
        records = slice_records(records_ok,collect_only=True)
    except Exception:
        logger.error("Pipeline crashed at slice records...",exc_info=True)
        raise
    try:
        class_map = build_class_map(records)
    except Exception:
        logger.error("Pipeline crashed at build class map...",exc_info=True)
        raise
    try:
        slice_records(records_ok, class_map,collect_only=False)
        logger.info("Pipeline almost finished successfully...")
    except Exception:
        logger.error("Pipeline crashed at slice records final ...",exc_info=True)
        raise
    try:
        meta = pd.read_csv("meta_index_full.csv")
        p33 = meta["real_fill"].quantile(0.33)
        p66 = meta["real_fill"].quantile(0.66)
        case_ids = meta['case_id'].unique()

        if len(case_ids) == 0:
            logger.error("No case_ids found!")
            raise ValueError("Empty dataset")

        logger.info(f"IQR coverage: p33={p33:.3f}, p66={p66:.3f}")
        logger.info(f"Pipeline finally finished successfully...")
    except Exception:
        logger.error("Pipeline crashed at meta index...",exc_info=True)
        raise

    logger.info('==============================================Load Preprocessed Data===========================================')

    train_ids, test_ids = train_test_split(case_ids, test_size=0.2,random_state=42)
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=0.1,
        random_state=42
    )
    train_files_2_5D = meta[meta['case_id'].isin(train_ids)]['file'].tolist()
    val_files_2_5D = meta[meta['case_id'].isin(val_ids)]['file'].tolist()
    test_files_2_5D = meta[meta['case_id'].isin(test_ids)]['file'].tolist()

    print('Jestem tuz przed zbieraniem train indexow')
    print("Jestem przed arrange diffs")
    easy_train,medium_train,hard_train = arrange_difficulties(train_files_2_5D,p33,p66)

    print("TRAIN DATASET LENGTH:", len(train_files_2_5D))
    print('Test ids : ',train_ids)
    print("easy:", len(easy_train))
    print("medium:", len(medium_train))
    print("hard:", len(hard_train))
    sampler_train = DifficultyBatchSampler(easy_train, medium_train, hard_train)

    train_2_5D_dataset = CTDataset(train_files_2_5D,[])
    val_dataset = CTDataset(val_files_2_5D,[])

    print("==================================== Now starting DataLoaders =========================================")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_2_5D_dataset,
        batch_sampler=sampler_train,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(val_dataset,batch_size=4,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    test_loader = DataLoader(test_files_2_5D, batch_size=4)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=5,
        classes=55
    ).to(device,non_blocking=True)

    score = model.eval()

    def dice_score(pred, target):

        pred = torch.argmax(pred, dim=1)

        dice_scores = []
        classes = torch.unique(target)
        for cls in classes:
            if cls <= 1:
                continue          # pomijamy tło
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ce_loss = nn.CrossEntropyLoss(ignore_index=1)
    dice_loss = smp.losses.TverskyLoss(mode='multiclass',ignore_index=1,alpha=0.3,beta=0.7)

    def custom_lossfn(y_pred,y_true):
        return 0.5 * ce_loss(y_pred,y_true) + 0.5 * dice_loss(y_pred,y_true)

    for epoch in range(30):
        pred_true_dict = {
            "GT classes":set(),
            "Predicted classes":set(),
        }
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for ct, mask in pbar:

            ct = ct.to(device,non_blocking=True)
            mask = mask.to(device,non_blocking=True)

            optimizer.zero_grad()

            pred = model(ct)
            loss = custom_lossfn(pred,mask)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred_true_dict['GT classes'].update(torch.unique(mask).cpu().numpy())
            pred_true_dict['Predicted classes'].update(torch.unique(torch.argmax(pred,dim=1)).cpu().numpy())

            pbar.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)
        print(f"Epoch {epoch} Train Loss: {train_loss}")
        print(f"Epoch {epoch} GT classes: {pred_true_dict['GT classes']}")
        print(f"Epoch {epoch} Predicted classes: {pred_true_dict['Predicted classes']}")
        model.eval()

        val_loss = 0
        dice_total = 0
        all_predval_classes = set()
        all_trueval_classes = set()
        with torch.no_grad():

            for ct, mask in tqdm(val_loader):
                ct = ct.to(device)
                mask = mask.to(device)

                pred = model(ct)
                pred_mask = torch.argmax(pred, dim=1)
                loss = custom_lossfn(pred,mask)
                val_loss += loss.item()

                dice_total += dice_score(pred, mask)

                all_trueval_classes.update(torch.unique(mask).cpu().numpy())
                all_predval_classes.update(torch.unique(pred_mask).cpu().numpy())

            val_loss /= len(val_loader)
            dice_avg = dice_total / len(val_loader)


            print("GT val classes:", sorted(all_trueval_classes))
            print("Pred val classes:", sorted(all_predval_classes))

            print(f"Epoch {epoch} Val Loss: {val_loss} Dice: {dice_avg}")
        torch.save(model.state_dict(), f"model_last_v.pth")


if __name__ == '__main__':
    try:
        start = time.time()
        pipeline()
        end = time.time()
    except Exception as e:
        print(e)







# print("tu printuje maski",records_ok['mask'])
# mismatched = []
# slices_with_masks = []
# no_zero_voxels_per_mask = []
# print(records_ok)
# print('Unique test')
# print(records.groupby("sample_nr")["ct"].nunique().value_counts())
# print("rekordy ok:", len(records_ok))
#
# print(slices_with_masks)
# print(no_zero_voxels_per_mask)
#
#
# for i,ct_path in enumerate(records_ok["ct"]):
#       ct, mk = load_pair_from_ct(Path(ct_path))
#       if i == 0:
#           view_one3D(ct, mk)
#       if ct.shape != mk.shape:
#           mismatched.append(Path(ct))
#           raise ValueError(f"CT shape mismatch: {ct.shape} != {mk.shape}")
#       slice_count = 0
#       classes, counts = np.unique(mk, return_counts=True)
#
#       print(f"\n{ct_path.name}")
#       for c, n in zip(classes, counts):
#           print(f"class {c}: {n}")
#       for z in range(mk.shape[0]):
#         if np.any(mk[z] > 0):
#             slice_count += 1
#       mask_voxels = np.count_nonzero(mk)
#       ratio = mask_voxels / mk.size
#       # show_3slices(i,ct,mk)
#       no_zero_voxels_per_mask.append(f"{ct_path.name}, no_zero voxels in mask: {mask_voxels}, ratio: {ratio}")
#       slices_with_masks.append(f"{ct_path.name}, slices with mask:, {slice_count}")
#       print(ct.shape, mk.shape)
#       print('Unique classes :',ct.shape, mk.shape, np.unique(mk)[:30])
#       count_vox_per_class(mk)
#       # outlieres verification
#
#       print(f'Ct min-max from path:{ct_path} : {ct.min()}, {ct.max()}')
#       print(f'percentiles :',np.percentile(ct, [1, 5, 50, 95, 99]))
#
# #all classes
#
# plt.hist(ratios, bins=20)
# plt.title("Mask voxel ratio distribution")
# plt.show()
#
# for ct_path in records_ok["ct"]:
#     ct, mk = load_pair_from_ct(Path(ct_path))
#     ratio = np.count_nonzero(mk) / mk.size
#     ratios.append(ratio)
#
