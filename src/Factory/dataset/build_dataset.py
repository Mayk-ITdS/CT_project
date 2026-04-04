import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from numpy import random
import cv2
import logging
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "src/Factory/dataset/Dataset"
DATASET_DIR.mkdir(exist_ok=True)
DATA_MAIN_DIR = Path(r"D:\Kozmin\jadrakostnienia2024\jadrakostnienia2024")
HERE = Path(__file__).parent.parent
principal_path = HERE / "principal_sharp_only.csv"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def load_rdata(ct_path: str | Path, rdata_header_path: str | Path, dtype=np.int16) -> np.ndarray:

    """Reads .rdata as (Z,Y,X) based on sizes from header CT headers"""

    x, y, z = parse_sizes_from_header(rdata_header_path)
    vox = x * y * z
    ct_path = Path(ct_path)

    arr = np.fromfile(ct_path, dtype=dtype)

    if arr.size != vox:
        raise ValueError(f"CT size mismatch: {ct_path.name} ma {arr.size} elem, oczekiwane {vox}")

    return arr.reshape((z, y, x))  # (Z,Y,X)

def load_mask_voxi(mask_path: str | Path) -> np.ndarray:

    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    hdr_path = Path(str(mask_path) + ".header")

    if not hdr_path.exists():
        raise FileNotFoundError(f"Mask header not found: {hdr_path}")

    try:
        x, y, z = parse_sizes_from_header(hdr_path)
        expected = x * y * z

        mask = np.fromfile(mask_path, dtype=np.uint16)

        if mask.size != expected:
            raise ValueError(
                f"Mask size mismatch for {mask_path}: got {mask.size}, expected {expected}"
            )

        return mask.reshape((z, y, x))

    except Exception as e:
        raise RuntimeError(f"Failed to load mask {mask_path}: {e}")

def load_pair_from_ct(ct_rdata_path: str | Path,voxpath:str | Path) -> tuple[np.ndarray, np.ndarray]:

    ct_hdr = Path(str(ct_rdata_path) + ".header")
    ct_rdata_path = Path(ct_rdata_path)

    maskVox_hdr = Path(str(voxpath) + ".header")

    maskVox = Path(voxpath)

    if not ct_hdr.exists():
        raise FileNotFoundError(f"Brak CT header: {ct_hdr}")
    if not maskVox.exists() or not maskVox_hdr.exists():
        raise FileNotFoundError(f"Brak maskSet lub header: {maskVox} / {maskVox_hdr}")

    ct = load_rdata(ct_rdata_path, ct_hdr, dtype=np.int16)

    mk = load_mask_voxi(maskVox)

    if ct.shape != mk.shape:
        raise AssertionError(f"CT shape mismatch: {ct.shape} != {mk.shape}")
    return ct, mk

def build_records(principal: pd.DataFrame) -> pd.DataFrame:
    """
    Building records for training.
    CT volume + its maskSet.
    """

    rows = []
    print(principal)
    if principal.empty:
        logger.error("Pipeline received empty principal build records [ok]!...", exc_info=True)
        raise ValueError("Principals empty ")

    for _, row in principal.iterrows():

        sample_nr = row["sample_nr"]

        pairs = [
            ("DATA_1", "VOXELIZED")
        ]


        # mk = np.fromfile(DATA_MAIN_DIR / sample_nr['sample_nr'] / sample_nr['VOXELIZED'], dtype=np.uint16)
        for data_key, mask_key in pairs:

            rel_ct = row.get(data_key)
            rel_mask = row.get(mask_key)

            # skip missing
            if pd.isna(rel_ct) or pd.isna(rel_mask):
                print("skip missing path")
                continue

            ct_path = DATA_MAIN_DIR / sample_nr /  rel_ct
            mask_path = DATA_MAIN_DIR / sample_nr / rel_mask

            ct_hdr = Path(str(ct_path) + ".header")
            mask_hdr = Path(str(mask_path) + ".header")

            ok = (
                ct_path.exists()
                and ct_hdr.exists()
                and mask_path.exists()
                and mask_hdr.exists()
            )
            # if not ok:
            #
            #     logger.error("Pipeline crashed at build records [ok]!...", exc_info=True)
            #     raise ValueError('Not wsyckie ok')
            rows.append({
                "sample_nr": sample_nr,
                "type": data_key,
                "ct": ct_path,
                "mk": mask_path,
                "ok": ok
            })

    print("Je retourne ton DF ;)")

    return pd.DataFrame(rows)
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
def hash_roi(mk_crop):
    """
    must have cleaner, deduplicates
    :param mk_crop:
    :return:
    """

    return hashlib.md5(mk_crop.tobytes()).hexdigest()
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

            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2

            bbox_h = y_max - y_min +1
            bbox_w = x_max - x_min +1

            roi_side = int(max(bbox_h, bbox_w) * 1.3)
            roi_side = np.clip(roi_side, 16, 196)

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

def slice_records(df_paths):
    import gc
    records = []
    case_ids = set()
    eda_logs = []
    meta_rows= []
    seen = set()
    success = 0
    skipped = 0
    fail = 0
    for row in tqdm(df_paths.itertuples(),total=len(df_paths)):

        ct_path = row.ct
        mk_path = row.mk
        try:
            ct, mk = load_pair_from_ct(Path(ct_path), Path(mk_path))
        except Exception as e:
            print("FAILED SAMPLE:", ct_path)
            fail +=1
            continue

        if ct.size == 0 or mk.size == 0:
            Exception('Popraw cos z maskami')
        case_ID = Path(ct_path).name
        case_ids.add(case_ID)

        # ================= AXIAL =================

        for z in range(2, ct.shape[0] - 2):

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

            for roi_id, roi in enumerate(filtered):

                y0, y1, x0, x1 = roi

                y0, y1, x0, x1 = compute_clamp(y0, y1, x0, x1, *mk_slice.shape)

                ct_crop, mk_crop = crop_clean(ct_stack, mk_slice, y0, y1, x0, x1)
                if ct_crop is None or mk_crop is None:
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

                assert ct_crop.shape[1:] == mk_crop.shape, "SHAPE MISMATCH"

                logging.info(np.unique(mk_crop).astype(int))

                if mk_crop is None:
                    print("mk_crop is None → skipping")
                    continue

                records.append((case_ID, 'axial', ct_crop, mk_crop))

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

                save_path = Path(DATASET_DIR / f'{case_ID}_axial_{z}_{roi_id}.pt')
                try:
                    torch.save({
                        "ct": torch.from_numpy(ct_crop).float(),
                        "mk": torch.from_numpy(mk_crop).long()
                     }, save_path)
                    success += 1
                except Exception as e:
                    print("Save failed")
                    fail +=1

                meta_rows.append({
                    "file": str(save_path),
                    "ct_path": str(ct_path),
                    "mk_path": str(mk_path),
                    "case_id": case_ID,
                    "plane": 'axial',
                    "real_fill":coverage,
                    "roi":(y0, y1, x0, x1),
                    'roi_id':roi_id,
                    "slice_idx": z,
                    'classes': np.unique(mk_crop).tolist(),
                    'mask_pixels': mask_pixels,
                    'roi_pixels': roi_pixels,
                    'ct_crop_shape': ct_crop.shape,
                    'mask_crop_shape': mk_crop.shape,
                })

        print(f"Axial finished for {case_ID}")
        del ct, mk
        gc.collect()

    meta_path = ROOT / "src/Factory/meta_index_full.csv"
    eda_path = ROOT / "src/Factory/eda_analysis_full.csv"
    if len(meta_rows) == 0:
        print("WARNING: meta_rows is empty!")
    df = pd.DataFrame(meta_rows)
    df.to_csv(meta_path, index=False)
    eda_df = pd.DataFrame(eda_logs)
    eda_df.to_csv(eda_path, index=False)

    logging.info("Slicing ended")
    logging.info(f"Cases no: {len(case_ids)}")
    logging.debug("Meta saved as ",meta_path, "")
    logging.debug(f"EDA saved as {eda_path}")
    print(f"""
    SUCCESS: {success}
    FAILED: {fail}
    SKIPPED: {skipped}
    """)
    total = success + fail + skipped
    print(f"TOTAL PROCESSED: {total}")
    return records

def build_dataset(path):
    logger.info("Building dataset...")
    records = pd.read_csv(path)
    records = build_records(records)
    records_ok = records[records["ok"]]
    for i,row in enumerate(records_ok.itertuples()):

        ct, mk = load_pair_from_ct(row.ct, row.mk)
        print("ct.shape:", ct.shape,'mk.shape:', mk.shape)
        print('classes :',np.unique(mk))
        print('sample_nr :',row.sample_nr,'ct_dtype',ct.dtype, 'mask_dtype',mk.dtype)




    #slice_records(records_ok)

if __name__ == '__main__':
    try:
        build_dataset(principal_path)
    except Exception:
        import traceback
        traceback.print_exc()
        logger.error("Dataset build failed", exc_info=True)
        raise
