import hashlib
import numpy as np
import cv2

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


def hash_roi(mk_crop):
    """
    must have cleaner, deduplicates
    :param mk_crop:
    :return:
    """

    return hashlib.md5(mk_crop.tobytes()).hexdigest()