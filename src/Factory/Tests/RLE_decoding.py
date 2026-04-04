from pathlib import Path

import numpy as np
import pandas as pd

from src.Factory.dataset.build_dataset import parse_sizes_from_header


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