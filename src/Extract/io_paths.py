import re
from os import mkdir
from pathlib import Path
import pandas as pd

pd.set_option("display.width", 160)              # szerokość w znakach (ustaw pod swój terminal)
pd.set_option("display.max_columns", None)       # pokaż wszystkie kolumny
pd.set_option("display.max_rows", 200)           # ile wierszy pokazać zanim utnie
pd.set_option("display.max_colwidth", 120)       # max długość komórki (None = bez limitu, ale bywa masakra)
pd.set_option("display.expand_frame_repr", False) # nie łam DataFrame na kilka bloków kolumn
pd.set_option("display.show_dimensions", False)  # nie pokazuj [rows x cols] (opcjonalnie)
principal_keys = ["DATA_0","DATA_1","MASK_"]

RE_KEYS = re.compile(r'^\s*([A-Z][A-Z0-9_]*)\s*:\s*$')

DATA_MAIN_DIR = Path('D:/Kozmin/jadrakostnienia2024/jadrakostnienia2024')
JOB3_PATH = DATA_MAIN_DIR / 'jobX.job3'
VOXELIZED_FILE_PATH = Path('done/voxelizedSurfaces.rdata')

def extract_key(line: str) -> str:
    s = line.strip().lstrip("\ufeff")
    m = RE_KEYS.match(s)
    return m.group(1) if m else None

def read_lines_if_exists(p: Path) -> list[str] | None:
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()

def swap_done_to_done2(p: Path) -> Path:
    new_name = re.sub(r'(?i)^done1?(?=\.)', 'done2', p.name)
    return p.with_name(new_name)

def has_mask_key(lines: list[str]) -> bool:
    if not lines:
        return False
    return any((extract_key(l) or "").startswith("MASK_") for l in lines)
from pathlib import Path

def pick_done_file(sample_path: Path) -> tuple[Path, list[str] | None]:

    folder = sample_path.parent
    candidates = [
        folder / "done.arterydata",
        folder / "done2.arterydata",
        sample_path,
    ]

    uniq = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)

    for c in uniq:
        lines = read_lines_if_exists(c)
        if has_mask_key(lines):
            return c, lines

    return sample_path, read_lines_if_exists(sample_path)

def to_dict(path: Path):

    CTS_DIRS_PATHS = {}

    for artery in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not artery.strip():
            continue

        sample_path = Path(artery.strip())


        if not sample_path.is_absolute():
            sample_path = DATA_MAIN_DIR / sample_path

        lines = read_lines_if_exists(sample_path)

        if lines is None:
            print(f"[WARN] Brak pliku: {sample_path}")
            continue

        sample_path, lines = pick_done_file(sample_path)
        if lines is None:
            print(f"[WARN] Brak pliku: {sample_path}")
            continue

        sample_name = sample_path.parent.name


        CTS_DIRS_PATHS[sample_name] = {}
        CTS_DIRS_PATHS[sample_name]["VOXELIZED"] = VOXELIZED_FILE_PATH

        keys = []
        for i, line2 in enumerate(lines):
            k = extract_key(line2)
            if k:
                keys.append((i, k))

        if not keys:
            print(f"[WARN] Nie znaleziono żadnych kluczy KEY: w {sample_path}")
            continue

        for i, (key_index, key_name) in enumerate(keys):
            start = key_index + 1
            stop = keys[i + 1][0] if i + 1 < len(keys) else len(lines)
            raw_vals = lines[start:stop]
            lines_cleaned = [(v.strip("[]")) for v in raw_vals if v.strip()]
            CTS_DIRS_PATHS[sample_name][key_name.strip().strip(':')] = lines_cleaned

    return CTS_DIRS_PATHS

def unbracket(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x[0] if x else None

def get_principal(dct):
    rows = []
    for sample_nr, content in dct.items():
        d1 = unbracket(content.get("DATA_1"))
        d0 = unbracket(content.get("DATA_0"))
        if "Cropped_1" in Path(d1).name:
            d1 = d1
        elif "Cropped_1" in Path(d0).name:
            d1 = d0
        else:
            d1 = None
        def ms_from_data(rel_rdata: str | None):
            if not rel_rdata:
                return (None, None)
            p = Path(rel_rdata)
            ms = str(p.with_suffix(".maskSet"))
            msh = str(p.with_suffix(".maskSet.header"))
            return (ms, msh)

        ms0, ms0h = ms_from_data(d0)
        ms1, ms1h = ms_from_data(d1)

        rows.append({
            "sample_nr": sample_nr,
            # "DATA_0": d0,
            "DATA_1": d1,
            # "MASKSET_0": ms0,
            # "MASKSET_0_HDR": ms0h,
            "MASKSET_1": ms1,
            "MASKSET_1_HDR": ms1h,
            "VOXELIZED": content.get("VOXELIZED", None),
            "MASK_DESC_0": unbracket(content.get("MASK_0_0")),
            "MASK_DESC_1": unbracket(content.get("MASK_1_0")),
        })

    return pd.DataFrame(rows).set_index("sample_nr")



def main():
    cts_dict = to_dict(JOB3_PATH)
    print(cts_dict)
    principal = get_principal(cts_dict)
    principal.to_csv("principal_sharp_only.csv")

    return principal

if __name__ == "__main__":
    main()