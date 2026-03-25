from pathlib import Path

from io_paths import main as paths
import pandas as pd

from src.Factory.load import DATA_MAIN_DIR

df_= paths()
rdata_file = Path(DATA_MAIN_DIR / '001-275' / 'done/origdata - Cropped.rdata')
ct_paths = DATA_MAIN_DIR / df_.index.to_series() / df_['DATA_0']
mask_paths = DATA_MAIN_DIR / df_.index.to_series() / df_['MASKSET_0']

complete_df = df_.loc[ct_paths.apply(lambda x: x.exists()) & mask_paths.apply(lambda x: x.exists())]

m = complete_df['DATA_0'].shape == complete_df['MASKSET_0'].shape
with open(rdata_file, 'rb') as f:
    header = f.read(100)

print(header)