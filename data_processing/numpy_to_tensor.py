'''Converts imputed NumPy arrays to PyTorch tensors.

Single channel (1 feature):  (T, H, W)    → (T, Lat, Lon)     3D tensor
Multi-feature (C features):   (T, C, H, W) → (T, C, Lat, Lon)  4D tensor

Dates stored as separate list in metadata dict alongside each tensor
'''

import torch
import numpy as np
from pathlib import Path

ARRAY_DIR = Path('earthdata-drought-crop-forecast/3d_numpy_arrays/soil_moisture_array')
OUT_DIR = ARRAY_DIR / '3d_tensor_time_lat_long'


def convert_single_channel():
    '''smap_sm_west_arsi_3day_imputed.npz → sm_tensor_T_Lat_Lon.pt'''
    src = ARRAY_DIR / 'smap_sm_west_arsi_3day_imputed.npz'
    data = np.load(src)

    sm_cube = data['sm_cube']       # (1324, 64, 64) float32
    dates = list(data['dates'])     # list of 'YYYY-MM-DD' strings

    tensor = torch.from_numpy(sm_cube)  # (T, Lat, Lon)

    out_path = OUT_DIR / 'sm_tensor_T_Lat_Lon.pt'
    torch.save({'tensor': tensor, 'dates': dates}, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f'Single-channel tensor:')
    print(f'  Shape: {tensor.shape}  (T={tensor.shape[0]}, Lat={tensor.shape[1]}, Lon={tensor.shape[2]})')
    print(f'  dtype: {tensor.dtype}')
    print(f'  Date range: {dates[0]} → {dates[-1]}')
    print(f'  Saved: {out_path.name} ({size_mb:.1f} MB)\n')


def convert_multifeature():
    '''smap_multifeature_africa_3day_imputed.npz → multifeature_tensor_T_C_Lat_Lon.pt'''
    src = ARRAY_DIR / 'smap_multifeature_africa_3day_imputed.npz'
    data = np.load(src)

    cube = data['cube']                       # (T, 8, H, W) float32
    dates = list(data['dates'])               # list of 'YYYY-MM-DD' strings
    feature_names = list(data['feature_names'])

    tensor = torch.from_numpy(cube)  # (T, C, Lat, Lon)

    out_path = OUT_DIR / 'multifeature_tensor_T_C_Lat_Lon.pt'
    torch.save({
        'tensor': tensor,
        'dates': dates,
        'feature_names': feature_names
    }, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f'Multi-feature tensor:')
    print(f'  Shape: {tensor.shape}  (T={tensor.shape[0]}, C={tensor.shape[1]}, Lat={tensor.shape[2]}, Lon={tensor.shape[3]})')
    print(f'  dtype: {tensor.dtype}')
    print(f'  Channels: {feature_names}')
    print(f'  Date range: {dates[0]} → {dates[-1]}')
    print(f'  Saved: {out_path.name} ({size_mb:.1f} MB)\n')


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    convert_multifeature()

    # Verify round-trip
    print('Verification:')
    for pt_file in sorted(OUT_DIR.glob('*.pt')):
        loaded = torch.load(pt_file, weights_only=False)
        t = loaded['tensor']
        print(f'  {pt_file.name}: shape={t.shape}, dtype={t.dtype}, NaN={torch.isnan(t).sum().item()}, dates={len(loaded["dates"])}')
