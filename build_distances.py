"""
Region centroids and centroid distances for the M5 distance prior (Protocol and Model Spec,
sections 3 and 6).

    python build_distances.py          # writes data_processed/centroids_ito36.npz

Inputs (frozen, listed in hashes.sha256):
  data_raw/ito_68_atlas/vfb_68_2018-11-03_5.nii.gz       Ito atlas label volume (the spec's mask)
  data_raw/ito_68_atlas/Original_Index_panda_full.csv    atlas label -> region name
  data_processed/fly_ito36.npz                           36-region order (build_dataset.load)

Recipe:
  * Each of the 36 structural-matrix regions maps to its Ito labels exactly as in
    build_dataset.build (displayName match; IB = IB_R + IB_L).
  * Centroid = mean world coordinate (the volume's affine applied to voxel centres) of every
    voxel carrying one of the region's labels. For IB that is the centroid of the union of the
    IB_R and IB_L voxels, i.e. voxel-weighted.
  * D[i, j] = Euclidean distance between centroids i and j, in the affine's units (the header
    says mm; the voxel size is about 3, which matches micrometres for a fly brain). M5 uses
    exp(-D / l) with l = median off-diagonal D (prior.distance_wiring), so the unit cancels.

Output data_processed/centroids_ito36.npz: regions (36,), labels_json, voxel_counts (36,),
centroids (36, 3), D (36, 36), affine (4, 4), meta_json (inputs with SHA-256, units, recipe).
Freeze it (add to hashes.sha256) before any run that uses M5; load it only with load_distances().
"""
import argparse
import json
import os

import numpy as np

from build_dataset import INC37, display, load

MASK = 'data_raw/ito_68_atlas/vfb_68_2018-11-03_5.nii.gz'
CSV = 'data_raw/ito_68_atlas/Original_Index_panda_full.csv'
NPZ = 'data_processed/fly_ito36.npz'
OUT = 'data_processed/centroids_ito36.npz'


def region_labels(names, csv_path):
    """Structural-matrix region -> list of (Ito name, atlas label), as in build_dataset.build."""
    import pandas as pd
    idx = pd.read_csv(csv_path, keep_default_na=False); idx.columns = ['name', 'num', 'i0']
    num_of = idx.set_index('name')['num']
    fmap = {s: [n for n in INC37 if display(n) == s] for s in names}
    fmap['IB'] = ['IB_R', 'IB_L']
    assert sorted(sum(fmap.values(), [])) == sorted(INC37), "region mapping does not cover the 37 authors' regions exactly"
    return {s: [(n, int(num_of[n])) for n in fmap[s]] for s in names}


def build(mask_path=MASK, csv_path=CSV, npz_path=NPZ):
    import nibabel as nib
    from integrity import sha256
    _, names, _, _ = load(npz_path)
    labels = region_labels(names, csv_path)
    img = nib.load(mask_path)
    m = np.asarray(np.squeeze(img.get_fdata()), 'uint16')
    aff = np.asarray(img.affine, np.float64)
    cent, counts = np.zeros((len(names), 3)), np.zeros(len(names), np.int64)
    for i, s in enumerate(names):
        vox = np.argwhere(np.isin(m, [lab for _, lab in labels[s]]))
        if len(vox) == 0:
            raise ValueError(f"region {s} has no voxels in {mask_path}")
        world = vox @ aff[:3, :3].T + aff[:3, 3]
        cent[i] = world.mean(0); counts[i] = len(vox)
    D = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
    meta = {'inputs': {p: sha256(p) for p in (mask_path, csv_path, npz_path)},
            'units': f"affine units (header: {img.header.get_xyzt_units()[0]}); voxel size "
                     f"{[round(float(z), 4) for z in img.header.get_zooms()[:3]]}",
            'recipe': 'centroid = mean world coordinate of all voxels with the region labels (IB = IB_R + IB_L union); '
                      'D = Euclidean distance between centroids'}
    return {'regions': np.array(names), 'labels_json': json.dumps(labels), 'voxel_counts': counts,
            'centroids': cent, 'D': D, 'affine': aff, 'meta_json': json.dumps(meta)}


def load_distances(path=OUT, names=None):
    """(D, regions) from the frozen centroid file; checks the region order if names is given."""
    z = np.load(path, allow_pickle=False)
    regions = [str(x) for x in z['regions']]
    if names is not None and list(names) != regions:
        raise ValueError("centroid file region order does not match the dataset")
    return np.asarray(z['D'], np.float64), regions


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', default=MASK)
    ap.add_argument('--csv', default=CSV)
    ap.add_argument('--npz', default=NPZ)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    out = build(a.mask, a.csv, a.npz)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez_compressed(a.out, **out)
    D = out['D']; off = D[~np.eye(len(D), dtype=bool)]
    print(f"{len(D)} regions; voxels per region {out['voxel_counts'].min()}-{out['voxel_counts'].max()}; "
          f"off-diagonal D {off.min():.1f}-{off.max():.1f}, median {np.median(off):.1f} ({json.loads(out['meta_json'])['units']})")
    print(f"written {a.out} -- add it and build_distances.py to hashes.sha256 before any M5 run")
