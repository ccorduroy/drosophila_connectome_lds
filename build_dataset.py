"""
Connectome-aligned, whole-brain imaging dataset from Turner, Mann, and Clandinin (2021)
Dataset is open-sourced with Github repo for building methods

Inputs (all from the figshare zip):
  ito_responses/ito_<date>_<n>.pkl     per-fly DataFrame: rows = Ito atlas label (0 = background), cols = frames @ 1.2 Hz
  connectome_connectivity/*_computed_20210114.pkl   36x36 hemibrain-derived structural matrices
  Original_Index_panda_full.csv        atlas label -> region name
  (optional) vfb_68_*.nii.gz           atlas mask, used only to weight IB_R/IB_L when merging

Processing (follows the authors' SC-FC repo, with the intended trims applied explicitly):
  1. 1st-order Butterworth high-pass 0.01 Hz (fs = 1.2 Hz), then drop the first 100 frames
  2. Merge IB_R + IB_L into one region 'IB' (EM data is not lateralized for IB), voxel-weighted if mask given
  3. Per-fly trims for recordings with documented artifacts, kept as SEPARATE SEGMENTS (never concatenated)
  4. Region order = row order of the structural matrices (36 regions)

Output: fly_ito36.npz with arrays  seg__<fly>__<k>  (36 x T_k, float32), plus region names and structural matrices
"""
import argparse, glob, os, json
import numpy as np, pandas as pd
from scipy import signal

FS, HP_CUTOFF, START = 1.2, 0.01, 100
# Intended trims from the authors' code (frame ranges kept, after filtering). Everything else: frames [100:T].
TRIMS = {'2018-10-19_1': [(100, 900), (1100, 2000)],   # transient single-frame dropouts near frames 1009-1056
         '2017-11-08_1': [(100, 1900), (2000, 4000)],  # baseline step at ~1955
         '2018-10-20_1': [(100, 1000)]}                # signal lost from ~frame 1010 onward

INC37 = ['AL_R','AOTU_R','ATL_R','ATL_L','AVLP_R','BU_R','BU_L','CAN_R','CRE_R','CRE_L','EB','EPA_R','FB','GOR_R','GOR_L',
         'IB_R','IB_L','ICL_R','LAL_R','LH_R','MB_CA_R','MB_ML_R','MB_ML_L','MB_PED_R','MB_VL_R','NO','PB','PLP_R','PVLP_R',
         'SCL_R','SIP_R','SLP_R','SMP_R','SMP_L','SPS_R','VES_R','WED_R']

display = lambda n: n.replace('_R', '(R)').replace('_L', '(L)').replace('_', '')   # authors' bridge.displayName

def build(ito_dir, cc_dir, csv_path, mask_path=None, out_dir='.'):
    idx = pd.read_csv(csv_path, keep_default_na=False); idx.columns = ['name', 'num', 'i0']
    num_of = idx.set_index('name')['num']

    S = {os.path.basename(f).split('_computed')[0]: pd.read_pickle(f) for f in sorted(glob.glob(cc_dir + '/*_computed_*.pkl'))}
    names = list(S['Connectivity'].index)

    assert all(list(m.index) == names == list(m.columns) for m in S.values())

    # structural name -> functional ito names
    fmap = {s: [n for n in INC37 if display(n) == s] for s in names}
    fmap['IB'] = ['IB_R', 'IB_L']

    assert sorted(sum(fmap.values(), [])) == sorted(INC37), "region mapping does not cover the 37 authors' regions exactly"
   
    w = {'IB_R': 1.0, 'IB_L': 1.0}

    if mask_path:
        import nibabel as nib
        m = np.asarray(np.squeeze(nib.load(mask_path).get_fdata()), 'uint16')
        w = {n: float((m == num_of[n]).sum()) for n in ('IB_R', 'IB_L')}
    
    sos = signal.butter(1, HP_CUTOFF, 'hp', fs=FS, output='sos')
    out, meta = {}, {'regions': names, 'fs': FS, 'ib_weights': w, 'flies': {}}

    for f in sorted(glob.glob(ito_dir + '/ito_*.pkl')):
        fid = os.path.basename(f)[4:-4]
        df = pd.read_pickle(f)
        rows = []

        for s in names:
            fn = fmap[s]
            x = np.stack([df.loc[num_of[n]].values.astype(float) * w.get(n, 1.0) for n in fn]).sum(0) / sum(w.get(n, 1.0) for n in fn)
            rows.append(x)

        X = signal.sosfilt(sos, np.vstack(rows), axis=1)          # filter, THEN trim (as the authors do)
        T = X.shape[1]

        segs = TRIMS.get(fid, [(START, T)])

        for k, (a, b) in enumerate(segs):
            out[f'seg__{fid}__{k}'] = X[:, a:b].astype(np.float32)

        meta['flies'][fid] = {'T_raw': T, 'segments': segs, 'n_frames_kept': int(sum(b - a for a, b in segs))}

    out['regions'] = np.array(names)
    for k, m in S.items(): 
        out['struct__' + k] = m.values.astype(float)

    out['meta_json'] = np.array(json.dumps(meta))

    np.savez_compressed(os.path.join(out_dir, 'fly_ito36.npz'), **out)

    return out, meta, S

def load(npz_path):
    """-> (flies: {fly_id: [segment arrays (36 x T_k)]}, region_names, structural: {name: DataFrame}, meta)"""
    
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z['meta_json'])); names = list(z['regions'])
    flies = {}

    for k in sorted(z.files):
        if k.startswith('seg__'):
            _, fid, i = k.split('__'); flies.setdefault(fid, []).append((int(i), z[k]))

    flies = {fid: [a for _, a in sorted(v, key=lambda t: t[0])] for fid, v in flies.items()}

    S = {k[8:]: pd.DataFrame(z[k], index=names, columns=names) for k in z.files if k.startswith('struct__')}

    return flies, names, S, meta

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--ito_dir', default='data_raw/ito_responses')
    ap.add_argument('--cc_dir', default='data_raw/connectome_connectivity')
    ap.add_argument('--csv', default='data_raw/ito_68_atlas/Original_Index_panda_full.csv')
    ap.add_argument('--mask') 
    ap.add_argument('--out_dir', default='data_processed')

    a = ap.parse_args()

    os.mkdir(a.out_dir) if not os.path.exists(a.out_dir) else None
    
    out, meta, _ = build(a.ito_dir, a.cc_dir, a.csv, a.mask, a.out_dir)

    print(json.dumps({k: v['n_frames_kept'] for k, v in meta['flies'].items()}, indent=1)); print('IB weights', meta['ib_weights'])