##############################################################################################################
# Source : LaBraM Repository (https://github.com/935963004/LaBraM)
##############################################################################################################
# @inproceedings{
# jiang2024large,
# title={Large Brain Model for Learning Generic Representations with Tremendous {EEG} Data in {BCI}},
# author={Wei-Bang Jiang and Li-Ming Zhao and Bao-Liang Lu},
# booktitle={The Twelfth International Conference on Learning Representations},
# year={2024},
# url={https://openreview.net/forum?id=QzTpTRVtrP}
# }
##############################################################################################################

import argparse
import mne
from tqdm import tqdm
from pathlib import Path
from hdf5_utils.utils import h5Dataset
from hdf5_utils.utils import process_fif

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EEG Preprocessing Script', add_help=False)
    parser.add_argument('--rawpath', type=Path, required=True, help='Path to raw EEG .fif files')
    parser.add_argument('--savepath', type=Path, required=True, help='Path to save HDF5 Dataset')
    parser.add_argument('--rsfreq', type=int, default=256, help='Resampling frequency')
    parser.add_argument('--std', action='store_true', default=True, help='Use standard 10-20 montage for preprocessing')
    parser.add_argument('--bipolar', action='store_false', help='Use bipolar montage for preprocessing', dest='std')

    args = parser.parse_args()

    savePath = args.savepath
    rawDataPath = args.rawpath
    group = list(rawDataPath.glob('*.fif'))

    mne.set_log_level('ERROR')  # Suppress MNE info logs

    if args.std:
        # standard 10-20 montage channels
        standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
    else:
        # standard 10-20 montage channels
        standard_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'T3-C3', 'C3-CZ', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'CZ-C4', 'C4-T4', 'A1-T3', 'T4-A2'] # Bipolar montage

    # channel number * rsfreq
    chunks = (len(standard_channels), args.rsfreq)

    dataset = h5Dataset(savePath, 'dataset')
    with tqdm(group) as pbar:
        for cntFile in pbar:
            pbar.set_description(f'processing {cntFile.name}')
            eegData, chOrder = process_fif(cntFile)
            chOrder = [s.upper() for s in chOrder]
            eegData = eegData[:, :-10*args.rsfreq] # remove last 10 seconds to avoid potential artifacts
            grp = dataset.addGroup(grpName=cntFile.stem)
            dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    dataset.save()
