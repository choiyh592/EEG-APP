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

from pathlib import Path
from hdf5_utils.utils import h5Dataset
from hdf5_utils.utils import process_fif

savePath = Path('/home/yhchoi/EEG_Data/HDF5/251202_LUNA_Bipolar')
rawDataPath = Path('/home/yhchoi/EEG_Data/Preprocessed/251202_LUNA_Bipolar')
group = rawDataPath.glob('*.fif')

# standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
standard_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'T3-C3', 'C3-CZ', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'CZ-C4', 'C4-T4', 'A1-T3', 'T4-A2'] # Bipolar montage

# preprocessing parameters (disabled)
l_freq = 0
h_freq = 0
rsfreq = 0

# channel number * rsfreq
chunks = (len(standard_channels), rsfreq)

dataset = h5Dataset(savePath, 'dataset')
for cntFile in group:
    print(f'processing {cntFile.name}')
    eegData, chOrder = process_fif(cntFile)
    chOrder = [s.upper() for s in chOrder]
    eegData = eegData[:, :-10*rsfreq] # remove last 10 seconds to avoid potential artifacts
    grp = dataset.addGroup(grpName=cntFile.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)

dataset.save()
