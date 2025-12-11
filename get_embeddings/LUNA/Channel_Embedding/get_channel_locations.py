import mne
import numpy as np
import torch

CHANNEL_LIST = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'
]
BIPOLAR_CHANNEL_LIST = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "A1-T3",
    "T4-A2"
]

def get_channel_locations(channel_names):
    if "-" in channel_names[0]:
        names = list(set([part for ch in channel_names for part in ch.split('-')]))
    else:
        names = channel_names
        
    ch_types = ['eeg'] * len(names)
    info = mne.create_info(ch_names=names, sfreq=256, ch_types=ch_types)
    
    try:
        info.set_montage(
            mne.channels.make_standard_montage("standard_1005"),
            match_case=False,
            # match_alias=aliases
        )
    except ValueError as e:
        print(f"Montage Error: {e}")

    locs = []

    montage_pos = info.get_montage().get_positions()['ch_pos']
    
    for name in channel_names:
        if name in BIPOLAR_CHANNEL_LIST:
            electrode1, electrode2 = name.split('-')
            loc1 = info.get_montage().get_positions()['ch_pos'][electrode1]
            loc2 = info.get_montage().get_positions()['ch_pos'][electrode2]
            locs.append(((loc1 + loc2) / 2))
        elif name in CHANNEL_LIST:
            locs.append(montage_pos[name])
            
    return np.array(locs)

if __name__ == "__main__":
    target_channels = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'
    ]
    target_channels_bipolar = BIPOLAR_CHANNEL_LIST

    save_location = '/home/yhchoi/EEG_Data/Embeddings_LUNA/channel_locations.pt'
    bipolar = True
    if bipolar:
        target_channels = target_channels_bipolar
        save_location = '/home/yhchoi/EEG_Data/Embeddings_LUNA/channel_locations_bipolar.pt'

    locations = get_channel_locations(target_channels)
    locations_tensor = torch.from_numpy(locations).float()
    torch.save(locations_tensor, save_location)
    print(f"Extracted locations shape: {locations_tensor.shape}")
    print("First 3 locations:\n", locations_tensor[:3])