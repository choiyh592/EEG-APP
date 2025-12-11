import os
import argparse

import h5py
import torch
import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from models.LUNA import LUNA
from safetensors.torch import load_model

class EEGHDF5Dataset(Dataset):
    """
    Custom Dataset for HDF5 EEG data.
    Returns raw EEG snippets of shape (Channels, Timepoints).
    
    Args:
        hdf5_path (str): Path to .h5 file.
        window_size (int): The duration of the snippet to load (e.g., 2000 for 10s).
        stride (int): How much to slide the window to create the next sample.
    """
    def __init__(self, hdf5_path, window_size=1600, stride=None):
        self.hdf5_path = hdf5_path
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        
        # stores (group_name, start_idx, total_timepoints)
        self.samples = [] 
        self.n_channels = None
        
        with h5py.File(hdf5_path, 'r') as f:
            for group_name in f.keys():
                # Ensure we are looking at a group containing 'eeg'
                if 'eeg' in f[group_name]:
                    dset = f[group_name]['eeg']
                    
                    if self.n_channels is None:
                        self.n_channels = dset.shape[0]
                    
                    time_points = dset.shape[1] 
                    
                    # Create sliding window indices
                    if time_points >= self.window_size:
                        n_windows = (time_points - self.window_size) // self.stride + 1
                        for i in range(n_windows):
                            start_idx = i * self.stride
                            self.samples.append((group_name, start_idx, time_points))
                    else:
                        print(f"Warning: Recording {group_name} too short ({time_points}) for window {window_size}")

        print(f"HDF5 Dataset initialized: {len(self.samples)} samples.")
        print(f"Output Shape per sample: ({self.n_channels}, {self.window_size})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        '''
        Returns:
            data: [Channels, Timepoints] -> e.g., (19, 1600)
            label: 0 (dummy)
        '''
        group_name, start_idx, _ = self.samples[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # 1. Read the snippet defined by window_size
            # Data structure in H5 is assumed to be (Channels, Time)
            data = f[group_name]['eeg'][:, start_idx : start_idx + self.window_size]
            
            # 2. Convert to float32
            data = data.astype(np.float32)
            
            # 3. Convert to Tensor
            # Shape is preserved as (Channels, Timepoints)
            data = torch.from_numpy(data) 

        return data, 0
    
    def get_sample_info(self, idx):
        group_name, start_idx, total_timepoints = self.samples[idx]
        return {
            'group_name': group_name,
            'start_index': start_idx,
            'end_index': start_idx + self.window_size,
            'total_timepoints': total_timepoints,
            'window_size': self.window_size
        }
    
    def get_all_metadata(self):
        df = pd.DataFrame(self.samples, columns=['group_name', 'start_index', 'total_timepoints'])
        df['end_index'] = df['start_index'] + self.window_size
        df['window_size'] = self.window_size
        df['dataset_idx'] = df.index 
        return df

@torch.no_grad()
def extract(data_loader, model, device, location_emb_path, nerf_embedding_size):
    model.eval()

    # Load location embeddings
    location_embeddings = torch.load(location_emb_path).to(device)

    all_embeddings = []
    all_labels = []

    print(f"Starting extraction on {len(data_loader)} batches...")
    
    for i, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        # Depending on dataset implementation, batch might be (samples, targets)
        # Our EEGHDF5Dataset returns (data, label)
        images = batch[0]
        target = batch[-1]
        
        images = images.to(device, non_blocking=True)

        batch_size = images.shape[0]
        location_embeddings_batch = location_embeddings.unsqueeze(0).expand(batch_size, -1, -1) # coords: (N, C, 3)
        
        # compute output
        #  forward(self, x_signal, mask, channel_locations, channel_names=None):
        x_embed, x_original = model(images, None, location_embeddings_batch)
        
        # cls = x_embed[:, :, 0]

        all_embeddings.append(x_embed.cpu().numpy())
        all_labels.append(target.numpy())

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return embeddings, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract EEG Embeddings using LUNA model")
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to the input HDF5 file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--location_emb_path', type=str, required=True, help='Path to channel location embeddings')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save extracted embeddings')
    parser.add_argument('--safetensor_path', type=str, required=True, help='Path to pretrained safetensors model')
    parser.add_argument('--window_size', type=int, default=1600, help='Window size for EEG snippets')
    parser.add_argument('--stride', type=int, default=None, help='Stride for sliding window')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--variant', type=str, default='base', choices=['base', 'large', 'huge'], help='LUNA model variant')
    parser.add_argument('--montage_1020', action='store_true', help='Use 10-20 montage channel names')
    parser.set_defaults(montage_1020 = True)
    parser.add_argument('--montage_bipolar', action='store_false', help='Use bipolar montage channel names', dest='montage_1020')

    args = parser.parse_args()

    dataset = EEGHDF5Dataset(hdf5_path=args.hdf5_path, window_size=args.window_size, stride=args.stride)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.montage_1020:
        channel_names = [
            'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'
        ]
    else:
        channel_names = [
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

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    # | Variant | Parameters | (Layers, Queries, Q_size, Hidden_size) | >> depth, num_queries, embed_dim, embed_dim * mlp_ratio
    # | LUNA-Base | 7M  | (8, 4, 64, 256)  |
    # | LUNA-Large | 43M  | (10, 6, 96, 576)  |
    # | LUNA-Huge | 311.4M  | (24, 8, 128, 1024)  |
    LUNA_model_specs_dict = {
        "base": {
            'patch_size': 40,
            'num_queries': 4,
            'embed_dim': 64,
            'depth': 8,
            'num_heads': 2,
            'mlp_ratio': 4.,
            'norm_layer': nn.LayerNorm,
            'drop_path': 0.1,
            'num_classes': 1
        },
        "large": {
            'patch_size': 40,
            'num_queries': 6,
            'embed_dim': 96,
            'depth': 10,
            'num_heads': 2,
            'mlp_ratio': 6.,
            'norm_layer': nn.LayerNorm,
            'drop_path': 0.1,
            'num_classes': 1
        },
        "huge": {
            'patch_size': 40,
            'num_queries': 8,
            'embed_dim': 128,
            'depth': 24,
            'num_heads': 8,
            'mlp_ratio': 8.,
            'norm_layer': nn.LayerNorm,
            'drop_path': 0.1,
            'num_classes': 1
        }
    }

    patch_size = LUNA_model_specs_dict[args.variant]['patch_size']
    assert args.window_size % patch_size == 0, f"Window size {args.window_size} must be divisible by patch size {patch_size}"

    nerf_embedding_size = args.window_size // patch_size

    model = LUNA(**LUNA_model_specs_dict[args.variant])

    model.classifier = nn.Identity()  # Remove classifier for embedding extraction

    model.eval()  # Set model to evaluation mode

    embedding_save_dir = args.output_path
    os.makedirs(embedding_save_dir, exist_ok=True)

    # Load pretrained weights
    load_model(model, args.safetensor_path, strict=False)
    model.to(device)

    embeddings, _ = extract(data_loader, model, device, args.location_emb_path, nerf_embedding_size)

    # Save embeddings
    np.save(os.path.join(embedding_save_dir, 'embeddings.npy'), embeddings)

    # Save metadata
    metadata_df = dataset.get_all_metadata()
    metadata_df.to_csv(os.path.join(embedding_save_dir, 'metadata.csv'), index=False)