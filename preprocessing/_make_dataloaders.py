import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, hdf5_path, pids, window_size=None, stride=None, transform=None):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file.
            pids (list): List of Patient IDs (strings) to include in this dataset.
            window_size (int, optional): Number of time samples per window. 
                                         If None, returns the full recording (batch_size must be 1).
            stride (int, optional): Step size for sliding window. Defaults to window_size (non-overlapping).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.hdf5_path = hdf5_path
        self.pids = [str(p) for p in pids]
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.transform = transform
        self.h5_file = None # Handle for lazy loading
        
        # Pre-calculate the index map
        # This maps a linear index (0 to N) to a specific (PID, start_sample, end_sample)
        self.samples_index = []
        
        self._build_index()

    def _build_index(self):
        """
        Scans the HDF5 file to calculate how many windows exist for each patient.
        """
        print(f"Indexing dataset for {len(self.pids)} patients...")
        with h5py.File(self.hdf5_path, 'r') as f:
            for pid in self.pids:
                if pid not in f:
                    print(f"Warning: PID {pid} not found in HDF5 file.")
                    continue
                
                # Get shape of signals: (n_channels, n_samples)
                # We only need the number of samples (axis 1)
                n_samples = f[pid]['signals'].shape[1]
                
                if self.window_size is None:
                    # Case: Return full recording
                    self.samples_index.append((pid, 0, n_samples))
                else:
                    # Case: Slice into windows
                    # Generate start indices: 0, stride, 2*stride, ...
                    for start in range(0, n_samples - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        self.samples_index.append((pid, start, end))
        
        print(f"Index built. Total samples: {len(self.samples_index)}")

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        if self.h5_file is None:
            # Open file in read mode. 
            # We do this here (lazy loading) to ensure compatibility with 
            # multiple num_workers in DataLoader.
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        # Retrieve metadata for this specific sample index
        pid, start, end = self.samples_index[idx]
        
        # Read the data from HDF5
        # The data is (Channels, Time). We slice the Time axis.
        # Output shape: (Channels, Window_Size)
        data = self.h5_file[pid]['signals'][:, start:end]
        
        # Convert to Float32 Tensor
        data = torch.from_numpy(data).float()
        
        if self.transform:
            data = self.transform(data)
            
        # You might also want to return the label or PID here depending on your task
        # For autoencoders, we often just return data. 
        # For classification, you'd need a label map (pid -> class_id).
        return data

def prepare_data(hdf5_path, train_pids, test_pids, batch_size=32, window_size=256, stride=None, num_workers=0):
    """
    Utility function to create Train and Test DataLoaders.
    
    Args:
        hdf5_path (str): Path to the HDF5 file.
        train_pids (list): List of strings identifying training patients.
        test_pids (list): List of strings identifying test patients.
        batch_size (int): Batch size.
        window_size (int): Size of the EEG window in samples.
        stride (int): Stride for the sliding window.
        num_workers (int): Number of subprocesses for data loading.
    
    Returns:
        train_loader, test_loader
    """
    
    # 1. Create Datasets
    train_dataset = EEGDataset(
        hdf5_path=hdf5_path,
        pids=train_pids,
        window_size=window_size,
        stride=stride
    )
    
    test_dataset = EEGDataset(
        hdf5_path=hdf5_path,
        pids=test_pids,
        window_size=window_size,
        stride=stride
    )
    
    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

# Example Usage Block (for testing)
if __name__ == "__main__":
    # Mock usage
    # Assuming 'data.h5' exists and has PIDs '001', '002'
    path = "data.h5"
    if os.path.exists(path):
        tr_pids = ['001']
        te_pids = ['002']
        
        train_dl, test_dl = prepare_data(path, tr_pids, te_pids, batch_size=4, window_size=512)
        
        print(f"Train batches: {len(train_dl)}")
        for batch in train_dl:
            print(f"Batch shape: {batch.shape}") # Should be [4, Channels, 512]
            break