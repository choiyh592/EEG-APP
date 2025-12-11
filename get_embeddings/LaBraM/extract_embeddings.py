# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Modified for Embedding Extraction with Custom HDF5 Support
# ---------------------------------------------------------

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import h5py # Added for HDF5 support
import pandas as pd  # Added pandas

from pathlib import Path
from collections import OrderedDict
from timm.models import create_model
from torch.utils.data import Dataset

from modeling_finetune import labram_base_patch200_200

import utils

import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class EEGHDF5Dataset(Dataset):
    """
    Custom Dataset for HDF5 EEG data.
    
    Args:
        hdf5_path (str): Path to .h5 file.
        window_size (int): The total duration to load (e.g., 2000 for 10s).
        stride (int): How much to slide the window (e.g., 2000 for non-overlapping).
        segment_size (int): The size of the internal chunks (e.g., 200 for 1s).
                            The output shape will be (Channels, window/segment, segment_size).
    """
    def __init__(self, hdf5_path, window_size=1600, stride=None, segment_size=200):
        self.hdf5_path = hdf5_path
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.segment_size = segment_size
        
        # Sanity check: The big window must be divisible by the small segment
        if self.window_size % self.segment_size != 0:
            raise ValueError(f"window_size ({window_size}) must be divisible by segment_size ({segment_size})")
        
        self.num_segments = self.window_size // self.segment_size
        
        # stores (group_name, start_idx, total_timepoints)
        self.samples = [] 
        self.n_channels = None
        
        with h5py.File(hdf5_path, 'r') as f:
            for group_name in f.keys():
                if 'eeg' in f[group_name]:
                    dset = f[group_name]['eeg']
                    
                    if self.n_channels is None:
                        self.n_channels = dset.shape[0]
                    
                    time_points = dset.shape[1] 
                    
                    if time_points >= self.window_size:
                        n_windows = (time_points - self.window_size) // self.stride + 1
                        for i in range(n_windows):
                            start_idx = i * self.stride
                            self.samples.append((group_name, start_idx, time_points))
                    else:
                        print(f"Warning: Recording {group_name} too short ({time_points}) for window {window_size}")

        print(f"HDF5 Dataset initialized: {len(self.samples)} samples. ")
        print(f"Output Shape per sample: ({self.n_channels}, {self.num_segments}, {self.segment_size})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        '''
        Returns:
            data: [Channels, Num_Segments, Segment_Size]
            label: 0 (dummy)
        '''
        group_name, start_idx, _ = self.samples[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # 1. Read the full "Big Window" (e.g., 60 seconds)
            # Data is (Channels, Time)
            data = f[group_name]['eeg'][:, start_idx : start_idx + self.window_size]
            
            # 2. Convert to float32
            data = data.astype(np.float32)
            
            # 3. Reshape into segments
            # Current shape: (Channels, Window_Size) -> e.g. (Channels, 12000)
            # Target shape:  (Channels, Num_Segments, Segment_Size) -> e.g. (Channels, 6, 200)
            C, T = data.shape
            data = data.reshape(C, self.num_segments, self.segment_size)
            
            # 4. Convert to Tensor
            data = torch.from_numpy(data) 

        return data, 0
    
    def get_sample_info(self, idx):
        group_name, start_idx, total_timepoints = self.samples[idx]
        return {
            'group_name': group_name,
            'start_index': start_idx,
            'end_index': start_idx + self.window_size,
            'total_timepoints': total_timepoints,
            'window_size': self.window_size,
            'num_segments': self.num_segments # Metadata updated
        }
    
    def get_all_metadata(self):
        df = pd.DataFrame(self.samples, columns=['group_name', 'start_index', 'total_timepoints'])
        df['end_index'] = df['start_index'] + self.window_size
        df['window_size'] = self.window_size
        df['num_segments'] = self.num_segments
        df['dataset_idx'] = df.index 
        return df

def get_args():
    parser = argparse.ArgumentParser('LaBraM embedding extraction script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    # Epochs arg kept just to avoid breaking other utils if they check it, though unused
    parser.add_argument('--epochs', default=1, type=int) 
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size (window length)')
    
    parser.add_argument('--return_all_tokens', action='store_true',
                        help='Whether to return all tokens or just the class token')
    parser.set_defaults(return_all_tokens=False)
    parser.add_argument('--return_patch_tokens', action='store_true',
                        help='Whether to return patch tokens along with class token')
    parser.set_defaults(return_patch_tokens=False)

    # ADDED: num_channels to fix dimension mismatch
    parser.add_argument('--num_channels', default=23, type=int,
                        help='Number of EEG channels (acts as image height)')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Checkpoint params
    parser.add_argument('--finetune', default='', required=True,
                        help='path to checkpoint to load weights from')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='', required=True,
                        help='path where to save embeddings')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV')
    
    # Custom HDF5 args
    parser.add_argument('--hdf5_path', default=None, type=str,
                        help='Path to HDF5 dataset file. If set, overrides standard datasets.')
    parser.add_argument('--stride', default=None, type=int,
                        help='Stride for sliding window in HDF5 dataset')

    return parser.parse_args()

def get_models(args):
    # Set num_classes to 0 to remove the head and get embeddings
    # Pass img_size to ensure pos_embed is initialized with correct channel count (Height)
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=0, # Force 0 for embedding extraction
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
        EEG_size=args.input_size,
    )
    return model

def get_dataset(args):
    # Use HDF5 dataset if path is provided
    if args.hdf5_path:
        print(f"Loading custom HDF5 dataset from {args.hdf5_path}")
        # We treat the single HDF5 file as the 'test' set for extraction purposes
        # If you need train/val split, you can implement logic here to split 'dset'
        full_dataset = EEGHDF5Dataset(
            hdf5_path=args.hdf5_path,
            window_size=args.input_size, # window_size is total input size (window size in seconds * sampling rate)
            segment_size=args.patch_size,
            stride=args.stride
        )
        return None, full_dataset, None

    # Fallback to original logic
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("path/to/TUAB")
        args.nb_classes = 1 
    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset("path/to/TUEV")
        args.nb_classes = 6
    return train_dataset, test_dataset, val_dataset

@torch.no_grad()
def extract(data_loader, model, device, return_all_tokens=False, return_patch_tokens = False, ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 
                                                                                                          'C4',  'P3',  'P4', 'O1', 'O2', 
                                                                                                          'F7',  'F8',  'T3', 'T4', 'T5', 
                                                                                                          'T6',  'A1',  'A2', 'FZ', 'CZ', 
                                                                                                          'PZ',  'T1',  'T2']):
    model.eval()
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)

    all_embeddings = []
    all_labels = []

    print(f"Starting extraction on {len(data_loader)} batches...")
    
    for i, batch in enumerate(data_loader):
        # Depending on dataset implementation, batch might be (samples, targets)
        # Our EEGHDF5Dataset returns (data, label)
        images = batch[0]
        target = batch[-1]
        
        images = images.to(device, non_blocking=True)
        
        # compute output
        output = model(images, input_chans=input_chans, return_all_tokens=return_all_tokens, return_patch_tokens=return_patch_tokens)
        
        all_embeddings.append(output.cpu().numpy())
        all_labels.append(target.numpy())

        if i % 50 == 0:
            print(f"Processed {i}/{len(data_loader)} batches")

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return embeddings, labels

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    model = get_models(args)
    
    patch_size = model.patch_size
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    dataset_train, dataset_test, dataset_val = get_dataset(args)
    
    if dataset_test is not None and hasattr(dataset_test, 'n_channels') and dataset_test.n_channels is not None:
         if args.num_channels != dataset_test.n_channels:
             print(f"Updating num_channels from {args.num_channels} to {dataset_test.n_channels} based on dataset.")
             args.num_channels = dataset_test.n_channels

    # Prepare DataLoaders
    # We use SequentialSampler for extraction to keep order deterministic
    
    if dataset_val is not None:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    
    data_loader_test = None
    if dataset_test is not None:
        # Force usage of a single test dataset
        if isinstance(dataset_test, list):
            print("Warning: Multiple test datasets found. Using the first one only.")
            dataset_test = dataset_test[0]
            
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    # Load Checkpoint
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        
        # Filter logic similar to original script
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        # # Remove head keys if they exist in checkpoint
        # for k in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias']:
        #     if k in checkpoint_model:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
                
        # Remap norm to fc_norm if necessary
        if args.use_mean_pooling:
            if 'norm.weight' in checkpoint_model and 'fc_norm.weight' not in checkpoint_model:
                print("Remapping 'norm.weight' to 'fc_norm.weight'")
                checkpoint_model['fc_norm.weight'] = checkpoint_model['norm.weight']
                checkpoint_model['fc_norm.bias'] = checkpoint_model['norm.bias']

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)
    
    # Run Extraction
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Model created. Feature dim: {model.num_features}")

    # 1. Extract Validation
    if dataset_val is not None:
        print("Extracting Validation Set...")
        embeddings, labels = extract(data_loader_val, model, device, return_all_tokens=args.return_all_tokens, return_patch_tokens=args.return_patch_tokens)
        np.save(os.path.join(args.output_dir, 'val_embeddings.npy'), embeddings)
        np.save(os.path.join(args.output_dir, 'val_labels.npy'), labels)
        print(f"Saved validation embeddings: {embeddings.shape}")

    # 2. Extract Test (Single)
    if data_loader_test is not None:
        print("Extracting Test Set...")
        embeddings, labels = extract(data_loader_test, model, device, return_all_tokens=args.return_all_tokens, return_patch_tokens=args.return_patch_tokens)
        np.save(os.path.join(args.output_dir, 'test_embeddings.npy'), embeddings)
        np.save(os.path.join(args.output_dir, 'test_labels.npy'), labels)
        print(f"Saved test embeddings: {embeddings.shape}")

    df = dataset_test.get_all_metadata()
    df.to_csv(os.path.join(args.output_dir, 'test_metadata.csv'), index=False)
    print("Saved test metadata CSV.")

    print("Done!")

if __name__ == '__main__':
    opts = get_args()
    main(opts)