import yaml
import logging
from tqdm import tqdm
import multiprocessing
import os
import time
import mne
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def __init__(self, config_yaml_path):
        """
        Initialize the Preprocessor with configuration from a YAML file.    
        """
        with open(config_yaml_path, 'r', encoding='UTF8') as file:
            config = yaml.safe_load(file)

        # Store settings from config
        self.input_dir = config['input_dir']
        self.output_dir = config['output_dir']
        self.lowcut_hz = config['lowcut_hz']
        self.highcut_hz = config['highcut_hz']
        self.notch_hz = config['notch_hz']
        self.filter_order = config['filter_order']
        self.resample_sfreq = config['resample_sfreq']
        self.num_workers = config['num_workers']
        self.channels_dict = config['channels_dict']
        self.channels_to_process = config['channels_to_process']

        
        # Build the arguments for mne.pick_types
        # This creates a dict like {'eeg': True, 'ecg': True}
        self.picks_args = {
            ch_type: included 
            for ch_type, included in self.channels_to_process.items()
            if included
        }

        if not self.picks_args:
            raise ValueError("Config file does not specify any channel types to process.")

        logging.info(f"Will process channel types: {list(self.picks_args.keys())}")

    def get_input_output_file_pairs(self):
        """
        Generate input and output file path pairs for processing.
        
        Returns:
            list of tuples: Each tuple contains (input_file_path, output_file_path)
        """
        search_pattern = os.path.join(self.input_dir, "*.edf")
        input_files = glob.glob(search_pattern)
        file_pairs = []

        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        for i, input_file in enumerate(input_files):
            filename = os.path.basename(input_file)
            output_file = os.path.join(self.output_dir, str(i), filename)
            output_file = output_file.replace('.EDF', '_processed.fif').replace('.edf', '_processed.fif')
            os.makedirs(os.path.join(self.output_dir, str(i)), exist_ok=True)
            file_pairs.append((input_file, output_file))
        
        return file_pairs

    def process_single_file(self, task):
        """
        Worker function to load, process, and save a single EEG file.
        
        Args:
            task (tuple): A tuple containing (input_file_path, output_file_path)
        """
        input_file, output_file = task
        
        try:
            # 1. Load file header without data (memory efficient)
            # We set verbose='DEBUG' to hide excessive MNE output
            raw = mne.io.read_raw_edf(input_file, preload=False, verbose='DEBUG')
            
            # 2. Set Channel Types based on config
            file_channels = set(raw.ch_names)
            
            # Find which channels from our config dict are in this file
            known_channels_in_file = file_channels.intersection(self.channels_dict.keys())
            
            # Create the dictionary for MNE
            ch_types_dict = {
                ch: self.channels_dict[ch] 
                for ch in known_channels_in_file
            }
            
            if ch_types_dict:
                raw.set_channel_types(ch_types_dict, verbose='DEBUG')

            # 3. Get picks for processing
            # We use the picks_args we built in __init__
            # This selects channels based on TYPE (e.g., all 'eeg' and 'ecg')
            picks_to_process = mne.pick_types(raw.info, **self.picks_args)
            
            if len(picks_to_process) == 0:
                # No channels to process in this file
                return (input_file, "Skipped: No channels to process")

            # 4. Load data for the selected channels and process
            raw.load_data()

            # Band-pass filter
            raw.filter(
                l_freq=self.lowcut_hz, 
                h_freq=self.highcut_hz, 
                picks=picks_to_process, 
                filter_length='auto', 
                l_trans_bandwidth='auto', 
                h_trans_bandwidth='auto',
                fir_design='firwin',
                verbose='DEBUG'
            )
            
            # Notch filter
            raw.notch_filter(
                freqs=self.notch_hz, 
                picks=picks_to_process,
                filter_length='auto',
                trans_bandwidth=0.5,
                fir_design='firwin',
                verbose='DEBUG'
            )
            
            # Resample (this is applied *only* to the loaded data)
            raw.resample(sfreq=self.resample_sfreq, verbose='DEBUG')
            
            # 6. Save the raw object
            raw.save(output_file, overwrite=True, verbose='DEBUG')
            return (input_file,f"Success")

        except Exception as e:
            # If one file fails, we log it and continue
            logging.error(f"--- FAILED: {os.path.basename(input_file)}, Error: {e} ---")
            return (input_file, f"Failed: {e}")
        
    def run_preprocessing(self):
        """
        Run the preprocessing on all files in parallel.
        """
        start_time = time.time()
        print("=========== BATCH START ===========")
        file_pairs = self.get_input_output_file_pairs()

        # Create a pool of workers and process files in parallel
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = list(tqdm(pool.imap(self.process_single_file, file_pairs), total=len(file_pairs)))
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        success_count = 0
        fail_count = 0
        for file, status in results:
            if status == "Success":
                success_count += 1
            else:
                fail_count += 1

        print("\n=========== BATCH COMPLETE ===========")
        logging.info(f"Preprocessing completed in {elapsed_time:.2f} seconds.")
        
        logging.info(f"Total time: {time.time() - start_time:.2f}s")
        logging.info(f"Successfully processed: {success_count} files")
        logging.info(f"Failed to process: {fail_count} files")
        logging.info(f"Processed files are in '{self.output_dir}'")
        print("=======================================")

if __name__ == "__main__":
    pass
    