import logging
from tqdm import tqdm
import multiprocessing
import os
import time
import mne
import numpy as np
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def __init__(self, input_dir, output_dir, lowcut_hz=1.0, highcut_hz=40.0, notch_hz=50.0, filter_order=4, resample_sfreq=250, num_workers=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.notch_hz = notch_hz
        self.filter_order = filter_order
        self.resample_sfreq = resample_sfreq
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
    
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
            # 1. Load the raw file
            # preload=True loads the data into memory for filtering
            raw = mne.io.read_raw_edf(input_file, preload=True, verbose=False)

            # 2. Apply "Global" (Serial) Processing
            
            # Apply bandpass filter
            raw.filter(l_freq=self.lowcut_hz, h_freq=self.highcut_hz, 
                    fir_design='firwin', verbose=False)
            
            # Apply notch filter to remove power line noise
            raw.notch_filter(freqs=self.notch_hz, verbose=False)

            # You could add other steps here:
            # raw.set_eeg_reference('average', projection=True)
            raw.resample(sfreq=self.resample_sfreq)
            
            # 3. Save the processed file
            # We use raw.save() to save as a new FIF file
            raw.save(output_file, overwrite=True)
            
            return (input_file, "Success")

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