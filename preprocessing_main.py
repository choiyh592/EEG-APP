import argparse
from pathlib import Path
from preprocessing import Preprocessor, BipolarPreprocessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EEG Preprocessing Script', add_help=False)
    parser.add_argument('--config', type=Path, required=True, help='Path to preprocessing configuration .yaml file')
    parser.add_argument('--std', action='store_true', default=True, help='Use standard 10-20 montage for preprocessing')
    parser.add_argument('--bipolar', action='store_false', help='Use bipolar montage for preprocessing', dest='std')

    args = parser.parse_args()

    if args.std:
        preprocessor = Preprocessor(args.config)
    else:
        preprocessor = BipolarPreprocessor(args.config)

    preprocessor.run_preprocessing()

