Extracting LaBraM Embeddings

Weibang Jiang, Liming Zhao, Bao-liang Lu. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI. ICLR 2024.

```bibtex
@inproceedings{
jiang2024large,
title={Large Brain Model for Learning Generic Representations with Tremendous {EEG} Data in {BCI}},
author={Weibang Jiang and Liming Zhao and Bao-liang Lu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=QzTpTRVtrP}
}
```

WIP

Run extraction

```bash
python extract_embeddings.py \
    --model labram_base_patch200_200 \
    --finetune checkpoints/labram-base.pth \
    --output_dir DIR_TO_YOUR_EMBEDDINGS \
    --disable_qkv_bias \
    --disable_rel_pos_bias \
    --dataset CUSTOM \
    --batch_size 64 \
    --input_size 3200 \
    --use_mean_pooling \
    --abs_pos_emb \
    --hdf5_path  PATH_TO_YOUR_EEG_HDF5_FILE 

input size = 3200 max (time embedding size = 16)
```