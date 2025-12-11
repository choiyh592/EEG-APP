Extracting LUNA Embeddings

Berkay Doner, Thorir Mar Ingolfsson, Luca Benini, Yawei Li. LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis. ICML 2025.

```bibtex
@inproceedings{
doner2025luna,
title={{LUNA}: Efficient and Topology-Agnostic Foundation Model for {EEG} Signal Analysis},
author={Berkay D{\"o}ner and Thorir Mar Ingolfsson and Luca Benini and Yawei Li},
booktitle={1st ICML Workshop on Foundation Models for Structured Data},
year={2025},
url={https://openreview.net/forum?id=TWaw5qtKQf}
}
```

WIP

Run extraction

```bash
python extract_embeddings.py \
    --hdf5_path YOUR_HDF5_PATH \
    --batch_size 16 \
    --num_workers 2 \
    --location_emb_path YOUR_LOCATION_EMB_PATH \
    --output_path  YOUR_OUTPUT_PATH \
    --safetensor_path LUNA_base.safetensors \
    --window_size 1600 \
    --use_gpu \
    --variant base \
    --montage_bipolar
```