# [ICIAP 2025] CoLoR-GAN: Continual Few-Shot Learning with Low-Rank Adaptation in Generative Adversarial Networks

Official PyTorch implementation for ICIAP 2025 paper:

**CoLoR-GAN: Continual Few-Shot Learning with Low-Rank Adaptation in Generative Adversarial Networks**  
Munsif Ali, Leonardo Rossi, and Massimo Bertozzi

# Environment
- Python 3.8.11
- PyTorch 2.3.0
- Torchvision 0.18.0
- NVIDIA GeForce A100


# Preparation
## Environment
Before running our code, please set up an environment by running commands below:
```bash
git clone https://github.com/munsifali11/CoLoR-GAN.git
cd CoLoR-GAN
conda env create -f environmenmt.yaml
```
## Dataset
You can download the training datasets (10-shot) and their full dataset for evaluation from LFS-GAN github page.  
We recommend you to extract these training sets to `./data` directory.

| |Sketches|Female|Sunglasses|Male|Babies| 
|--|--|--|--|--|--|
|10-shot|[Download](https://drive.google.com/file/d/1QvvPiY0Br7bS5eFOjAw7fWrDvCdnDeRE/view?usp=drive_link)|[Download](https://drive.google.com/file/d/10C9aBzRF4GW68URfUcm1bw_R5xHr7B-6/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1OWJMQC1RhEkwX9UwJAefVNwHS23EOA15/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1DjlEcs6_W2cg26lbWdyTlsRQHeFTF0xg/view?usp=drive_link)|[Download](https://drive.google.com/file/d/13Y6sdqUx75xJTCZ0f5MYV7IvecMFp3CT/view?usp=drive_link)|
|Full|[Download](https://drive.google.com/file/d/1aM9fe7LUQelLIc09FLUdEy-wdlyJ_boK/view?usp=drive_link)|[Download](https://drive.google.com/file/d/11r6dlaQioXWSwF4Evo7RKt5cfYSIBehP/view?usp=drive_link)|[Download](https://drive.google.com/file/d/19NkyLLI87v92vL_3KqE7EJUZ0bq9ZY0n/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1g-4B5IOvTeGyM6W3655OsFL-o9L5kka_/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1H48T5fdZoqwlQXAmaaI5nSqMDwpeUWgf/view?usp=drive_link)|

Before running `train.py`, please process training datasets to be lmdb.

```bash
python prepare_data.py --out processed_data/{dataset} --size 256 data/{dataset}
```
For example, if you want to process `Sketches` dataset, run the command below:
```bash
python prepare_data.py --out processed_data/Sketches --size 256 data/Sketches
```

# Train
You can train CoLoR-GAN by running:
```bash
python train_colorgan.py --data_path processed_data/{dataset} --ckpt ffhq.pt --exp color-gan \
                --rank 1 --use_act --lora_alpha_conv 0.1 --lora_alpha_fc 4
```

The trained checkpoints are saved to `./checkpoints`

# Evaluation
Before the evaluation of the trained model, you first sample images:
```bash
python generate_colorgan.py --pretrained_ckpt ffhq.pt \
                   --ckpt checkpoints/color-gan/{some_checkpoint_name}.pt \
                   --result_path fake_images/color-gan/{dataset} \
                   --rank 1 --use_act --lora_alpha_conv 0.1 --lora_alpha_fc 4
```

You can measure the generation quality by using `pytorch-fid`.
```
python -m pytorch_fid {real_path} fake_images/color-gan/{dataset} --device cuda
```
The `{real_path}` denotes the path of the full dataset.

You can also measure the generation diversity by running `evaluate_b_lpips.py`.

```
python evaluate_b_lpips.py --real_path processed_data/{dataset} --fake_path fake_images/color-gan/{dataset}
```

# Acknowledgment
This code is based on [lfs-gan](https://github.com/KU-VGI/LFS-GAN), [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [few-shot-gan-adaptation](https://github.com/utkarshojha/few-shot-gan-adaptation), [CelebAHQ-Gender](https://github.com/JJuOn/CelebAHQ-Gender), and [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity).

# BibTex

```bibtex
@InProceedings{10.1007/978-3-032-10192-1_5,
author="Ali, Munsif
and Rossi, Leonardo
and Bertozzi, Massimo",
editor="Rodol{\`a}, Emanuele
and Galasso, Fabio
and Masi, Iacopo",
title="CoLoR-GAN: Continual Few-Shot Learning with Low-Rank Adaptation in Generative Adversarial Networks",
booktitle="Image Analysis and Processing -- ICIAP 2025",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="52--64",
}
```


TODO
