# CSI4Free: GAN-Augmented mmWave CSI for Improved Pose Classification

This repository contains the code for the paper:

**CSI4Free: GAN-Augmented mmWave CSI for Improved Pose Classification**  
*Nabeel Nisar Bhat; Rafael Berkvens; Jeroen Famaey*  
[Link to Paper](https://ieeexplore.ieee.org/document/10646223)

In this work, we demonstrate **stable GAN training** on mmWave CSI data. Using a **Wasserstein GAN (WGAN)**, we can generate synthetic CSI samples to augment limited real-world datasets, improving performance in **pose classification** tasks.

A subset of the GAN-generated dataset is available here:  
[Zenodo Dataset](https://zenodo.org/records/10702154)

---


![generate_new](https://github.com/user-attachments/assets/fea671eb-3503-4dbf-a5e7-94cd5b5b7964)


## Dataset

Place your CSI dataset in the `data/` folder:


## Features

- **WGAN-GP** for stable training  
- **Conditional GAN (cWGAN)** for class-specific sample generation  
- Modular code for:
  - Loading datasets  
  - Training GANs  
  - Saving generated CSI data  
  - Tracking generator/discriminator losses  
- Easy dataset augmentation for downstream pose classification

---

## Configuration

All dataset and hyperparameter options can be set in the **`config.yaml`** file, including:

# Data settings
Place the data in the dataset folder
data_path: "data/data_mm.pth"
labels_path: "data/labels.pth"

# CSI shape
height: 30
width: 50

# Output
output_path: "outputs/"

Citation
If you use this repository or dataset, please cite:

bibtex
Copy code
@inproceedings{bhat2024csi4free,
  title={CSI4Free: GAN-Augmented mmWave CSI for Improved Pose Classification},
  author={Bhat, Nabeel Nisar and Berkvens, Rafael and Famaey, Jeroen},
  booktitle={2024 IEEE 4th International Symposium on Joint Communications \& Sensing (JC\&S)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
