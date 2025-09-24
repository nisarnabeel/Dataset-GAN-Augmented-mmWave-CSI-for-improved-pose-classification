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

data/
├── data_mm.pth # CSI data tensor
└── labels.pth # Corresponding labels

yaml
Copy code

- `data_mm.pth` → Tensor of shape `(N, 1, 30, 50)`  
- `labels.pth` → Tensor of shape `(N,)` with integer class labels  
- `N` = Number of samples

---

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

```yaml
dataset: mmWGesture         # Choose dataset: mmWGesture, 5GmmGesture, mmWPose, etc.
epochs: 40000               # Number of training epochs
batch_size: 32
lr: 0.0002                  # Learning rate
latent_dim: 100             # Noise vector dimension
n_critic: 5                 # Discriminator steps per generator step
background: false           # For datasets with background augmentation
This allows easy modification of training parameters without editing the code directly.

Optional Command-Line Arguments
Argument	Default	Description
--n_epochs	40000	Number of training epochs
--batch_size	64	Training batch size
--lr	0.0002	Learning rate for generator and discriminator
--latent_dim	100	Dimension of the latent noise vector
--n_critic	5	Number of discriminator updates per generator update

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
