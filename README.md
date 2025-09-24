# CSI4Free: GAN-Augmented mmWave CSI for Improved Pose Classification

This repository contains the code for the paper:

**CSI4Free: GAN-Augmented mmWave CSI for Improved Pose Classification**  
*Nabeel Nisar Bhat; Rafael Berkvens; Jeroen Famaey.*  
[Link to Paper](https://ieeexplore.ieee.org/document/10646223)

In this work, we show how to perform **stable training of GANs** with mmWave CSI. Using **WGAN**, we can generate an arbitrarily large dataset to augment limited real-world CSI data for **improved pose classification**.

A subset of the GAN-generated dataset can be accessed here:  
[Zenodo Dataset](https://zenodo.org/records/10702154)

---

## Dataset

Place your CSI dataset in the `data/` folder:

data/
├── data_mm.pth # CSI data tensor
└── labels.pth # Corresponding labels

## Features

- Stable GAN Training with **Wasserstein GAN + Gradient Penalty (WGAN-GP)**  
- Conditional GAN (cWGAN) for **class-specific synthetic CSI samples**  
- Code:
  - Loading datasets  
  - Training GANs  
  - Saving generated CSI data  
  - Tracking generator/discriminator losses  
- Dataset augmentation for downstream **pose classification**


## Repository Structure

CSI4Free/
│
├── train.py # Main training script
├── models/ # Model definitions
│ ├── generator.py
│ └── discriminator.py
├── utils/ # Helper functions
│ └── data_loader.py # For loading CSI data
├── data/ # Place CSI dataset files here
│ ├── data_mm.pth
│ └── labels.pth
├── outputs/ # Generated data & losses (created after training)
└── README.md


**Optional Arguments:**
Argument	Default	Description
--n_epochs	40000	Number of training epochs
--batch_size	64	Training batch size
--lr	0.0002	Learning rate for generator and discriminator
--latent_dim	100	Dimension of noise vector
--n_critic	5	Number of discriminator updates per generator update



**Citation**
@inproceedings{bhat2024csi4free,
  title={CSI4Free: GAN-Augmented mmWave CSI for Improved Pose Classification},
  author={Bhat, Nabeel Nisar and Berkvens, Rafael and Famaey, Jeroen},
  booktitle={2024 IEEE 4th International Symposium on Joint Communications \& Sensing (JC\&S)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
