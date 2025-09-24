import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset_loader import load_csi_dataset

# -----------------------------
#  Hyperparameters
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=40000, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="latent space dimensionality")
parser.add_argument("--n_critic", type=int, default=5, help="number of discriminator updates per generator update")
parser.add_argument("--clip_value", type=float, default=0.01, help="weight clipping for WGAN")
parser.add_argument("--data_path", type=str, default="data", help="path to CSI dataset")
opt = parser.parse_args()

# -----------------------------
#  Load Dataset
# -----------------------------
train_loader, test_loader, n_classes, input_shape = load_csi_dataset(opt.data_path, batch_size=opt.batch_size)
opt.n_classes = n_classes
opt.channels = input_shape[0]
opt.height = input_shape[1]
opt.width = input_shape[2]

# -----------------------------
#  Models
# -----------------------------
generator = Generator(opt)
discriminator = Discriminator(opt)

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()

# -----------------------------
#  Optimizers
# -----------------------------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# -----------------------------
#  Training Loop
# -----------------------------
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(train_loader):

        batch_size = imgs.shape[0]

        real_imgs = imgs.type(Tensor)
        labels = labels.type(LongTensor)

        # Train discriminator
        optimizer_D.zero_grad()

        z = Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))
        fake_imgs = generator(z, labels)

        real_validity = discriminator(real_imgs, labels)
        fake_validity = discriminator(fake_imgs.detach(), labels)

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        d_loss.backward()
        optimizer_D.step()

        # Train generator every n_critic steps
        if i % opt.n_critic == 0:
            optimizer_G.zero_grad()
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()

    print(f"[Epoch {epoch}/{opt.n_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
