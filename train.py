import os
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import argparse
import yaml

from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset_loader import dataset_loader

# ------------------
# Arguments / Config
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
args = parser.parse_args()

# Load config
with open(args.config) as f:
    opt = yaml.safe_load(f)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ------------------
# Dataset
# ------------------
train_loader, test_loader, n_classes, input_shape = dataset_loader(
    data_path="data",
    batch_size=32,
    test_size=0.25
)
# Add input dimensions to config
opt['channels'] = input_shape[0]
opt['height'] = input_shape[1]
opt['width'] = input_shape[2]
opt['n_classes'] = 8

# ------------------
# Models
# ------------------
generator = Generator(opt)
discriminator = Discriminator(opt)

if cuda:
    generator.cuda()
    discriminator.cuda()
if cuda and torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

# ------------------
# Optimizers
# ------------------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))

lambda_gp = 10

# ------------------
# Gradient Penalty
# ------------------
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = LongTensor(labels)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# ------------------
# Training Loop
# ------------------
os.makedirs(opt['output_path'], exist_ok=True)
g_loss_list, d_loss_list = [], []
batches_done = 0

for epoch in range(opt['n_epochs']):
    d_l, g_l = 0, 0
    generator.train()
    discriminator.train()
    for i, (imgs, labels) in enumerate(train_loader):
        real_imgs = imgs.type(Tensor)
        labels = labels.type(LongTensor)

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        z = Tensor(np.random.normal(0, 1, (imgs.size(0), opt['latent_dim'])))
        fake_imgs = generator(z, labels)
        real_validity = discriminator(real_imgs, labels)
        fake_validity = discriminator(fake_imgs, labels)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels.data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        if i % opt['n_critic'] == 0:
            optimizer_G.zero_grad()
            z = Tensor(np.random.normal(0, 1, (imgs.size(0), opt['latent_dim'])))
            fake_imgs = generator(z, labels)
            g_loss = -torch.mean(discriminator(fake_imgs, labels))
            g_loss.backward()
            optimizer_G.step()
            batches_done += opt['n_critic']

            # Print
            print(f"[Epoch {epoch}/{opt['n_epochs']}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            d_l += d_loss.item()
            g_l += g_loss.item()

    # Save generated samples every 400 epochs (configurable)
    if epoch % 400 == 0:
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            z = Tensor(np.random.normal(0, 1, (30000, opt['latent_dim'])))
            gen_labels = (torch.rand(30000, 1) * opt['n_classes']).type(LongTensor).squeeze()
            gen_imgs = generator(z, gen_labels)
            torch.save(gen_imgs, os.path.join(opt['output_path'], f"augmented_images_epoch_{epoch+1}.pth"))
            torch.save(gen_labels, os.path.join(opt['output_path'], f"augmented_labels_epoch_{epoch+1}.pth"))

    g_loss_list.append(g_l / len(train_loader))
    d_loss_list.append(d_l / len(train_loader))

np.save(os.path.join(opt['output_path'], 'g_loss.npy'), np.array(g_loss_list))
np.save(os.path.join(opt['output_path'], 'd_loss.npy'), np.array(d_loss_list))
