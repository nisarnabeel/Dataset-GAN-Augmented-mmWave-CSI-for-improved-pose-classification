import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.height = opt['height']   # was opt.height
        self.width = opt['width']     # was opt.width
        self.channels = opt['channels']
        self.latent_dim = opt['latent_dim']
        self.n_classes = opt['n_classes']

        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 2048),
            nn.Linear(2048, self.height * self.width),
            nn.Tanh()
        )

    def forward(self, z, labels):
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.channels, self.height, self.width)
        return img
