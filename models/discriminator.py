import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.height = opt['height']   # was opt.height
        self.width = opt['width']     # was opt.width
        self.channels = opt['channels']
        self.n_classes = opt['n_classes']

        self.label_embedding = nn.Embedding(self.n_classes, self.n_classes)

        self.model = nn.Sequential(
            nn.Linear(self.height * self.width + self.n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat(
            (img.view(img.size(0), -1), self.label_embedding(labels)), -1
        )
        validity = self.model(d_in)
        return validity
