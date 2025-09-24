import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        # Use dot notation instead of dict indexing
        self.height = opt.height
        self.width = opt.width
        self.latent_dim = opt.latent_dim
        self.n_classes = opt.n_classes

        self.output_dim = self.height * self.width  # 30 * 50 = 1500

        # Label embedding
        self.label_emb = nn.Embedding(self.n_classes, self.latent_dim)

        # Fully connected network to upscale
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Combine noise + label embedding
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        output = self.model(gen_input)
        return output.view(output.size(0), 1, self.height, self.width)
