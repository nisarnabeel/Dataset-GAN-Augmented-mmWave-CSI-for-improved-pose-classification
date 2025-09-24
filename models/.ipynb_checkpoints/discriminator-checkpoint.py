import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # Use dot notation instead of dict indexing
        self.height = opt.height
        self.width = opt.width
        self.n_classes = opt.n_classes

        self.input_dim = self.height * self.width  # 30 * 50 = 1500

        # Embedding for labels
        self.label_embedding = nn.Embedding(self.n_classes, self.input_dim)

        # Model for joint [CSI + label] input
        self.model = nn.Sequential(
            nn.Linear(self.input_dim * 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)  # output single scalar
        )

    def forward(self, data, labels):
        # Flatten the input data [batch, 1, 30, 50] -> [batch, 1500]
        data_flat = data.view(data.size(0), -1)

        # Get label embedding
        label_emb = self.label_embedding(labels)

        # Concatenate data + label
        combined_input = torch.cat((data_flat, label_emb), -1)

        # Pass through the discriminator network
        validity = self.model(combined_input)
        return validity
